"""Deterministic campaign plan: the full experiment matrix as an indexable task list.

Why this exists. A SLURM array maps `SLURM_ARRAY_TASK_ID` to work. If that mapping lives
in a shell loop or is recomputed per job, two failure modes follow, both silent:

* `--array=0-47` against 50 real tasks quietly drops the last two, and the campaign looks
  complete because every submitted task succeeded.
* A plan that changes between submissions makes task 17 mean different things in different
  jobs, so a resubmitted failure re-runs the wrong experiment.

So the plan is built once, deterministically, from prespecified constants; every task
carries a stable `task_id` derived from its own content; and the array bounds are printed
by `scripts/param/plan.py --sbatch-array` rather than typed by hand.

Ordering is fixed by construction (nested loops over sorted constants). Adding a model or a
seed appends to the end of its family but does not renumber earlier tasks — verified by
`tests/test_campaign_plan.py`.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Prespecified constants. Changing any of these changes the campaign; they are
# recorded in every run directory so a result can be traced to the plan that produced it.
# --------------------------------------------------------------------------- #
MODELS = ("dmstcn", "lstm", "temporal-cnn", "transformer", "itransformer", "timesnet")

TUNING_TRIALS_PER_MODEL = 8      # equal budget for every model (tracker E4-01)
CONFIRMATION_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)   # 10 seeds (tracker E4-14)
ABLATION_SEEDS = (0, 1, 2)

# Model-specific search spaces, 8 trials each. Spaces differ because the architectures
# differ; the BUDGET is identical, which is what fairness requires.
SEARCH_SPACES: dict[str, dict[str, list]] = {
    "dmstcn":       {"D": [64, 128], "dropout": [0.0, 0.2], "lr": [1e-4, 3e-4]},
    "lstm":         {"hidden": [64, 128], "layers": [1, 2], "lr": [1e-4, 3e-4]},
    "temporal-cnn": {"D": [64, 128], "dropout": [0.0, 0.2], "lr": [1e-4, 3e-4]},
    "transformer":  {"d_model": [64, 128], "layers": [1, 2], "lr": [1e-4, 3e-4]},
    "itransformer": {"d_model": [64, 128], "layers": [1, 2], "lr": [1e-4, 3e-4]},
    "timesnet":     {"d_model": [16, 32], "layers": [1, 2], "lr": [1e-4, 3e-4]},
}

# Gate 6 ablation families. Dilation schedules are named here; their receptive fields are
# DERIVED from the implementation at run time, never typed (tracker E4-08).
DILATION_SCHEDULES: dict[str, dict[str, tuple]] = {
    "original":         {"ssb": (1, 2, 4, 8),   "msb": (8, 16, 32, 64),  "lsb": (32, 64, 128, 256)},
    "compressed":       {"ssb": (1, 2, 3, 4),   "msb": (4, 8, 12, 16),   "lsb": (16, 32, 48, 64)},
    "expanded":         {"ssb": (1, 4, 16, 64), "msb": (8, 32, 128, 512), "lsb": (32, 128, 512, 2048)},
    "uniform":          {"ssb": (1, 1, 1, 1),   "msb": (8, 8, 8, 8),     "lsb": (32, 32, 32, 32)},
    "duration_aligned": {"ssb": (1, 2, 4, 8),   "msb": (16, 32, 64, 128), "lsb": (64, 128, 256, 512)},
}

BRANCH_COMBINATIONS = (
    ("ssb",), ("msb",), ("lsb",),
    ("ssb", "msb"), ("ssb", "lsb"), ("msb", "lsb"),
    ("ssb", "msb", "lsb"),
)

FUSION_VARIANTS = (
    {"name": "mean",           "csag_mode": "mean"},
    {"name": "static",         "csag_mode": "static"},
    {"name": "linear_csag",    "csag_mode": "linear_csag"},
    {"name": "nonlinear_csag", "csag_mode": "nonlinear_csag", "csag_nonlinearity": "relu"},
    {"name": "temp_half",      "csag_mode": "linear_csag", "temperature": 5.656854249},
    {"name": "temp_double",    "csag_mode": "linear_csag", "temperature": 22.627416998},
)

PERSONALIZATION_VARIANTS = (
    {"name": "no_film",        "use_film": False},
    {"name": "subject_film",   "use_film": True, "film_mode": "subject"},
    {"name": "global_film",    "use_film": True, "film_mode": "global"},
    {"name": "matched_global", "use_film": True, "film_mode": "global_matched"},
)

PREPROCESSING_CONDITIONS = ("causal_ffill", "zero", "train_mean", "mask_aware_zero")

DATASETS = ("studentlife", "daicwoz")


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Task:
    """One schedulable unit of work. Exactly one SLURM array task."""

    family: str
    experiment_id: str
    dataset: str
    model: str
    condition: str
    protocol: str
    seed: int | None = None
    trial: int | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        payload = json.dumps({"family": self.family, "experiment_id": self.experiment_id,
                              "dataset": self.dataset, "model": self.model,
                              "condition": self.condition, "protocol": self.protocol,
                              "seed": self.seed, "trial": self.trial,
                              "params": self.params}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def task_id(self) -> str:
        """Stable, human-readable, collision-resistant."""
        bits = [self.experiment_id, self.dataset, self.model, self.condition]
        if self.trial is not None:
            bits.append(f"t{self.trial}")
        if self.seed is not None:
            bits.append(f"s{self.seed}")
        return "__".join(b.replace("/", "-") for b in bits) + f"__{self.config_hash}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["task_id"] = self.task_id
        d["config_hash"] = self.config_hash
        return d


def _grid(space: dict[str, list]) -> list[dict[str, Any]]:
    """Deterministic full factorial over sorted keys."""
    keys = sorted(space)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(space[k] for k in keys))]


# --------------------------------------------------------------------------- #
# Family builders
# --------------------------------------------------------------------------- #
def tuning_tasks(dataset: str) -> list[Task]:
    """EXP-2.2/2.3 — equal-budget development search. Test is never loaded."""
    tasks = []
    for model in MODELS:
        grid = _grid(SEARCH_SPACES[model])
        if len(grid) != TUNING_TRIALS_PER_MODEL:
            raise ValueError(
                f"search space for {model!r} yields {len(grid)} trials, "
                f"but the prespecified equal budget is {TUNING_TRIALS_PER_MODEL}. "
                f"Fairness (tracker E4-01) requires identical budgets."
            )
        for trial, params in enumerate(grid):
            tasks.append(Task(
                family="tuning", experiment_id="EXP-2.2", dataset=dataset, model=model,
                condition=f"{model}_search", protocol="official_train_dev_search",
                trial=trial, seed=0, params=params,
            ))
    return tasks


def confirmation_tasks(dataset: str) -> list[Task]:
    """EXP-4.1/4.2 — frozen configuration, 10 seeds, test evaluated once per seed."""
    protocol = ("subject_grouped_5fold" if dataset == "studentlife"
                else "official_dev_selected_test_once")
    return [
        Task(family="confirm", experiment_id="EXP-4.1" if dataset == "studentlife" else "EXP-4.2",
             dataset=dataset, model=model, condition=f"{model}_confirm",
             protocol=protocol, seed=seed,
             params={"config_source": "frozen_from_tuning"})
        for model in MODELS for seed in CONFIRMATION_SEEDS
    ]


def ablation_tasks(dataset: str = "studentlife") -> list[Task]:
    """Gate 6 families. D-MSTCN only — an ablation of the proposed model."""
    tasks: list[Task] = []
    for branches in BRANCH_COMBINATIONS:
        for seed in ABLATION_SEEDS:
            tasks.append(Task(
                family="ablation", experiment_id="EXP-5.1", dataset=dataset, model="dmstcn",
                condition="branch_" + "+".join(branches), protocol="subject_grouped_5fold",
                seed=seed, params={"enabled_branches": list(branches)}))
    for name, sched in DILATION_SCHEDULES.items():
        for seed in ABLATION_SEEDS:
            tasks.append(Task(
                family="ablation", experiment_id="EXP-5.3", dataset=dataset, model="dmstcn",
                condition=f"dilation_{name}", protocol="subject_grouped_5fold",
                seed=seed, params={k: list(v) for k, v in sched.items()}))
    for variant in FUSION_VARIANTS:
        for seed in ABLATION_SEEDS:
            params = {k: v for k, v in variant.items() if k != "name"}
            tasks.append(Task(
                family="ablation", experiment_id="EXP-5.2", dataset=dataset, model="dmstcn",
                condition=f"fusion_{variant['name']}", protocol="subject_grouped_5fold",
                seed=seed, params=params))
    for variant in PERSONALIZATION_VARIANTS:
        for seed in ABLATION_SEEDS:
            params = {k: v for k, v in variant.items() if k != "name"}
            tasks.append(Task(
                family="ablation", experiment_id="EXP-5.5", dataset=dataset, model="dmstcn",
                condition=f"personalization_{variant['name']}",
                protocol="subject_grouped_5fold", seed=seed, params=params))
    for condition in PREPROCESSING_CONDITIONS:
        for seed in ABLATION_SEEDS:
            tasks.append(Task(
                family="ablation", experiment_id="EXP-1.3", dataset=dataset, model="dmstcn",
                condition=f"preprocessing_{condition}", protocol="subject_grouped_5fold",
                seed=seed, params={"imputation": condition}))
    return tasks


FAMILIES = {
    "tuning-studentlife":  lambda: tuning_tasks("studentlife"),
    "tuning-daicwoz":      lambda: tuning_tasks("daicwoz"),
    "confirm-studentlife": lambda: confirmation_tasks("studentlife"),
    "confirm-daicwoz":     lambda: confirmation_tasks("daicwoz"),
    "ablation":            lambda: ablation_tasks("studentlife"),
}


def build_plan(family: str | None = None) -> list[Task]:
    """Full plan, or one family. Order is deterministic and stable across invocations."""
    if family is not None:
        if family not in FAMILIES:
            raise KeyError(f"unknown family {family!r}; expected one of {sorted(FAMILIES)}")
        return FAMILIES[family]()
    tasks: list[Task] = []
    for name in sorted(FAMILIES):
        tasks.extend(FAMILIES[name]())
    return tasks


def get_task(family: str, index: int) -> Task:
    """Resolve one SLURM array index. Out-of-range is a hard error, never a silent skip."""
    tasks = build_plan(family)
    if not 0 <= index < len(tasks):
        raise IndexError(
            f"array index {index} is outside family {family!r}, which has {len(tasks)} "
            f"task(s) (valid --array range: 0-{len(tasks) - 1}). "
            f"The sbatch --array bound does not match the plan."
        )
    return tasks[index]


def sbatch_array_spec(family: str, throttle: int = 4) -> str:
    """The exact --array value for this family. Print it; do not type it."""
    return f"0-{len(build_plan(family)) - 1}%{throttle}"


def plan_digest(family: str | None = None) -> str:
    """SHA-256 over the ordered task ids. Recorded per run so plan drift is detectable."""
    ids = [t.task_id for t in build_plan(family)]
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def summarize(family: str | None = None) -> dict[str, Any]:
    tasks = build_plan(family)
    by_family: dict[str, int] = {}
    by_experiment: dict[str, int] = {}
    for t in tasks:
        by_family[t.family] = by_family.get(t.family, 0) + 1
        by_experiment[t.experiment_id] = by_experiment.get(t.experiment_id, 0) + 1
    return {
        "family_filter": family,
        "n_tasks": len(tasks),
        "plan_digest": plan_digest(family),
        "by_family": dict(sorted(by_family.items())),
        "by_experiment": dict(sorted(by_experiment.items())),
        "unique_task_ids": len({t.task_id for t in tasks}),
    }
