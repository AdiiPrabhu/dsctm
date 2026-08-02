"""Gate 5 — campaign plan and run-directory contract."""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from dsctm.campaign import (
    FAMILIES, MODELS, REQUIRED_FILES, RunDirectory, Task, build_plan, get_task,
    plan_digest, sbatch_array_spec, summarize,
)
from dsctm.campaign.plan import (
    ABLATION_SEEDS, BRANCH_COMBINATIONS, CONFIRMATION_SEEDS, DILATION_SCHEDULES,
    FUSION_VARIANTS, PERSONALIZATION_VARIANTS, PREPROCESSING_CONDITIONS,
    SEARCH_SPACES, TUNING_TRIALS_PER_MODEL, _grid,
)


# --------------------------------------------------------------------------- #
# Plan determinism and addressability
# --------------------------------------------------------------------------- #
def test_plan_is_deterministic_across_invocations():
    a = [t.task_id for t in build_plan()]
    b = [t.task_id for t in build_plan()]
    assert a == b, "the plan changed between calls; array index N would mean two things"


def test_every_task_id_is_unique():
    tasks = build_plan()
    ids = [t.task_id for t in tasks]
    assert len(set(ids)) == len(ids), "task id collision — the plan is not addressable"


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_family_plans_are_deterministic_and_unique(family):
    ids = [t.task_id for t in build_plan(family)]
    assert len(set(ids)) == len(ids)
    assert ids == [t.task_id for t in build_plan(family)]


def test_plan_digest_is_stable():
    assert plan_digest("ablation") == plan_digest("ablation")
    assert plan_digest("ablation") != plan_digest("tuning-daicwoz")


# --------------------------------------------------------------------------- #
# Array bounds — the silent-drop bug
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_sbatch_array_spec_matches_the_plan_length(family):
    """A wrong --array bound silently drops work and the campaign still 'succeeds'."""
    n = len(build_plan(family))
    spec = sbatch_array_spec(family, throttle=4)
    assert spec == f"0-{n - 1}%4"
    upper = int(spec.split("%")[0].split("-")[1])
    assert upper == n - 1, f"array upper bound {upper} would drop {n - 1 - upper} task(s)"


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_array_index_in_range_resolves(family):
    n = len(build_plan(family))
    for i in (0, n // 2, n - 1):
        assert get_task(family, i).task_id


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_out_of_range_index_is_a_hard_error_not_a_silent_skip(family):
    n = len(build_plan(family))
    with pytest.raises(IndexError, match="outside family"):
        get_task(family, n)
    with pytest.raises(IndexError):
        get_task(family, -1)


def test_unknown_family_is_rejected():
    with pytest.raises(KeyError, match="unknown family"):
        build_plan("does-not-exist")


# --------------------------------------------------------------------------- #
# Fairness: equal tuning budget (tracker E4-01)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", MODELS)
def test_every_model_receives_the_same_tuning_budget(model):
    assert len(_grid(SEARCH_SPACES[model])) == TUNING_TRIALS_PER_MODEL


def test_tuning_family_size_is_models_times_budget():
    tasks = build_plan("tuning-daicwoz")
    assert len(tasks) == len(MODELS) * TUNING_TRIALS_PER_MODEL == 48
    per_model = {}
    for t in tasks:
        per_model[t.model] = per_model.get(t.model, 0) + 1
    assert set(per_model.values()) == {TUNING_TRIALS_PER_MODEL}, (
        f"unequal tuning budgets across models: {per_model}"
    )


def test_tuning_tasks_never_declare_a_test_protocol():
    for t in build_plan("tuning-daicwoz") + build_plan("tuning-studentlife"):
        assert "test" not in t.protocol, (
            f"tuning task {t.task_id} declares protocol {t.protocol!r}; "
            f"test must be inaccessible during search"
        )


def test_a_model_with_a_wrong_budget_is_refused(monkeypatch):
    import dsctm.campaign.plan as plan_mod
    bad = dict(SEARCH_SPACES)
    bad["lstm"] = {"hidden": [64, 128], "lr": [1e-4]}   # 2 trials, not 8
    monkeypatch.setattr(plan_mod, "SEARCH_SPACES", bad)
    with pytest.raises(ValueError, match="equal budget"):
        plan_mod.tuning_tasks("daicwoz")


# --------------------------------------------------------------------------- #
# Confirmation and ablation coverage
# --------------------------------------------------------------------------- #
def test_confirmation_has_ten_seeds_per_model():
    tasks = build_plan("confirm-daicwoz")
    assert len(tasks) == len(MODELS) * len(CONFIRMATION_SEEDS) == 60
    for model in MODELS:
        seeds = sorted(t.seed for t in tasks if t.model == model)
        assert seeds == sorted(CONFIRMATION_SEEDS)


def test_ablation_covers_every_prespecified_family():
    tasks = build_plan("ablation")
    conditions = {t.condition for t in tasks}
    for branches in BRANCH_COMBINATIONS:
        assert "branch_" + "+".join(branches) in conditions
    for name in DILATION_SCHEDULES:
        assert f"dilation_{name}" in conditions
    for v in FUSION_VARIANTS:
        assert f"fusion_{v['name']}" in conditions
    for v in PERSONALIZATION_VARIANTS:
        assert f"personalization_{v['name']}" in conditions
    for c in PREPROCESSING_CONDITIONS:
        assert f"preprocessing_{c}" in conditions


def test_ablation_task_count_is_the_sum_of_its_families():
    expected = len(ABLATION_SEEDS) * (
        len(BRANCH_COMBINATIONS) + len(DILATION_SCHEDULES) + len(FUSION_VARIANTS)
        + len(PERSONALIZATION_VARIANTS) + len(PREPROCESSING_CONDITIONS))
    assert len(build_plan("ablation")) == expected == 78


def test_fusion_ablation_contains_both_csag_variants():
    """The manuscript-faithful gate and the declared deviation must both be measured."""
    conditions = {t.condition for t in build_plan("ablation")}
    assert "fusion_linear_csag" in conditions
    assert "fusion_nonlinear_csag" in conditions


def test_all_five_dilation_schedules_are_planned():
    assert len(DILATION_SCHEDULES) == 5
    assert set(DILATION_SCHEDULES) == {"original", "compressed", "expanded",
                                       "uniform", "duration_aligned"}


def test_dilation_schedules_do_not_hardcode_receptive_fields():
    """RF must be derived from the implementation, never typed (tracker E4-08)."""
    from dsctm.models.blocks import Branch
    for name, sched in DILATION_SCHEDULES.items():
        for branch, dilations in sched.items():
            rf = Branch(16, 3, dilations).theoretical_rf_two_conv()
            assert rf == 1 + 2 * 2 * sum(dilations)
            assert rf > 1


# --------------------------------------------------------------------------- #
# Run-directory contract
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _FakeTask:
    task_id: str = "TEST__task__0001"
    def to_dict(self): return {"task_id": self.task_id}


def _open_run(tmp_path):
    run = RunDirectory(tmp_path, _FakeTask(), is_main=True)
    run.open(resolved_config={"precision": "fp16"},
             dataset_hashes={"data_version_hash": "abc123"},
             split_hashes={"split_hash": "def456"}, plan_digest="deadbeef")
    return run


def test_open_writes_provenance_before_training(tmp_path):
    run = _open_run(tmp_path)
    for name in ("command.txt", "resolved_config.yaml", "environment.json", "git.json",
                 "slurm.json", "hardware.json", "dataset_hashes.json", "split_hashes.json",
                 "status.json", "stdout.log", "stderr.log"):
        assert (run.path / name).exists(), f"{name} missing after open()"
    assert json.loads((run.path / "status.json").read_text())["status"] == "running"


def test_incomplete_run_cannot_be_marked_completed(tmp_path):
    """The rule that makes the contract mean something."""
    run = _open_run(tmp_path)
    outcome = run.finalize("completed")
    assert outcome["status"] == "infrastructure_failed"
    assert not outcome["contract"]["complete"]
    assert "metrics.json" in outcome["contract"]["missing"]
    recorded = json.loads((run.path / "status.json").read_text())
    assert recorded["requested_status"] == "completed"
    assert "contract_violation" in recorded


def test_complete_run_is_accepted(tmp_path):
    run = _open_run(tmp_path)
    run.write_metrics({"macro_f1": 0.42})
    run.write_checkpoint(None, reason="retention disabled")
    (run.path / "predictions.parquet").write_text("stub")
    (run.path / "checkpoint.pt").write_text("stub")
    outcome = run.finalize("completed")
    assert outcome["status"] == "completed", outcome["contract"]["missing"]
    assert outcome["contract"]["complete"]
    assert len(outcome["receipt"]) == 64


def test_waived_file_is_recorded_not_silently_dropped(tmp_path):
    run = _open_run(tmp_path)
    run.write_metrics({"macro_f1": 0.1})
    (run.path / "predictions.parquet").write_text("stub")
    run.write_checkpoint(None, reason="retention disabled for array tasks")
    outcome = run.finalize("completed")
    assert outcome["status"] == "completed"
    recorded = json.loads((run.path / "status.json").read_text())
    assert "checkpoint.pt" in recorded["waivers"]
    assert "retention disabled" in recorded["waivers"]["checkpoint.pt"]


def test_receipt_changes_when_content_changes(tmp_path):
    run = _open_run(tmp_path)
    run.write_metrics({"macro_f1": 0.5})
    (run.path / "predictions.parquet").write_text("a")
    (run.path / "checkpoint.pt").write_text("b")
    first = run.finalize("completed")["receipt"]
    run.write_metrics({"macro_f1": 0.9})
    second = run.finalize("completed")["receipt"]
    assert first != second, "receipt must bind to content"


def test_failure_status_is_preserved(tmp_path):
    run = _open_run(tmp_path)
    outcome = run.finalize("model_failed", failure_class="RuntimeError: boom")
    assert outcome["status"] == "model_failed"
    assert json.loads((run.path / "status.json").read_text())["failure_class"] == \
        "RuntimeError: boom"


def test_required_file_list_is_the_fifteen_from_the_contract():
    assert len(REQUIRED_FILES) == 15


def test_non_main_rank_writes_nothing(tmp_path):
    run = RunDirectory(tmp_path, _FakeTask(), is_main=False)
    run.open({}, {}, {})
    run.write_metrics({"macro_f1": 1.0})
    assert not run.path.exists() or not any(run.path.iterdir())


# --------------------------------------------------------------------------- #
def test_summary_totals_are_self_consistent():
    s = summarize()
    assert s["n_tasks"] == s["unique_task_ids"] == 294
    assert sum(s["by_family"].values()) == s["n_tasks"]
    assert sum(s["by_experiment"].values()) == s["n_tasks"]
