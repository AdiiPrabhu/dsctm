#!/usr/bin/env python
"""Single SLURM array-task entry point for the whole campaign.

    python scripts/param/run_task.py --family ablation --index $SLURM_ARRAY_TASK_ID

One entry point rather than one script per family, because the run-directory contract, the
provenance capture, the hash agreement checks and the failure semantics must be identical
everywhere. A per-family script drifts; this does not.

Guarantees:
  * the array index resolves to exactly one prespecified Task, or the job fails loudly
  * the run directory is opened BEFORE training, so a job killed mid-run still leaves
    provenance behind rather than nothing
  * status.json is always written, including on failure, with the failure class
  * a run that ends without its required files is recorded as infrastructure_failed,
    never as completed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from dsctm.campaign import RunDirectory, get_task, plan_digest  # noqa: E402
from dsctm.distributed import (  # noqa: E402
    assert_agrees_across_ranks, cleanup, init_distributed, seed_everything,
)
from dsctm.distributed.errors import fail_together  # noqa: E402


def _results_root() -> Path:
    root = os.environ.get("DSCTM_RESULTS_ROOT")
    if not root:
        print("FATAL: DSCTM_RESULTS_ROOT is unset. Run: source scripts/param/env.sh",
              file=sys.stderr)
        sys.exit(2)
    return Path(root)


def load_dataset(task):
    """Build the dataset this task needs, and return it with its provenance hashes."""
    if task.dataset == "studentlife":
        from dsctm.data.studentlife import build_studentlife
        cache = os.environ.get("DSCTM_SL_CACHE",
                               "artifacts/cache/studentlife_causal_ffill_v2.npz")
        imputation = task.params.get("imputation", "causal_ffill")
        ds = build_studentlife(cache=cache, imputation=imputation)
        return ds, None
    if task.dataset == "daicwoz":
        from dsctm.data.daic import build_daicwoz88
        cache = os.environ.get("DSCTM_DAICWOZ_CACHE", "artifacts/cache/daicwoz_egemaps88")
        return build_daicwoz88(cache_dir=cache)
    if task.dataset == "edaic":
        from dsctm.data.daic import build_daic88
        cache = os.environ.get("DSCTM_EDAIC_CACHE", "artifacts/cache/daic_egemaps88")
        return build_daic88(cache_dir=cache)
    raise KeyError(f"unknown dataset {task.dataset!r}")


def build_model_factory(task, ds):
    """Model constructor for this task, with the ablation overrides applied."""
    from dsctm.models import DMSTCN, DMSTCNConfig
    from dsctm.models.baselines import build_baseline

    if task.model != "dmstcn":
        params = {k: v for k, v in task.params.items() if k in ("d_model", "layers", "hidden")}
        return lambda n: build_baseline(task.model, ds.F, ds.n_classes, seq_len=ds.T)

    overrides = {}
    for key in ("enabled_branches", "ssb", "msb", "lsb", "csag_mode", "csag_nonlinearity",
                "temperature", "use_film", "film_mode", "D", "dropout"):
        if key in task.params:
            value = task.params[key]
            overrides[key] = tuple(value) if isinstance(value, list) else value
    return lambda n: DMSTCN(DMSTCNConfig(input_dim=ds.F, n_classes=ds.n_classes,
                                         n_subjects=n, **overrides))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--precision", default=os.environ.get("DSCTM_PRECISION", "fp16"))
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print the task, touch nothing")
    args = ap.parse_args()

    task = get_task(args.family, args.index)      # IndexError here is intentional and fatal
    digest = plan_digest(args.family)

    if args.dry_run:
        print(json.dumps({"task": task.to_dict(), "plan_digest": digest},
                         indent=2, default=str))
        return 0

    ctx = init_distributed()
    run = RunDirectory(_results_root() / args.family, task, is_main=ctx.is_main)
    status, failure_class = "completed", None

    try:
        # Every rank must agree it is running the same task against the same plan.
        # Disagreement means the ranks are running different experiments and any number
        # they produce is invalid.
        assert_agrees_across_ranks(task.task_id, "task_id")
        assert_agrees_across_ranks(digest, "plan_digest")

        with fail_together("load_dataset", ctx.device):
            ds, manifest = load_dataset(task)

        data_hash = ds.data_version_hash()
        assert_agrees_across_ranks(data_hash, "dataset_hash")

        from dsctm.data.splits import subject_grouped_kfold
        if task.protocol == "subject_grouped_5fold":
            folds, split_manifest = subject_grouped_kfold(ds.subject_id, ds.y, 5, seed=0)
        else:
            folds, split_manifest = None, {"scheme": task.protocol,
                                           "split_of_subject": manifest["split_of_subject"]}
        split_hash = split_manifest.get("split_hash", "official")
        assert_agrees_across_ranks(split_hash, "split_hash")

        resolved = {
            "precision": args.precision,
            "world_size": ctx.world_size,
            "node_count": ctx.node_count,
            "dataset": {"name": ds.dataset, "version": ds.version,
                        "N": int(ds.N), "T": int(ds.T), "F": int(ds.F),
                        "n_classes": int(ds.n_classes)},
            "params": task.params,
        }
        run.open(resolved,
                 dataset_hashes={"data_version_hash": data_hash, "version": ds.version,
                                 "summary": ds.summary()},
                 split_hashes=split_manifest,
                 plan_digest=digest)

        seed_everything(task.seed or 0, ctx)
        build_model = build_model_factory(task, ds)

        from dsctm.train.trainer import headline_cv, train_model, train_select_evaluate
        personalize = task.model == "dmstcn" and task.params.get("use_film", True)

        with fail_together("train", ctx.device):
            if task.protocol == "subject_grouped_5fold":
                result = headline_cv(build_model, ds, folds, _cfg(task, ds), ctx.device,
                                     seed=task.seed or 0, personalize=personalize)
                metrics = {"pooled": result["pooled"],
                           "per_fold_macro_f1": result["per_fold_macro_f1"]}
            else:
                idx = _official_indices(ds, manifest)
                if task.family == "tuning":
                    fit = train_model(build_model, ds, idx["train"], idx["dev"],
                                      _cfg(task, ds), ctx.device, seed=task.seed or 0,
                                      personalize=personalize, ctx=ctx,
                                      precision=args.precision)
                    metrics = {"dev_metrics": fit["val_metrics"],
                               "test_accessed": False, "curve": fit["curve"]}
                else:
                    fit = train_select_evaluate(build_model, ds, idx["train"], idx["dev"],
                                                idx["test"], _cfg(task, ds), ctx.device,
                                                seed=task.seed or 0, personalize=personalize,
                                                ctx=ctx, precision=args.precision)
                    metrics = {"dev_metrics": fit["dev_metrics"],
                               "test_metrics": fit["test_metrics"],
                               "best_epoch": fit["best_epoch"]}

        metrics["task"] = task.to_dict()
        metrics["plan_digest"] = digest
        run.write_metrics(metrics)
        run.write_checkpoint(None, reason="checkpoint retention disabled for array tasks; "
                                          "predictions and metrics are retained")

    except IndexError:
        raise
    except Exception as exc:
        status = "model_failed" if isinstance(exc, (ValueError, RuntimeError)) \
            else "infrastructure_failed"
        failure_class = f"{type(exc).__name__}: {exc}"
        print(f"TASK FAILED [{task.task_id}]: {failure_class}", file=sys.stderr)
        traceback.print_exc()
    finally:
        outcome = run.finalize(status, failure_class=failure_class,
                               world_size=ctx.world_size)
        if ctx.is_main and outcome:
            print(f"\ntask   : {task.task_id}")
            print(f"status : {outcome['status']}")
            print(f"receipt: {outcome['receipt']}")
            if not outcome["contract"]["complete"]:
                print(f"MISSING: {outcome['contract']['missing']}", file=sys.stderr)
        cleanup()

    return 0 if status == "completed" else 1


def _cfg(task, ds):
    base = {"batch_size": int(os.environ.get("DSCTM_BATCH_SIZE", 32)),
            "lr": task.params.get("lr", 3e-4), "lr_min": 1e-6, "weight_decay": 1e-4,
            "max_epochs": 100 if task.dataset == "studentlife" else 40,
            "early_stop_patience": 15 if task.dataset == "studentlife" else 8}
    if task.dataset != "studentlife":
        base["class_weight"] = "balanced"
    return base


def _official_indices(ds, manifest):
    import numpy as np
    split_of = manifest["split_of_subject"]
    return {s: np.array([i for i in range(ds.N)
                         if split_of.get(str(ds.subject_id[i])) == s])
            for s in ("train", "dev", "test")}


if __name__ == "__main__":
    sys.exit(main())
