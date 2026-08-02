"""Gate 2/3 — multi-process distributed tests over the gloo backend on CPU.

These spawn real processes and real collectives. They validate the distributed LOGIC:
sharding, gathering, coverage, broadcast, rank-0 write discipline, checkpoint resume and
failure propagation.

They do NOT validate NCCL, fp16 numerics, V100 kernels, or any performance figure. Those
require the PARAM `gpu` partition and are Gate 3's hardware half. Results here are
recorded as LOGIC-VERIFIED (CPU/gloo) and never as a hardware gate pass — see
DECISIONS.md D-005.
"""
from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

N_SAMPLES = 47          # DAIC-WOZ official test split size — the duplication hazard
FEATURES = 6
CLASSES = 3


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _toy_model(seed: int = 0, dropout: float = 0.0) -> nn.Module:
    """Toy model.

    ``dropout=0.0`` for the DDP-vs-single-process parity case. Dropout draws its mask
    from the input SHAPE, so a rank holding an (8, F) shard and a reference process
    holding the (16, F) union consume the RNG stream differently and produce different
    masks. That divergence is correct PyTorch behaviour, not a DDP defect — so the parity
    test removes the confound rather than papering over it. The checkpoint-resume case
    keeps dropout on precisely because it must prove RNG state is restored.
    """
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(FEATURES, 16), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(16, CLASSES))


def _toy_batch(n: int, seed: int = 1234):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, FEATURES, generator=g)
    y = torch.randint(0, CLASSES, (n,), generator=g)
    return X, y


# --------------------------------------------------------------------------- #
# Worker entry point — must be module level for the spawn start method
# --------------------------------------------------------------------------- #
def _worker(rank: int, world_size: int, port: int, out_dir: str, case: str) -> None:
    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK="0",
        LOCAL_WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
    )
    os.environ.pop("SLURM_JOB_ID", None)

    from dsctm.distributed import (
        EarlyStopCoordinator, PredictionRecord, RunLogger, assert_agrees_across_ranks,
        assert_replicas_agree, build_records, cleanup, gather_and_validate,
        init_distributed, make_eval_sampler, make_train_sampler, resolve_batch_semantics,
        save_checkpoint, load_checkpoint, seed_everything, state_digest, wrap_ddp,
    )
    from dsctm.distributed.errors import PreflightFailure, RankFailure, fail_together

    result: dict = {"rank": rank}
    ctx = init_distributed(timeout_minutes=2)
    try:
        result["ctx"] = ctx.describe()

        if case == "ddp_one_step_parity":
            seed_everything(7, ctx)
            model = wrap_ddp(_toy_model(7), ctx)
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            X, y = _toy_batch(world_size * 8)
            sem = resolve_batch_semantics(world_size * 8, world_size)
            lo = rank * sem.per_rank_batch_size
            hi = lo + sem.per_rank_batch_size
            model.train()
            torch.manual_seed(99)          # identical dropout masks across ranks
            opt.zero_grad()
            nn.functional.cross_entropy(model(X[lo:hi]), y[lo:hi]).backward()
            opt.step()
            from dsctm.distributed import unwrap
            result["digest"] = state_digest(model)
            result["per_rank_batch"] = sem.per_rank_batch_size
            result["effective_global_batch"] = sem.effective_global_batch
            if rank == 0:
                ref = _toy_model(7)
                ref_opt = torch.optim.SGD(ref.parameters(), lr=0.1)
                ref.train()
                torch.manual_seed(99)
                ref_opt.zero_grad()
                nn.functional.cross_entropy(ref(X), y).backward()
                ref_opt.step()
                # Bitwise equality is the WRONG criterion here. DDP forms the global
                # gradient by all-reducing per-rank means; the reference forms it as one
                # mean over the union. Floating-point addition is not associative, so the
                # two differ in the last ulp or two even when the mathematics is identical.
                # Compare with a declared tolerance instead.
                ddp_flat = torch.cat([p.detach().flatten() for p in unwrap(model).parameters()])
                ref_flat = torch.cat([p.detach().flatten() for p in ref.parameters()])
                result["max_abs_param_diff"] = float((ddp_flat - ref_flat).abs().max())
                result["max_rel_param_diff"] = float(
                    ((ddp_flat - ref_flat).abs() / ref_flat.abs().clamp_min(1e-12)).max())
                result["param_count"] = int(ddp_flat.numel())

        elif case == "eval_coverage":
            X, y = _toy_batch(N_SAMPLES)
            ids = torch.arange(N_SAMPLES)
            ds = TensorDataset(X, y, ids)
            sampler = make_eval_sampler(ds, ctx)
            loader = DataLoader(ds, batch_size=4, sampler=sampler, num_workers=0)
            model = _toy_model(3).eval()
            records: list[PredictionRecord] = []
            with torch.no_grad():
                for xb, yb, ib in loader:
                    records.extend(build_records(
                        ib.tolist(), [f"p{int(i)}" for i in ib], yb.tolist(),
                        model(xb), rank=ctx.rank))
            result["local_count"] = len(records)
            merged, audit = gather_and_validate(records, N_SAMPLES, list(range(N_SAMPLES)))
            result["audit"] = audit
            result["merged_ids"] = [r.sample_id for r in merged]

        elif case == "eval_coverage_detects_padding_duplicates":
            # Deliberately use the STOCK padded sampler to prove the guard fires.
            from torch.utils.data import DistributedSampler
            X, y = _toy_batch(N_SAMPLES)
            ids = torch.arange(N_SAMPLES)
            ds = TensorDataset(X, y, ids)
            sampler = DistributedSampler(ds, num_replicas=ctx.world_size, rank=ctx.rank,
                                         shuffle=False, drop_last=False)
            loader = DataLoader(ds, batch_size=4, sampler=sampler, num_workers=0)
            model = _toy_model(3).eval()
            records = []
            with torch.no_grad():
                for xb, yb, ib in loader:
                    records.extend(build_records(
                        ib.tolist(), [f"p{int(i)}" for i in ib], yb.tolist(),
                        model(xb), rank=ctx.rank))
            try:
                gather_and_validate(records, N_SAMPLES, list(range(N_SAMPLES)))
                result["guard_fired"] = False
            except Exception as exc:
                result["guard_fired"] = True
                result["guard_message"] = str(exc)[:200]

        elif case == "early_stop_broadcast":
            stopper = EarlyStopCoordinator(patience=1, ctx=ctx)
            # Every rank feeds a DIFFERENT score. Only rank 0's may decide.
            decisions = []
            for epoch, base in enumerate([0.5, 0.4, 0.3]):
                score = base + 0.1 * ctx.rank
                d = stopper.step(score, epoch)
                decisions.append({"epoch": d.epoch, "improved": d.improved,
                                  "should_stop": d.should_stop,
                                  "best": round(d.best_score, 6),
                                  "patience": d.patience})
                if d.should_stop:
                    break
            result["decisions"] = decisions
            result["epochs_executed"] = len(decisions)

        elif case == "rank_zero_only_writes":
            logger = RunLogger(Path(out_dir) / "run", ctx, echo=False)
            logger.log(f"hello from rank {ctx.rank}")
            logger.write_json("metrics.json", {"written_by_rank": ctx.rank})
            logger.mark_status("completed", rank=ctx.rank)
            logger.close()
            result["wrote"] = ctx.is_main

        elif case == "checkpoint_resume":
            seed_everything(11, ctx)
            model = wrap_ddp(_toy_model(11, dropout=0.3), ctx)
            opt = torch.optim.SGD(model.parameters(), lr=0.05)
            X, y = _toy_batch(world_size * 8)
            lo, hi = rank * 8, rank * 8 + 8

            def one_step():
                model.train()
                opt.zero_grad()
                nn.functional.cross_entropy(model(X[lo:hi]), y[lo:hi]).backward()
                opt.step()

            one_step()
            ckpt = Path(out_dir) / "ckpt.pt"
            save_checkpoint(ckpt, model=model, optimizer=opt, epoch=1, global_step=1,
                            best_score=0.5, patience=0, ctx=ctx,
                            dataset_hash="d0", split_hash="s0")
            one_step()
            result["uninterrupted_digest"] = state_digest(model)

            resumed = wrap_ddp(_toy_model(999, dropout=0.3), ctx)   # deliberately different init
            resumed_opt = torch.optim.SGD(resumed.parameters(), lr=0.05)
            meta = load_checkpoint(ckpt, model=resumed, optimizer=resumed_opt,
                                   expect_dataset_hash="d0", expect_split_hash="s0")
            resumed.train()
            resumed_opt.zero_grad()
            nn.functional.cross_entropy(resumed(X[lo:hi]), y[lo:hi]).backward()
            resumed_opt.step()
            result["resumed_digest"] = state_digest(resumed)
            result["meta"] = {k: meta[k] for k in ("epoch", "global_step", "best_score")}

        elif case == "checkpoint_rejects_wrong_hash":
            model = _toy_model(1)
            ckpt = Path(out_dir) / f"ckpt_hash_{rank}.pt"
            save_checkpoint(ckpt, model=model, epoch=0, dataset_hash="RIGHT",
                            split_hash="s0", ctx=None)
            try:
                load_checkpoint(ckpt, model=_toy_model(2), expect_dataset_hash="WRONG")
                result["refused"] = False
            except ValueError as exc:
                result["refused"] = True
                result["message"] = str(exc)[:160]

        elif case == "failure_propagates":
            try:
                with fail_together("deliberate", ctx.device):
                    if ctx.rank == 1:
                        raise RuntimeError("simulated rank-1 model failure")
                result["raised"] = False
            except RuntimeError as exc:
                result["raised"] = True
                result["is_rank_failure"] = isinstance(exc, RankFailure)
                result["message"] = str(exc)[:120]

        elif case == "hash_disagreement_is_fatal":
            try:
                assert_agrees_across_ranks(f"split-{ctx.rank}", "split_hash")
                result["refused"] = False
            except PreflightFailure as exc:
                result["refused"] = True
                result["message"] = str(exc)[:160]

        elif case == "hash_agreement_passes":
            assert_agrees_across_ranks("split-identical", "split_hash")
            result["refused"] = False

        elif case == "train_sampler_set_epoch":
            X, y = _toy_batch(40)
            ds = TensorDataset(X, y)
            sampler = make_train_sampler(ds, ctx, shuffle=True, seed=0)
            sampler.set_epoch(0)
            first = list(sampler)
            sampler.set_epoch(1)
            second = list(sampler)
            result["epoch0"] = first
            result["epoch1"] = second
            result["order_changed"] = first != second

        elif case == "replicas_agree":
            seed_everything(5, ctx)
            model = wrap_ddp(_toy_model(5), ctx)
            result["digest"] = assert_replicas_agree(model)

        else:
            raise ValueError(f"unknown case {case!r}")

        result["ok"] = True
    except Exception as exc:  # record then re-raise so spawn surfaces it
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(result, default=str))
        cleanup()
        raise
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(result, default=str))
    cleanup()


def _run(case: str, world_size: int, tmp_path: Path) -> list[dict]:
    out = tmp_path / case
    out.mkdir(parents=True, exist_ok=True)
    mp.spawn(_worker, args=(world_size, _free_port(), str(out), case),
             nprocs=world_size, join=True)
    results = [json.loads((out / f"rank{r}.json").read_text()) for r in range(world_size)]
    assert all(r["ok"] for r in results), [r.get("error") for r in results]
    return results


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
#: Declared tolerance for DDP-vs-single-process parameter parity after one SGD step.
#: Justification: DDP builds the global gradient by all-reducing per-rank means, while the
#: reference builds it as a single mean over the union. Floating-point addition is not
#: associative, so identical mathematics yields results differing by a few ulp in fp32
#: (~1.2e-7 relative). 1e-6 absolute is ~8 ulp at unit scale — tight enough to catch a real
#: sharding or reduction bug (which shifts parameters by 1e-3 or more), loose enough to
#: ignore reduction-order noise. On PARAM this same test re-runs over NCCL, where the
#: reduction tree differs again; the tolerance is unchanged and the criterion still holds.
DDP_PARITY_ATOL = 1e-6


@pytest.mark.parametrize("world_size", [2, 3])
def test_ddp_one_step_matches_single_process_reference(world_size, tmp_path):
    """DDP over N shards must land on the same parameters as one process over the union."""
    results = _run("ddp_one_step_parity", world_size, tmp_path)

    # Replicas must be BITWISE identical to each other: all-reduce delivers the same
    # buffer to every rank, so any difference here is a genuine divergence, not noise.
    digests = {r["digest"] for r in results}
    assert len(digests) == 1, f"replicas diverged after one step: {digests}"

    rank0 = results[0]
    assert rank0["max_abs_param_diff"] < DDP_PARITY_ATOL, (
        f"DDP one-step result differs from the single-process reference by "
        f"{rank0['max_abs_param_diff']:.3e} (> {DDP_PARITY_ATOL:.0e}) across "
        f"{rank0['param_count']} parameters — this is a sharding or reduction bug, "
        f"not floating-point noise"
    )


def test_ddp_global_batch_semantics_are_preserved(tmp_path):
    results = _run("ddp_one_step_parity", 2, tmp_path)
    for r in results:
        assert r["per_rank_batch"] == 8
        assert r["effective_global_batch"] == 16


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_eval_contains_no_duplicate_samples_and_covers_the_split(world_size, tmp_path):
    """The DAIC-WOZ 47-session hazard, exercised end to end through a real DataLoader."""
    results = _run("eval_coverage", world_size, tmp_path)
    for r in results:
        audit = r["audit"]
        assert audit["covers_exactly_once"] is True
        assert audit["unique_n"] == N_SAMPLES
        assert audit["duplicates"] == 0
        assert sum(audit["per_rank_counts"].values()) == N_SAMPLES
    assert sum(r["local_count"] for r in results) == N_SAMPLES
    assert results[0]["merged_ids"] == list(range(N_SAMPLES))


def test_stock_padded_sampler_is_caught_by_the_coverage_guard(tmp_path):
    """Proof the guard is load-bearing: the padded sampler MUST be rejected."""
    results = _run("eval_coverage_detects_padding_duplicates", 2, tmp_path)
    assert all(r["guard_fired"] for r in results), (
        "the stock DistributedSampler padded 47 -> 48 and the guard did not fire"
    )
    assert "duplicate" in results[0]["guard_message"].lower()


@pytest.mark.parametrize("world_size", [2, 3])
def test_early_stop_decision_is_broadcast_from_rank_zero(world_size, tmp_path):
    """Ranks fed different scores must still stop on the same epoch."""
    results = _run("early_stop_broadcast", world_size, tmp_path)
    epochs = {r["epochs_executed"] for r in results}
    assert len(epochs) == 1, f"ranks executed different epoch counts {epochs} -> deadlock risk"
    baseline = results[0]["decisions"]
    for r in results[1:]:
        assert r["decisions"] == baseline, "a rank saw a different early-stop decision"
    assert baseline[-1]["should_stop"] is True
    # Rank 0 saw 0.5, 0.4, 0.3 -> best is 0.5 from epoch 0, regardless of other ranks.
    assert baseline[-1]["best"] == pytest.approx(0.5)


@pytest.mark.parametrize("world_size", [2, 3])
def test_only_rank_zero_writes_shared_artifacts(world_size, tmp_path):
    results = _run("rank_zero_only_writes", world_size, tmp_path)
    assert [r["wrote"] for r in results] == [True] + [False] * (world_size - 1)
    run_dir = tmp_path / "rank_zero_only_writes" / "run"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["written_by_rank"] == 0, "a non-zero rank overwrote the shared metrics"
    assert json.loads((run_dir / "status.json").read_text())["status"] == "completed"
    # Every rank keeps its own diagnosable log.
    for r in range(world_size):
        assert (run_dir / f"rank{r}.log").exists()
        assert f"hello from rank {r}" in (run_dir / f"rank{r}.log").read_text()


@pytest.mark.parametrize("world_size", [2])
def test_checkpoint_resume_matches_uninterrupted_run(world_size, tmp_path):
    results = _run("checkpoint_resume", world_size, tmp_path)
    for r in results:
        assert r["resumed_digest"] == r["uninterrupted_digest"], (
            "resumed run diverged from the uninterrupted run: optimizer or RNG state lost"
        )
        assert r["meta"]["global_step"] == 1 and r["meta"]["epoch"] == 1


def test_checkpoint_refuses_a_dataset_hash_mismatch(tmp_path):
    results = _run("checkpoint_rejects_wrong_hash", 2, tmp_path)
    assert all(r["refused"] for r in results)
    assert "dataset_hash mismatch" in results[0]["message"]


@pytest.mark.parametrize("world_size", [2, 3])
def test_failure_on_one_rank_terminates_every_rank(world_size, tmp_path):
    """Rank 1 fails inside fail_together; NO rank may be left blocked in a collective.

    The worker catches and records the exception rather than propagating it, so that this
    test can inspect *what each rank saw*. The property under test is not "the job dies"
    (that is trivial) but "every rank learns about it and leaves the region", which is the
    difference between a job that fails in seconds and a job that holds a two-node GPU
    reservation until the 72-hour wall-clock limit expires.
    """
    results = _run("failure_propagates", world_size, tmp_path)
    assert len(results) == world_size, "a rank never reported — it was left hanging"
    assert all(r["raised"] for r in results), (
        f"not every rank raised: {[(r['rank'], r['raised']) for r in results]}"
    )
    origin = next(r for r in results if r["rank"] == 1)
    assert origin["is_rank_failure"] is False, (
        "the origin rank must surface its OWN error, not a generic RankFailure, "
        "or the real cause is lost"
    )
    assert "simulated rank-1 model failure" in origin["message"]
    peers = [r for r in results if r["rank"] != 1]
    assert all(r["is_rank_failure"] for r in peers), "peers must raise RankFailure"
    assert all("aborting" in r["message"] for r in peers)


@pytest.mark.parametrize("world_size", [2, 3])
def test_split_hash_disagreement_across_ranks_is_fatal(world_size, tmp_path):
    results = _run("hash_disagreement_is_fatal", world_size, tmp_path)
    assert all(r["refused"] for r in results)
    assert "differs across ranks" in results[0]["message"]


def test_matching_split_hash_passes(tmp_path):
    results = _run("hash_agreement_passes", 2, tmp_path)
    assert all(r["refused"] is False for r in results)


@pytest.mark.parametrize("world_size", [2, 3])
def test_train_sampler_set_epoch_reshuffles_and_covers_the_dataset(world_size, tmp_path):
    """Training sampler: set_epoch must reshuffle, and coverage must be complete.

    Note what is deliberately NOT asserted: shard disjointness. The stock
    ``DistributedSampler(drop_last=False)`` pads the index list up to a multiple of
    world_size by repeating samples, so at n=40, world_size=3 it emits 42 indices and two
    of them are duplicates. That is correct and necessary for training — DDP requires every
    rank to run the same number of backward passes or the gradient all-reduce deadlocks.

    It is also exactly why this sampler must never be used for evaluation, which is
    asserted separately in test_eval_contains_no_duplicate_samples_and_covers_the_split.
    """
    n = 40
    results = _run("train_sampler_set_epoch", world_size, tmp_path)
    assert all(r["order_changed"] for r in results), (
        "set_epoch did not change the order — every epoch would see the same shuffle"
    )
    pad = (-n) % world_size
    for key in ("epoch0", "epoch1"):
        shards = [r[key] for r in results]
        flat = [i for s in shards for i in s]
        assert set(flat) == set(range(n)), f"{key}: training shards do not cover the dataset"
        assert len(flat) == n + pad, (
            f"{key}: emitted {len(flat)} indices, expected {n + pad} "
            f"({n} samples + {pad} padding)"
        )
        assert len(flat) - len(set(flat)) == pad, (
            f"{key}: duplication is {len(flat) - len(set(flat))}, expected exactly the "
            f"documented padding {pad}"
        )
        assert len({len(s) for s in shards}) == 1, (
            f"{key}: uneven training shards would deadlock the gradient all-reduce"
        )


@pytest.mark.parametrize("world_size", [2, 3])
def test_ddp_replicas_hold_identical_weights(world_size, tmp_path):
    results = _run("replicas_agree", world_size, tmp_path)
    assert len({r["digest"] for r in results}) == 1
