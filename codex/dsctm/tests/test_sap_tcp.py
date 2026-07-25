"""Gates 8 & 9 — Scale-Aware Partitioner and real Temporal Consistency Protocol.

SAP equivalence runs over real gloo processes at world_size 4 (3 branches + aggregator),
which is also the minimum PARAM topology: a node has 2 V100s, so this needs 2 nodes.

No performance claim is made here. Equivalence first; speed is Gate 10's problem, and a
faster wrong answer is worthless.
"""
from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from dsctm.distributed import (
    BranchAction, ExecutionMode, MODE_DESCRIPTIONS, Placement, SyncReason, TCPState,
    TemporalConsistencyProtocol, plan_placement, predicted_bytes_per_sample,
)
from dsctm.distributed.errors import PreflightFailure
from dsctm.distributed.sap import CommStats
from dsctm.models import DMSTCN, DMSTCNConfig

B, T, F, C, D = 2, 24, 5, 3, 16
SAP_ATOL = 1e-5   # fp32 p2p round-trip; see test docstring for justification


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _cfg() -> DMSTCNConfig:
    return DMSTCNConfig(input_dim=F, n_classes=C, D=D, head_hidden=D, n_subjects=2,
                        ssb=(1, 2), msb=(4, 8), lsb=(16, 32))


def _batch():
    g = torch.Generator().manual_seed(4242)
    return (torch.randn(B, T, F, generator=g),
            torch.randint(0, C, (B,), generator=g),
            torch.zeros(B, dtype=torch.long),
            torch.ones(B, T, dtype=torch.bool))


# --------------------------------------------------------------------------- #
# Placement (single process)
# --------------------------------------------------------------------------- #
def test_placement_is_deterministic_and_covers_every_role():
    a = plan_placement(("ssb", "msb", "lsb"), 4)
    b = plan_placement(("ssb", "msb", "lsb"), 4)
    assert a.to_dict() == b.to_dict()
    roles = a.to_dict()["roles"]
    assert sorted(roles.values()) == ["aggregator", "branch:lsb", "branch:msb", "branch:ssb"]
    assert a.aggregator_rank == 3
    assert len(set(a.branch_to_rank.values())) == 3


def test_placement_orders_branches_by_the_eq11_load_score():
    """Eq. 11: L_b = C_compute / (C_compute + C_comm), descending."""
    p = plan_placement(("ssb", "msb", "lsb"), 4,
                       compute_cost={"ssb": 0.9, "msb": 0.5, "lsb": 0.1},
                       comm_cost={"ssb": 0.2, "msb": 0.2, "lsb": 0.2})
    assert p.branch_to_rank["ssb"] < p.branch_to_rank["msb"] < p.branch_to_rank["lsb"]


def test_placement_refuses_a_two_gpu_node():
    """A PARAM node has 2 GPUs; the 3-branch layout needs >= 4 ranks = 2 nodes."""
    with pytest.raises(PreflightFailure, match="at least 4 ranks"):
        plan_placement(("ssb", "msb", "lsb"), 2)


def test_surplus_ranks_become_replicas_of_the_heaviest_branches():
    p = plan_placement(("ssb", "msb", "lsb"), 6)
    sizes = {b: len(r) for b, r in p.replica_groups.items()}
    assert sum(sizes.values()) == 5           # 6 ranks - 1 aggregator
    assert max(sizes.values()) == 2
    flat = [r for ranks in p.replica_groups.values() for r in ranks]
    assert len(flat) == len(set(flat)), "a rank was assigned to two branches"


def test_branch_order_is_fixed_not_set_iteration():
    assert plan_placement(("lsb", "ssb", "msb"), 4).branches == ("ssb", "msb", "lsb")


def test_communication_prediction_matches_the_manuscript_formula():
    pred = predicted_bytes_per_sample(T=2000, D=128, n_branches=3)
    assert pred["forward"] == 3 * 2000 * 128 * 4
    assert pred["total"] == 2 * pred["forward"]


def test_comm_stats_accumulate_by_direction():
    s = CommStats()
    s.record("forward", torch.zeros(10, 10))
    s.record("backward", torch.zeros(10, 10))
    s.record("allreduce", torch.zeros(5))
    assert s.forward_bytes == s.backward_bytes == 400
    assert s.total_bytes == 400 + 400 + 20
    assert s.forward_calls == s.backward_calls == 1


# --------------------------------------------------------------------------- #
# TCP protocol semantics (single process)
# --------------------------------------------------------------------------- #
def _tcp(delta_max=3, t_sync=1000, world=4):
    return TemporalConsistencyProtocol(plan_placement(("ssb", "msb", "lsb"), world),
                                       ctx=None, delta_max=delta_max, t_sync=t_sync)


def test_staleness_increments_then_hold_fires_at_delta_max():
    tcp = _tcp(delta_max=3)
    for _ in range(3):
        d = tcp.decide()
        assert all(a == BranchAction.UPDATE.value for a in d["actions"].values())
    d = tcp.decide()
    assert all(a == BranchAction.HOLD.value for a in d["actions"].values())
    assert d["sync"] and d["sync_reason"] == SyncReason.HOLD.value
    assert tcp.state.branch_staleness == {"ssb": 0, "msb": 0, "lsb": 0}


def test_hold_takes_precedence_over_the_periodic_schedule():
    tcp = _tcp(delta_max=2, t_sync=3)
    tcp.decide(); tcp.decide()
    d = tcp.decide()                      # step 3: both HOLD and periodic are due
    assert d["sync_reason"] == SyncReason.HOLD.value
    assert tcp.state.hold_triggered_syncs == 1 and tcp.state.periodic_syncs == 0


def test_periodic_sync_fires_on_schedule_when_no_hold():
    tcp = _tcp(delta_max=1000, t_sync=5)
    reasons = [tcp.decide()["sync_reason"] for _ in range(10)]
    assert reasons.count(SyncReason.PERIODIC.value) == 2
    assert reasons[4] == SyncReason.PERIODIC.value and reasons[9] == SyncReason.PERIODIC.value


def test_bounded_divergence_invariant_holds_over_a_long_run():
    tcp = _tcp(delta_max=10, t_sync=7)
    for _ in range(1000):
        tcp.decide()
        assert max(tcp.state.branch_staleness.values()) <= 10
    inv = tcp.check_invariants()
    assert inv["bounded_divergence"] and inv["max_staleness"] <= inv["delta_max"]


@pytest.mark.parametrize("delta_max,t_sync", [(1, 1000), (5, 3), (10, 50), (200, 7)])
def test_invariant_holds_across_the_sweep_grid(delta_max, t_sync):
    tcp = _tcp(delta_max=delta_max, t_sync=t_sync)
    for _ in range(500):
        tcp.decide()
    assert tcp.check_invariants()["bounded_divergence"]


def test_skipped_branches_do_not_advance_their_version():
    tcp = _tcp()
    tcp.decide(updating=["ssb"])
    assert tcp.state.branch_versions["ssb"] == 1
    assert tcp.state.branch_versions["msb"] == 0
    assert tcp.state.branch_staleness["msb"] == 0


def test_decisions_are_deterministic():
    a, b = _tcp(delta_max=4, t_sync=6), _tcp(delta_max=4, t_sync=6)
    for _ in range(60):
        assert a.decide() == b.decide()


def test_tcp_state_roundtrips_through_a_checkpoint():
    tcp = _tcp(delta_max=5, t_sync=9)
    for _ in range(40):
        tcp.decide()
    blob = json.loads(json.dumps(tcp.state_dict()))
    restored = _tcp()
    restored.load_state_dict(blob)
    assert restored.state.to_dict() == tcp.state.to_dict()
    assert restored.decide() == tcp.decide(), "resumed protocol diverged"


def test_optimizer_state_sync_is_off_by_default_and_recorded():
    tcp = _tcp()
    assert tcp.sync_optimizer_state is False
    assert tcp.state_dict()["sync_optimizer_state"] is False


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        _tcp(delta_max=0)
    with pytest.raises(ValueError):
        _tcp(t_sync=0)


def test_four_execution_modes_are_defined_and_described():
    assert len(ExecutionMode) == 4
    assert set(MODE_DESCRIPTIONS) == set(ExecutionMode)
    assert "Control" in MODE_DESCRIPTIONS[ExecutionMode.DDP_SYNC]
    for m in ExecutionMode:
        assert len(MODE_DESCRIPTIONS[m]) > 40


# --------------------------------------------------------------------------- #
# SAP equivalence over real processes
# --------------------------------------------------------------------------- #
def _sap_worker(rank: int, world_size: int, port: int, out_dir: str, case: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK="0",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    from dsctm.distributed import (SAPModel, cleanup, init_distributed, plan_placement,
                                   seed_everything)
    from dsctm.distributed.sap import sap_step

    result = {"rank": rank}
    ctx = init_distributed(timeout_minutes=2)
    try:
        placement = plan_placement(("ssb", "msb", "lsb"), world_size)
        seed_everything(31, ctx)
        model = DMSTCN(_cfg())
        sap = SAPModel(model, placement, ctx)
        X, y, subj, mask = _batch()
        result["role"] = sap.role

        if case == "forward_equivalence":
            logits = sap(X, subj, mask)
            if sap.is_aggregator:
                seed_everything(31, ctx)
                ref = DMSTCN(_cfg()).eval()
                sap.inner.eval()
                with torch.no_grad():
                    expected = ref(X, subj, mask=mask)
                result["max_abs_diff"] = float((logits - expected).abs().max())
                result["logits_shape"] = list(logits.shape)
            result["ok_forward"] = True

        elif case == "backward_equivalence":
            opt = torch.optim.SGD([p for p in sap.parameters() if p.requires_grad], lr=0.1)
            loss = sap_step(sap, X, y, subj, mask, nn.functional.cross_entropy, opt)
            result["loss"] = loss
            grads = {n: float(p.grad.abs().sum()) for n, p in sap.inner.named_parameters()
                     if p.grad is not None}
            result["n_params_with_grad"] = len(grads)
            result["grad_l1"] = round(sum(grads.values()), 6)
            if sap.is_aggregator:
                seed_everything(31, ctx)
                ref = DMSTCN(_cfg())
                ref.zero_grad()
                nn.functional.cross_entropy(ref(X, subj, mask=mask), y).backward()
                head_ref = float(sum(p.grad.abs().sum() for p in ref.head.parameters()))
                head_sap = float(sum(p.grad.abs().sum() for p in sap.inner.head.parameters()))
                result["head_grad_ref"] = head_ref
                result["head_grad_sap"] = head_sap
                result["head_grad_absdiff"] = abs(head_ref - head_sap)

        elif case == "only_owned_parts_have_gradients":
            opt = torch.optim.SGD([p for p in sap.parameters() if p.requires_grad], lr=0.1)
            sap_step(sap, X, y, subj, mask, nn.functional.cross_entropy, opt)
            owned, foreign = [], []
            for name, module in sap.inner._branches.items():
                has = any(p.grad is not None for p in module.parameters())
                (owned if name == sap.my_branch else foreign).append((name, has))
            result["my_branch"] = sap.my_branch
            result["owned_have_grad"] = owned
            result["foreign_have_grad"] = foreign
            result["head_has_grad"] = any(p.grad is not None
                                          for p in sap.inner.head.parameters())

        elif case == "comm_volume_recorded":
            sap(X, subj, mask)
            result["stats"] = sap.stats.to_dict()
            result["predicted"] = predicted_bytes_per_sample(T, D, 3)

        elif case == "tcp_all_ranks_agree":
            from dsctm.distributed import TemporalConsistencyProtocol
            tcp = TemporalConsistencyProtocol(placement, ctx, delta_max=3, t_sync=5)
            log = [tcp.decide() for _ in range(20)]
            result["decisions"] = log
            result["invariants"] = tcp.check_invariants()

        elif case == "deterministic_branch_order":
            result["branches"] = list(placement.branches)
            result["branch_to_rank"] = dict(placement.branch_to_rank)

        result["ok"] = True
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(result, default=str))
        cleanup()
        raise
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(result, default=str))
    cleanup()


def _run_sap(case: str, world_size: int, tmp_path: Path) -> list[dict]:
    out = tmp_path / case
    out.mkdir(parents=True, exist_ok=True)
    mp.spawn(_sap_worker, args=(world_size, _free_port(), str(out), case),
             nprocs=world_size, join=True)
    results = [json.loads((out / f"rank{r}.json").read_text()) for r in range(world_size)]
    assert all(r["ok"] for r in results), [r.get("error") for r in results]
    return results


def test_sap_forward_matches_the_monolithic_model(tmp_path):
    """Branch-parallel forward must equal single-process forward.

    Tolerance 1e-5 rather than bitwise: activations make a p2p round trip through gloo
    serialisation, and the aggregator sums three received tensors in a different order than
    the monolithic model's fused stack. A genuine placement or wiring bug produces a
    difference of order 1e-1, not 1e-6.
    """
    results = _run_sap("forward_equivalence", 4, tmp_path)
    agg = next(r for r in results if r["role"] == "aggregator")
    assert agg["logits_shape"] == [B, C]
    assert agg["max_abs_diff"] < SAP_ATOL, (
        f"SAP forward differs from the monolithic model by {agg['max_abs_diff']:.3e}")


def test_sap_backward_reaches_every_rank(tmp_path):
    results = _run_sap("backward_equivalence", 4, tmp_path)
    for r in results:
        assert r["n_params_with_grad"] > 0, f"{r['role']} received no gradient at all"
        assert r["grad_l1"] > 0, f"{r['role']} gradients are all zero"
    agg = next(r for r in results if r["role"] == "aggregator")
    assert agg["loss"] is not None and agg["loss"] > 0
    assert agg["head_grad_absdiff"] < 1e-4, (
        f"aggregator head gradient differs from the monolithic reference by "
        f"{agg['head_grad_absdiff']:.3e}")


def test_sap_ranks_only_hold_gradients_for_what_they_own(tmp_path):
    results = _run_sap("only_owned_parts_have_gradients", 4, tmp_path)
    for r in results:
        if r["role"] == "aggregator":
            assert r["head_has_grad"], "aggregator must own the head gradient"
            assert all(not has for _, has in r["foreign_have_grad"])
        else:
            assert all(has for _, has in r["owned_have_grad"]), r["owned_have_grad"]
            assert all(not has for _, has in r["foreign_have_grad"]), r["foreign_have_grad"]
            assert not r["head_has_grad"], "a branch rank must not own the head"


def test_sap_records_measured_communication_volume(tmp_path):
    """Tracker E4-17: measured, not estimated."""
    results = _run_sap("comm_volume_recorded", 4, tmp_path)
    agg = next(r for r in results if r["role"] == "aggregator")
    assert agg["stats"]["forward_calls"] == 3, "aggregator should receive from 3 branches"
    assert agg["stats"]["forward_bytes"] == 3 * B * T * D * 4
    for r in results:
        if r["role"] != "aggregator":
            assert r["stats"]["forward_calls"] == 1
            assert r["stats"]["forward_bytes"] == B * T * D * 4


def test_sap_branch_order_is_deterministic_across_ranks(tmp_path):
    results = _run_sap("deterministic_branch_order", 4, tmp_path)
    assert len({tuple(r["branches"]) for r in results}) == 1
    assert len({json.dumps(r["branch_to_rank"], sort_keys=True) for r in results}) == 1


def test_tcp_decisions_agree_on_every_rank(tmp_path):
    """A rank that disagrees about synchronising deadlocks the group."""
    results = _run_sap("tcp_all_ranks_agree", 4, tmp_path)
    baseline = results[0]["decisions"]
    for r in results[1:]:
        assert r["decisions"] == baseline, "ranks disagreed on a TCP decision"
    for r in results:
        assert r["invariants"]["bounded_divergence"]
