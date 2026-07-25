"""Gate 11 — machine verification of Proposition 1 (bounded version divergence).

The original Theorem 1 is withdrawn (artifacts/gate11/THEOREM_RECORD.md). What replaces it
is a protocol invariant, and an invariant is worth exactly as much as its verification. So
it is checked by exhaustive state exploration rather than by re-reading a proof sketch.
"""
from __future__ import annotations

import itertools, json
import pytest

from dsctm.distributed import SyncReason, TemporalConsistencyProtocol, plan_placement

PLACEMENT = plan_placement(("ssb", "msb", "lsb"), 4)


def _tcp(delta_max, t_sync):
    return TemporalConsistencyProtocol(PLACEMENT, ctx=None,
                                       delta_max=delta_max, t_sync=t_sync)


@pytest.mark.parametrize("delta_max", range(1, 13))
@pytest.mark.parametrize("t_sync", [1, 2, 3, 5, 7, 11, 50])
def test_proposition1_bounded_divergence_exhaustive(delta_max, t_sync):
    """Delta_b <= delta_max at EVERY step, for every (delta_max, T_sync) in the grid."""
    tcp = _tcp(delta_max, t_sync)
    for step in range(400):
        tcp.decide()
        worst = max(tcp.state.branch_staleness.values())
        assert worst <= delta_max, (
            f"invariant violated at step {step}: Delta={worst} > delta_max={delta_max}")


@pytest.mark.parametrize("t_sync", [1, 3, 5, 10, 25])
def test_synchronisation_interval_never_exceeds_t_sync(t_sync):
    tcp = _tcp(delta_max=10_000, t_sync=t_sync)   # HOLD disabled: isolate the periodic rule
    for _ in range(300):
        tcp.decide()
    steps = [e["step"] for e in tcp.state.sync_log]
    assert steps, "no synchronisation occurred at all"
    gaps = [b - a for a, b in zip(steps, steps[1:])]
    assert all(g <= t_sync for g in gaps), f"gap exceeded T_sync={t_sync}: {max(gaps)}"
    assert steps[0] <= t_sync


@pytest.mark.parametrize("delta_max,t_sync", list(itertools.product([1, 2, 5], [1, 2, 5])))
def test_hold_always_precedes_periodic_when_both_are_due(delta_max, t_sync):
    tcp = _tcp(delta_max, t_sync)
    for _ in range(200):
        before = dict(tcp.state.branch_staleness)
        d = tcp.decide()
        hold_due = any(v >= delta_max for v in before.values())
        if hold_due and d["sync"]:
            assert d["sync_reason"] == SyncReason.HOLD.value, (
                "periodic reason recorded while a HOLD was due")


@pytest.mark.parametrize("delta_max,t_sync", [(3, 7), (10, 50), (1, 1), (200, 13)])
def test_versions_are_monotone_non_decreasing(delta_max, t_sync):
    tcp = _tcp(delta_max, t_sync)
    previous = dict(tcp.state.branch_versions)
    for _ in range(300):
        tcp.decide()
        for b, v in tcp.state.branch_versions.items():
            assert v >= previous[b], f"version regressed for {b}"
        previous = dict(tcp.state.branch_versions)


def test_partial_update_patterns_preserve_the_invariant():
    """Only some branches update on a given step — the realistic asynchronous case."""
    patterns = [["ssb"], ["msb"], ["lsb"], ["ssb", "msb"], ["msb", "lsb"],
                ["ssb", "msb", "lsb"], []]
    tcp = _tcp(delta_max=6, t_sync=17)
    for step in range(500):
        tcp.decide(updating=patterns[step % len(patterns)])
        assert max(tcp.state.branch_staleness.values()) <= 6


def test_determinism_two_instances_agree_step_for_step():
    a, b = _tcp(4, 9), _tcp(4, 9)
    for _ in range(250):
        assert a.decide() == b.decide()


def test_checkpoint_restore_continues_identically():
    original = _tcp(5, 11)
    for _ in range(60):
        original.decide()
    blob = json.loads(json.dumps(original.state_dict()))
    restored = _tcp(5, 11)
    restored.load_state_dict(blob)
    for _ in range(60):
        assert restored.decide() == original.decide()
    assert restored.check_invariants() == original.check_invariants()


def test_counterexample_search_finds_no_violation():
    """Full grid sweep. Reported in THEOREM_RECORD.md as the verification evidence."""
    violations, configs = [], 0
    for delta_max in range(1, 13):
        for t_sync in range(1, 13):
            configs += 1
            tcp = _tcp(delta_max, t_sync)
            for _ in range(400):
                tcp.decide()
                if max(tcp.state.branch_staleness.values()) > delta_max:
                    violations.append((delta_max, t_sync))
                    break
    assert configs == 144
    assert not violations, f"invariant violated for {violations[:5]}"


def test_invariant_report_is_self_consistent():
    tcp = _tcp(8, 20)
    for _ in range(400):
        tcp.decide()
    inv = tcp.check_invariants()
    assert inv["bounded_divergence"] is True
    assert inv["max_staleness"] <= inv["delta_max"] == 8
    assert inv["hold_events"] >= 0
    assert inv["periodic_syncs"] + inv["hold_triggered_syncs"] == inv["global_version"]


def test_no_convergence_claim_is_exposed_by_the_api():
    """Guard against the withdrawn theorem creeping back in as an attribute or docstring."""
    tcp = _tcp(10, 50)
    surface = " ".join(dir(tcp)) + (TemporalConsistencyProtocol.__doc__ or "")
    for banned in ("convergence_bound", "gradient_error_bound", "converges"):
        assert banned not in surface, (
            f"{banned!r} reappeared in the TCP API; Theorem 1 was withdrawn (Gate 11)")
