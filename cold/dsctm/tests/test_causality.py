"""EXP-0.4 / EXP-0.3 as executable assertions: causality, batch-invariance,
determinism, checkpoint equivalence, and the TCP staleness invariants."""
import torch

from dsctm.experiments.gate0 import exp_0_3_sync_invariants, exp_0_4_causality


def test_causality_and_reproducibility():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = exp_0_4_causality(dev)["checks"]
    assert c["is_causal"], c["branch_causal_max_diff_upto_t0"]
    assert c["batch_invariant"], c["batch_invariance_max_diff"]
    assert c["deterministic"], c["determinism_max_diff"]
    assert c["variable_length_ok"]
    assert c["checkpoint_equivalent"], c["checkpoint_reload_max_diff"]


def test_tcp_invariants():
    c = exp_0_3_sync_invariants()["checks"]
    assert c["periodic_allreduce_at_t_sync"]
    assert c["hold_triggered"]
    assert c["hold_resets_delta"]
    assert c["invariant_delta_le_delta_max"]
    assert c["causal_mask_future_zeroed"]
