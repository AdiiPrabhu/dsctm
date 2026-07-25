"""EXP-0.4 / EXP-0.3 as executable assertions: causality, batch-invariance,
determinism, checkpoint equivalence, and the TCP staleness invariants."""
import torch

from dsctm.experiments.gate0 import exp_0_3_sync_invariants, exp_0_4_causality
from dsctm.models import DMSTCN, DMSTCNConfig


def test_causality_and_reproducibility():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = exp_0_4_causality(dev)["checks"]
    assert c["is_causal"], c["branch_causal_max_diff_upto_t0"]
    assert c["batch_invariant"], c["batch_invariance_max_diff"]
    assert c["deterministic"], c["determinism_max_diff"]
    assert c["variable_length_ok"]
    assert c["checkpoint_equivalent"], c["checkpoint_reload_max_diff"]
    if dev == "cuda":
        assert c["mixed_precision_loss_finite"]
        assert c["mixed_precision_gradients_finite"]


def test_tcp_invariants():
    c = exp_0_3_sync_invariants()["checks"]
    assert c["periodic_allreduce_at_t_sync"]
    assert c["hold_triggered"]
    assert c["hold_resets_delta"]
    assert c["invariant_delta_le_delta_max"]
    assert c["causal_mask_future_zeroed"]


def test_padding_mask_makes_head_invariant_to_masked_tail():
    """Right-padding values must not alter a prediction when marked invalid."""
    torch.manual_seed(7)
    model = DMSTCN(DMSTCNConfig(D=16, head_hidden=16, n_subjects=2)).eval()
    X = torch.randn(2, 60, 8)
    mask = torch.zeros(2, 60, dtype=torch.bool)
    mask[:, :35] = True
    X_changed = X.clone()
    X_changed[:, 35:] = torch.randn_like(X_changed[:, 35:]) * 100
    subjects = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        a = model(X, subjects, mask=mask)
        b = model(X_changed, subjects, mask=mask)
    # Causal convolutions ensure the changed future cannot affect valid positions;
    # masked pooling excludes the changed tail itself.
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)
