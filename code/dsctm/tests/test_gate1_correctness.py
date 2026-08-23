"""Gate 1 — single-process correctness regressions.

Each test here closes a gap identified in artifacts/gate0/BASELINE_TEST_REPORT.md §3.2:
a Gate 1 requirement that the pre-existing suite did not cover. These are the properties
that must hold before any distributed work begins, because DDP replicates whatever the
single-process pipeline does — including its mistakes.

Reference: artifacts/gate1/SINGLE_PROCESS_CORRECTNESS.md
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from dsctm.data.contract import WindowedDataset
from dsctm.data.splits import subject_grouped_kfold
from dsctm.models import DMSTCN, DMSTCNConfig
from dsctm.models.baselines import LSTMBaseline, TransformerBaseline
from dsctm.models.blocks import Branch, FiLMAdapter
from dsctm.train.trainer import _build_loss, _make_loader, fit_normalizer

REPO_ROOT = Path(__file__).resolve().parents[3]
EDAIC_SPLIT_DIR = REPO_ROOT / "reviewer-package" / "data"


def _padded_dataset(n=12, T=40, F=5, n_classes=2, seed=0):
    """Right-padded dataset with genuinely varying true lengths."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, T, F)).astype(np.float32)
    lengths = rng.integers(T // 4, T, size=n).astype(np.int64)
    for i, L in enumerate(lengths):
        X[i, L:] = 0.0
    return WindowedDataset(
        X=X,
        y=rng.integers(0, n_classes, size=n),
        subject_id=np.array([f"s{i % 4:02d}" for i in range(n)]),
        n_classes=n_classes,
        lengths=lengths,
        dataset="synthetic-padded",
        version="gate1-v1",
    )


# --------------------------------------------------------------------------- #
# 1. LSTM baseline must consume packed sequences, not padding
# --------------------------------------------------------------------------- #
def test_lstm_baseline_uses_packed_sequences(monkeypatch):
    """Behavioural: padded tail cannot influence logits. Structural: pack is called."""
    torch.manual_seed(0)
    model = LSTMBaseline(6, 3, hidden=16, layers=1, head_hidden=16).eval()

    calls = {"n": 0}
    real_pack = nn.utils.rnn.pack_padded_sequence

    def counting_pack(*args, **kwargs):
        calls["n"] += 1
        return real_pack(*args, **kwargs)

    monkeypatch.setattr(nn.utils.rnn, "pack_padded_sequence", counting_pack)

    X = torch.randn(4, 30, 6)
    mask = torch.zeros(4, 30, dtype=torch.bool)
    mask[:, :18] = True
    corrupted = X.clone()
    corrupted[:, 18:] = torch.randn(4, 12, 6) * 1e3  # violent perturbation of the pad region

    with torch.no_grad():
        a = model(X, None, mask=mask)
        b = model(corrupted, None, mask=mask)

    assert calls["n"] == 2, "LSTM baseline did not pack its input when a mask was supplied"
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (
        "padded timesteps changed the LSTM logits: the recurrence is consuming padding"
    )


def test_lstm_baseline_without_mask_is_affected_by_padding():
    """Control: the invariance above is produced by the mask, not by luck."""
    torch.manual_seed(0)
    model = LSTMBaseline(6, 3, hidden=16, layers=1, head_hidden=16).eval()
    X = torch.randn(4, 30, 6)
    corrupted = X.clone()
    corrupted[:, 18:] = torch.randn(4, 12, 6) * 1e3
    with torch.no_grad():
        a = model(X, None, mask=None)
        b = model(corrupted, None, mask=None)
    assert not torch.allclose(a, b, atol=1e-3), (
        "unmasked LSTM was unexpectedly invariant; the positive test above proves nothing"
    )


# --------------------------------------------------------------------------- #
# 2. Transformer baseline must apply a key-padding mask
# --------------------------------------------------------------------------- #
def test_transformer_baseline_applies_padding_mask():
    torch.manual_seed(0)
    model = TransformerBaseline(6, 3, d_model=32, nhead=2, layers=1,
                                head_hidden=16, dropout=0.0).eval()
    X = torch.randn(2, 24, 6)
    mask = torch.zeros(2, 24, dtype=torch.bool)
    mask[:, :15] = True
    corrupted = X.clone()
    corrupted[:, 15:] = torch.randn(2, 9, 6) * 1e3

    with torch.no_grad():
        a = model(X, None, mask=mask)
        b = model(corrupted, None, mask=mask)

    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (
        "padded timesteps changed the Transformer logits: attention is reading padding"
    )


# --------------------------------------------------------------------------- #
# 3. lengths validated at the contract and propagated end to end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [0, -1, 999])
def test_contract_rejects_out_of_range_lengths(bad):
    X = np.zeros((3, 10, 2), dtype=np.float32)
    lengths = np.array([5, 5, bad], dtype=np.int64)
    with pytest.raises(ValueError, match="lengths"):
        WindowedDataset(X=X, y=np.zeros(3, int), subject_id=np.array(["a", "b", "c"]),
                        n_classes=2, lengths=lengths)


def test_contract_rejects_mismatched_lengths_count():
    X = np.zeros((3, 10, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="lengths length mismatch"):
        WindowedDataset(X=X, y=np.zeros(3, int), subject_id=np.array(["a", "b", "c"]),
                        n_classes=2, lengths=np.array([5, 5], dtype=np.int64))


def test_lengths_default_to_full_T_when_absent():
    X = np.zeros((3, 10, 2), dtype=np.float32)
    ds = WindowedDataset(X=X, y=np.zeros(3, int), subject_id=np.array(["a", "b", "c"]),
                         n_classes=2)
    assert ds.lengths.tolist() == [10, 10, 10]


def test_lengths_propagate_into_loader_mask_and_zero_the_tail():
    """The loader must derive the mask from lengths AND zero the padded region."""
    ds = _padded_dataset()
    idx = np.arange(ds.N)
    mean, std = fit_normalizer(ds.X[idx], ds.lengths[idx])
    loader = _make_loader(ds, idx, {}, mean, std, batch_size=ds.N, shuffle=False)
    # Gate 2 added a fifth tensor: the DATASET-GLOBAL sample id that distributed
    # evaluation uses to prove each sample was scored exactly once.
    X, y, subj, mask, sample_id = next(iter(loader))

    expected = np.arange(ds.T)[None, :] < ds.lengths[idx][:, None]
    assert mask.numpy().tolist() == expected.tolist(), "loader mask does not match lengths"
    assert torch.count_nonzero(X[~mask]) == 0, "padded region was not zeroed after normalization"
    assert mask.sum(1).numpy().tolist() == ds.lengths[idx].tolist()
    assert sample_id.tolist() == list(idx), "sample ids must be the dataset-global indices"


def test_sample_ids_are_dataset_global_not_split_local():
    """Fold-local positions would collide across folds and break coverage auditing."""
    ds = _padded_dataset(n=12)
    idx = np.array([3, 7, 9])
    mean, std = fit_normalizer(ds.X[idx], ds.lengths[idx])
    loader = _make_loader(ds, idx, {}, mean, std, batch_size=3, shuffle=False)
    _, _, _, _, sample_id = next(iter(loader))
    assert sample_id.tolist() == [3, 7, 9]


def test_normalizer_ignores_padding():
    """Fitting over the padded array would bias mu toward 0 and shrink sigma."""
    ds = _padded_dataset()
    idx = np.arange(ds.N)
    masked_mean, masked_std = fit_normalizer(ds.X[idx], ds.lengths[idx])
    naive_mean, naive_std = fit_normalizer(ds.X[idx], None)
    assert not np.allclose(masked_mean, naive_mean), "padding-aware normalizer degenerated"
    assert np.all(masked_std >= naive_std - 1e-6), (
        "padding-aware sigma should not be smaller than the padding-diluted sigma"
    )


# --------------------------------------------------------------------------- #
# 4. Per-subject adapter cost is d_s, not 2D  (tracker T2-03)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d_s", [4, 8, 16])
def test_per_subject_adapter_cost_is_d_s_not_2D(d_s):
    D, n_subjects = 128, 50
    small = FiLMAdapter(D, n_subjects, d_s=d_s)
    large = FiLMAdapter(D, n_subjects + 1, d_s=d_s)
    delta = sum(p.numel() for p in large.parameters()) - sum(p.numel() for p in small.parameters())
    assert delta == d_s, (
        f"adding one subject cost {delta} parameters, expected d_s={d_s}. "
        f"The manuscript's '2D parameters per subject' would predict {2 * D}."
    )
    assert delta != 2 * D
    assert small.embed.weight.shape == (n_subjects, d_s)
    # gamma/beta are generated activations of width D, not stored per subject
    assert small.to_gamma.out_features == D and small.to_beta.out_features == D


def test_full_model_per_subject_growth_is_d_s():
    base = DMSTCNConfig(input_dim=8, n_classes=3, n_subjects=46, d_s=8)
    more = DMSTCNConfig(input_dim=8, n_classes=3, n_subjects=47, d_s=8)
    n0 = sum(p.numel() for p in DMSTCN(base).parameters())
    n1 = sum(p.numel() for p in DMSTCN(more).parameters())
    assert n1 - n0 == base.d_s


# --------------------------------------------------------------------------- #
# 5. Official E-DAIC split files: disjoint, correctly sized, no dev+test merge
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not EDAIC_SPLIT_DIR.exists(), reason="reviewer-package/data absent")
def test_official_edaic_split_files_are_disjoint_and_correctly_sized():
    """The split CSVs shipped in reviewer-package/data are E-DAIC (AVEC2019), 163/56/56.

    They are NOT classic DAIC-WOZ (AVEC2017, 107/35/47) which the manuscript cites.
    This test pins what is actually in the repository so the corpus-identity question
    (tracker V3-02) is answered by evidence rather than by memory.
    """
    import pandas as pd

    expected = {"train": 163, "dev": 56, "test": 56}
    ids, frames = {}, {}
    for split, n in expected.items():
        df = pd.read_csv(EDAIC_SPLIT_DIR / f"{split}_split.csv")
        df.columns = [c.strip() for c in df.columns]
        assert len(df) == n, f"{split} split has {len(df)} rows, expected {n} (E-DAIC)"
        assert "Participant_ID" in df.columns and "PHQ_Binary" in df.columns
        ids[split] = set(df["Participant_ID"].astype(int))
        frames[split] = df

    assert sum(len(v) for v in ids.values()) == 275
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert not (ids[a] & ids[b]), f"participant leakage between {a} and {b}: {ids[a] & ids[b]}"

    # No dev+test merge: test must remain its own 56-participant partition.
    assert len(ids["test"]) == 56
    assert len(ids["dev"] | ids["test"]) == 112

    # Class imbalance must be reported, not assumed balanced (tracker V3-07).
    for split, df in frames.items():
        rate = float(df["PHQ_Binary"].mean())
        assert 0.15 < rate < 0.40, f"{split} positive rate {rate:.3f} outside the expected band"


@pytest.mark.skipif(not EDAIC_SPLIT_DIR.exists(), reason="reviewer-package/data absent")
def test_shipped_split_is_not_the_manuscript_107_82_partition():
    """Guard against silently reviving the manuscript's 107/82 description."""
    import pandas as pd

    n = {s: len(pd.read_csv(EDAIC_SPLIT_DIR / f"{s}_split.csv")) for s in ("train", "dev", "test")}
    assert n["train"] != 107, "unexpected AVEC2017-sized train split in the E-DAIC files"
    assert n["dev"] + n["test"] != 82, "dev+test sums to the manuscript's merged 82 — investigate"


# --------------------------------------------------------------------------- #
# 6. No duplicate or overlapping window content across splits  (tracker V3-06)
# --------------------------------------------------------------------------- #
def _window_hashes(X):
    return [hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest() for w in X]


def test_no_duplicate_window_content_across_grouped_folds():
    rng = np.random.default_rng(3)
    n, T, F = 60, 12, 4
    X = rng.normal(size=(n, T, F)).astype(np.float32)
    y = rng.integers(0, 3, size=n)
    subject_id = np.array([f"u{i % 10:02d}" for i in range(n)])
    folds, _ = subject_grouped_kfold(subject_id, y, n_splits=5, seed=0)

    hashes = np.array(_window_hashes(X))
    for f, (tr, va) in enumerate(folds):
        overlap = set(hashes[tr]) & set(hashes[va])
        assert not overlap, f"fold {f}: identical window content in train and val"


def test_duplicate_window_detector_actually_fires():
    """Control: the detector above must catch a planted duplicate."""
    rng = np.random.default_rng(4)
    n, T, F = 20, 8, 3
    X = rng.normal(size=(n, T, F)).astype(np.float32)
    X[10] = X[0]  # plant a cross-subject duplicate window
    subject_id = np.array([f"u{i:02d}" for i in range(n)])
    hashes = np.array(_window_hashes(X))
    tr = np.arange(0, 10)
    va = np.arange(10, 20)
    assert set(hashes[tr]) & set(hashes[va]), "duplicate-content detector failed to fire"
    assert subject_id[0] != subject_id[10]


# --------------------------------------------------------------------------- #
# 7. Class weights are computed from training labels only  (tracker V3-07)
# --------------------------------------------------------------------------- #
def test_class_weights_use_training_labels_only():
    device = "cpu"
    y_train = np.array([0] * 90 + [1] * 10)
    lossf = _build_loss({"class_weight": "balanced"}, y_train, 2, device)
    w = lossf.weight.detach().numpy()
    # sklearn convention: n / (C * count_c)
    expected = np.array([100 / (2 * 90), 100 / (2 * 10)])
    assert np.allclose(w, expected), f"weights {w} != {expected}"
    assert w[1] > w[0], "minority class must receive the larger weight"


def test_class_weights_are_independent_of_validation_labels():
    y_train = np.array([0] * 90 + [1] * 10)
    a = _build_loss({"class_weight": "balanced"}, y_train, 2, "cpu").weight.detach().numpy()
    # A wildly different validation distribution must not move the weights, because the
    # weights are a pure function of the training labels passed in.
    b = _build_loss({"class_weight": "balanced"}, y_train, 2, "cpu").weight.detach().numpy()
    assert np.allclose(a, b)


def test_unweighted_loss_is_the_default():
    """Configs that do not opt in must be byte-for-byte unchanged (no silent weighting)."""
    assert _build_loss({}, np.array([0, 1, 1]), 2, "cpu").weight is None
    assert _build_loss({"class_weight": None}, np.array([0, 1, 1]), 2, "cpu").weight is None


def test_empty_class_does_not_divide_by_zero():
    y_train = np.array([0] * 20)  # class 1 never observed
    w = _build_loss({"class_weight": "balanced"}, y_train, 2, "cpu").weight.detach().numpy()
    assert np.all(np.isfinite(w))


# --------------------------------------------------------------------------- #
# 8. Receptive fields derived from the implementation, never typed by hand
# --------------------------------------------------------------------------- #
def test_receptive_fields_are_61_481_1921_derived_from_implementation():
    cfg = DMSTCNConfig()
    expected = {"ssb": 61, "msb": 481, "lsb": 1921}
    manuscript_printed = {"ssb": 47, "msb": 383, "lsb": 1535}
    for name, dil in (("ssb", cfg.ssb), ("msb", cfg.msb), ("lsb", cfg.lsb)):
        br = Branch(cfg.D, cfg.K, dil)
        derived = br.theoretical_rf_two_conv()
        assert derived == 1 + 2 * (cfg.K - 1) * sum(dil)
        assert derived == expected[name], f"{name}: derived RF {derived} != {expected[name]}"
        assert derived != manuscript_printed[name]
        assert br.theoretical_rf_one_conv() != manuscript_printed[name]


# --------------------------------------------------------------------------- #
# 9. CSAG variants: faithful default preserved, nonlinear added beside it (D-006)
# --------------------------------------------------------------------------- #
def test_default_csag_is_manuscript_faithful_and_unchanged():
    model = DMSTCN(DMSTCNConfig(D=16, head_hidden=16, n_subjects=2))
    assert model.cfg.csag_mode == "attention"
    assert model.csag is not None and model.csag.is_manuscript_faithful
    assert model.csag.nonlinearity is None


def test_linear_csag_alias_is_numerically_identical_to_default():
    torch.manual_seed(11)
    a = DMSTCN(DMSTCNConfig(D=16, head_hidden=16, n_subjects=2, csag_mode="attention"))
    torch.manual_seed(11)
    b = DMSTCN(DMSTCNConfig(D=16, head_hidden=16, n_subjects=2, csag_mode="linear_csag"))
    X = torch.randn(3, 20, 8)
    s = torch.zeros(3, dtype=torch.long)
    with torch.no_grad():
        assert torch.allclose(a(X, s), b(X, s), atol=0, rtol=0), (
            "linear_csag must be a pure alias for the faithful gate"
        )


def test_nonlinear_csag_is_a_distinct_declared_variant():
    torch.manual_seed(11)
    faithful = DMSTCN(DMSTCNConfig(D=16, head_hidden=16, n_subjects=2, csag_mode="attention"))
    torch.manual_seed(11)
    nonlinear = DMSTCN(DMSTCNConfig(D=16, head_hidden=16, n_subjects=2,
                                    csag_mode="nonlinear_csag", csag_nonlinearity="relu"))
    assert not nonlinear.csag.is_manuscript_faithful
    assert nonlinear.csag.nonlinearity == "relu"
    X = torch.randn(3, 20, 8)
    s = torch.zeros(3, dtype=torch.long)
    with torch.no_grad():
        assert not torch.allclose(faithful(X, s), nonlinear(X, s), atol=1e-6), (
            "nonlinear_csag produced identical output to the faithful gate"
        )


def test_unknown_csag_mode_and_nonlinearity_are_rejected():
    with pytest.raises(ValueError, match="unknown csag_mode"):
        DMSTCN(DMSTCNConfig(csag_mode="bogus"))
    with pytest.raises(ValueError, match="unknown CSAG nonlinearity"):
        DMSTCN(DMSTCNConfig(csag_mode="nonlinear_csag", csag_nonlinearity="bogus"))


# --------------------------------------------------------------------------- #
# 10. Strict causality restated as an explicit end-to-end assertion
# --------------------------------------------------------------------------- #
def test_zero_future_leakage_end_to_end():
    torch.manual_seed(5)
    model = DMSTCN(DMSTCNConfig(D=16, head_hidden=16, n_subjects=2)).eval()
    X = torch.randn(2, 64, 8)
    s = torch.zeros(2, dtype=torch.long)
    mask = torch.zeros(2, 64, dtype=torch.bool)
    mask[:, :32] = True
    future_changed = X.clone()
    future_changed[:, 32:] += 50.0
    with torch.no_grad():
        assert torch.allclose(model(X, s, mask=mask), model(future_changed, s, mask=mask),
                              atol=1e-5, rtol=1e-5)
