"""Tests for the statistical analysis plan — including the reviewer-facing
correctness facts (Wilcoxon n=5 impossibility) encoded as executable assertions."""
import numpy as np

from dsctm.eval import statistics as st


def test_wilcoxon_n5_two_sided_cannot_reach_significance():
    # master-prompt §8 / analytic check: 5 non-zero pairs, two-sided exact,
    # minimum achievable p = 2 / 2**5 = 0.0625 > 0.05.
    assert abs(st.min_achievable_wilcoxon_p(5, "two-sided") - 0.0625) < 1e-12
    x = np.array([0.70, 0.71, 0.69, 0.72, 0.68])
    y = np.array([0.60, 0.61, 0.59, 0.62, 0.58])  # x strictly > y everywhere
    res = st.wilcoxon_paired(x, y, alternative="two-sided")
    assert res["n_nonzero"] == 5
    assert res["significance_reachable"] is False
    assert res["p_value"] >= 0.0625 - 1e-9


def test_significance_reachable_at_n6():
    assert st.min_achievable_wilcoxon_p(6, "two-sided") <= 0.05


def test_cluster_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(10), 20)
    values = 0.7 + 0.02 * rng.standard_normal(200)
    point, lo, hi = st.cluster_bootstrap_ci(values, clusters, seed=1)
    assert lo <= point <= hi
    assert hi - lo < 0.1


def test_paired_effect_sizes_sign():
    x = np.array([0.8, 0.75, 0.9, 0.85])
    y = np.array([0.7, 0.70, 0.8, 0.80])
    assert st.paired_rank_biserial(x, y) > 0
    assert st.hodges_lehmann_paired(x, y) > 0
    assert st.cohens_dz(x, y) > 0


def test_multiplicity_monotone():
    p = [0.001, 0.02, 0.03, 0.5]
    holm = st.holm_bonferroni(p)
    bh = st.benjamini_hochberg(p)
    assert np.all(holm >= np.array(p) - 1e-12)  # adjusted >= raw
    assert np.all(bh >= np.array(p) - 1e-12)
    assert holm[0] <= 1.0 and bh[0] <= 1.0
