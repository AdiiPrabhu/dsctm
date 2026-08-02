"""Statistical analysis plan (master-prompt §8).

Design decisions encoded here (these ARE the fixes for the reviewer statistics
concerns, so they live in code, not prose):

  * Primary independent unit = participant / grouped fold, NOT the random seed.
    Seeds are paired secondary repetitions measuring optimization instability.
  * Confidence intervals via participant-level (cluster) bootstrap for nested
    predictions; plain bootstrap for already-aggregated fold/subject scores.
  * Paired effect sizes: matched-pairs rank-biserial correlation and the
    Hodges-Lehmann paired shift. Cohen's d_z is supplementary only.
  * Wilcoxon wrapper refuses to imply an unreachable significance claim:
    five non-zero paired observations cannot reach p<0.05 two-sided exact, so
    `significance_reachable` is surfaced explicitly.
  * Multiplicity: Holm and Benjamini-Hochberg; report raw and adjusted p-values.
"""
from __future__ import annotations

import numpy as np


def _rng(seed):
    return np.random.default_rng(seed)


def _rankdata(a):
    """Average ranks (1-based), ties averaged. Avoids a scipy dependency here."""
    a = np.asarray(a, float)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


# --------------------------------------------------------------------------- #
# Confidence intervals
# --------------------------------------------------------------------------- #
def bootstrap_ci(values, statistic=np.mean, n_boot=10000, ci=0.95, seed=0):
    """Nonparametric bootstrap CI for an already-aggregated set of scores
    (e.g. one score per fold or per subject)."""
    v = np.asarray(values, float)
    rng = _rng(seed)
    n = len(v)
    boots = np.array([statistic(v[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(statistic(v)), float(lo), float(hi)


def cluster_bootstrap_ci(values, clusters, statistic=np.mean, n_boot=10000, ci=0.95, seed=0):
    """Participant-level (hierarchical) bootstrap for nested predictions:
    resample clusters (subjects) with replacement, then resample within each."""
    v = np.asarray(values, float)
    c = np.asarray(clusters)
    groups = {g: v[c == g] for g in np.unique(c)}
    keys = list(groups)
    rng = _rng(seed)
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        sample = np.concatenate(
            [groups[k][rng.integers(0, len(groups[k]), len(groups[k]))] for k in chosen]
        )
        boots.append(statistic(sample))
    boots = np.array(boots)
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(statistic(v)), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Paired effect sizes
# --------------------------------------------------------------------------- #
def paired_rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation on the differences (x - y)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0
    ranks = _rankdata(np.abs(d))
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()
    return float((r_plus - r_minus) / (n * (n + 1) / 2))


def hodges_lehmann_paired(x, y):
    """Hodges-Lehmann estimate of the paired shift = median of the differences."""
    d = np.asarray(x, float) - np.asarray(y, float)
    return float(np.median(d))


def cohens_dz(x, y):
    """Paired Cohen's d_z (supplementary)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


# --------------------------------------------------------------------------- #
# Wilcoxon signed-rank with an honesty guard
# --------------------------------------------------------------------------- #
def min_achievable_wilcoxon_p(n_nonzero, alternative="two-sided"):
    """Smallest attainable exact Wilcoxon signed-rank p for n non-zero pairs."""
    if n_nonzero <= 0:
        return 1.0
    one_sided = 1.0 / (2 ** n_nonzero)
    return 2.0 * one_sided if alternative == "two-sided" else one_sided


def wilcoxon_paired(x, y, alternative="two-sided", zero_method="wilcox"):
    """Paired Wilcoxon signed-rank test with a reachability guard.

    Always reports `min_achievable_p` and `significance_reachable` so a caller
    can never present a p<0.05 marker the sample size cannot support (e.g. the
    manuscript's "† p<0.05 (5 seeds)": min two-sided exact p at n=5 is 0.0625).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    d = x - y
    n_nonzero = int(np.sum(d != 0))
    min_p = min_achievable_wilcoxon_p(n_nonzero, alternative)
    out = {
        "n_pairs": int(len(d)),
        "n_nonzero": n_nonzero,
        "alternative": alternative,
        "zero_method": zero_method,
        "min_achievable_p": min_p,
        "significance_reachable": bool(min_p <= 0.05),
    }
    try:
        from scipy.stats import wilcoxon

        if n_nonzero >= 1:
            res = wilcoxon(
                x, y,
                alternative=alternative,
                zero_method=zero_method,
                mode="exact" if n_nonzero <= 25 else "approx",
            )
            out["statistic"] = float(res.statistic)
            out["p_value"] = float(res.pvalue)
        else:
            out["statistic"] = float("nan")
            out["p_value"] = 1.0
    except Exception as e:  # pragma: no cover
        out["statistic"] = None
        out["p_value"] = None
        out["error"] = str(e)
    return out


# --------------------------------------------------------------------------- #
# Multiplicity correction
# --------------------------------------------------------------------------- #
def holm_bonferroni(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        prev = min(prev, p[idx] * m / (rank + 1))
        adj[idx] = min(prev, 1.0)
    return adj
