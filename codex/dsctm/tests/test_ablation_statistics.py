import numpy as np

from dsctm.eval import statistics as st


def test_ablation_multiplicity_helpers_preserve_shape_and_bounds():
    p = np.array([0.01, 0.04, 0.2, 0.8])
    for adjusted in (st.holm_bonferroni(p), st.benjamini_hochberg(p)):
        assert adjusted.shape == p.shape
        assert np.all((adjusted >= 0) & (adjusted <= 1))
