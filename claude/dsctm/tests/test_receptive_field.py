"""EXP-0.1 as executable assertions: the measured RF matches the two-conv formula
and NOT the manuscript's printed values."""
import torch

from dsctm.experiments.gate0 import exp_0_1_receptive_field


def test_measured_rf_matches_two_conv_formula():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = exp_0_1_receptive_field(dev)
    by = {r["branch"]: r for r in res["branch_rf"]}
    # two convs per block ⇒ 61 / 481 / 1921
    assert by["ssb"]["measured_rf"] == 61 == by["ssb"]["formula_two_conv"]
    assert by["msb"]["measured_rf"] == 481 == by["msb"]["formula_two_conv"]
    assert by["lsb"]["measured_rf"] == 1921 == by["lsb"]["formula_two_conv"]


def test_manuscript_printed_rf_is_wrong():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = exp_0_1_receptive_field(dev)
    for r in res["branch_rf"]:
        # 47 / 383 / 1535 match neither the measured value nor either formula
        assert r["manuscript_printed_rf"] != r["measured_rf"]
        assert r["manuscript_printed_rf"] != r["formula_one_conv"]
