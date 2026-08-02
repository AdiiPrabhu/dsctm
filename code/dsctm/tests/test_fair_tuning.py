from dsctm.experiments.fair_tuning import MODELS, SEARCH


def test_equal_prespecified_search_budget_and_model_specific_spaces():
    assert set(SEARCH) == set(MODELS)
    assert {len(v) for v in SEARCH.values()} == {8}
    assert "hidden" in SEARCH["lstm"][0]
    assert "D" in SEARCH["dmstcn"][0]
    assert "d_model" in SEARCH["timesnet"][0]
