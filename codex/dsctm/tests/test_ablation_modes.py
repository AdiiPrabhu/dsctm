import torch

from dsctm.experiments.ablation import ABLATIONS
from dsctm.models import DMSTCN, DMSTCNConfig


def test_prespecified_phase5_control_family_is_complete():
    assert set(ABLATIONS) == {
        "full", "noSSB", "noMSB", "noLSB", "1scale_SSB", "1scale_MSB",
        "1scale_LSB", "noCSAG", "staticCSAG", "tempLow", "tempHigh",
        "noAdapter", "globalAdapter", "matchedGlobal",
    }


def test_static_csag_and_global_film_modes_have_expected_semantics():
    x = torch.randn(2, 20, 8)
    subjects = torch.tensor([1, 2])
    static = DMSTCN(DMSTCNConfig(D=16, head_hidden=16, n_subjects=3,
                                 csag_mode="static"))
    assert static.static_alpha.shape == (3,)
    assert static(x, subjects).shape == (2, 3)

    global_model = DMSTCN(DMSTCNConfig(D=16, head_hidden=16, n_subjects=3,
                                       film_mode="global")).eval()
    matched = DMSTCN(DMSTCNConfig(D=16, head_hidden=16, n_subjects=3,
                                  film_mode="global_matched")).eval()
    with torch.no_grad():
        assert torch.allclose(global_model(x, subjects),
                              global_model(x, torch.zeros_like(subjects)))
        assert torch.allclose(matched(x, subjects),
                              matched(x, torch.zeros_like(subjects)))
    assert global_model.film.embed.num_embeddings == 1
    assert matched.film.embed.num_embeddings == 3
