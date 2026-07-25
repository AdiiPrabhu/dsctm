import pytest
import torch

from dmstcn import DMSTCN, DMSTCNConfig
from dmstcn.model import CausalConv1d, theoretical_receptive_field


@pytest.fixture
def model() -> DMSTCN:
    torch.manual_seed(7)
    return DMSTCN(
        DMSTCNConfig(
            input_dim=6,
            num_classes=3,
            num_subjects=5,
            hidden_dim=12,
            dropout=0.0,
            branch_dilations=((1, 2), (2, 4), (4, 8)),
        )
    ).eval()


def test_output_and_attention_shapes(model: DMSTCN) -> None:
    result = model(torch.randn(4, 17, 6), torch.tensor([0, 1, 2, 3]))
    assert result.logits.shape == (4, 3)
    assert result.attention.shape == (4, 17, 3)
    torch.testing.assert_close(result.attention.sum(dim=-1), torch.ones(4, 17))


def test_causal_convolution_has_no_future_dependence() -> None:
    torch.manual_seed(3)
    convolution = CausalConv1d(channels=2, kernel_size=3, dilation=2).eval()
    baseline = torch.randn(1, 2, 12)
    changed = baseline.clone()
    changed[:, :, 7:] += 100
    torch.testing.assert_close(convolution(baseline)[:, :, :7], convolution(changed)[:, :, :7])


def test_masked_padding_does_not_change_logits(model: DMSTCN) -> None:
    prefix = torch.randn(2, 10, 6)
    padded = torch.cat((prefix, torch.randn(2, 4, 6)), dim=1)
    ids = torch.tensor([1, 4])
    prefix_result = model(prefix, ids).logits
    mask = torch.cat((torch.ones(2, 10), torch.zeros(2, 4)), dim=1).bool()
    padded_result = model(padded, ids, mask).logits
    torch.testing.assert_close(prefix_result, padded_result, atol=1e-6, rtol=1e-5)


def test_default_receptive_fields_match_two_convolution_graph() -> None:
    config = DMSTCNConfig(input_dim=2, num_classes=2, num_subjects=2)
    assert DMSTCN(config).receptive_fields == (61, 481, 1921)
    assert theoretical_receptive_field(3, (1, 2, 4, 8)) == 61


def test_invalid_even_kernel_is_rejected() -> None:
    with pytest.raises(ValueError, match="odd"):
        DMSTCNConfig(input_dim=2, num_classes=2, num_subjects=2, kernel_size=4)

