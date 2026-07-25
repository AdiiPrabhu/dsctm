import torch

from dsctm.models.baselines import build_baseline


def test_official_timesnet_shape_and_mask():
    model = build_baseline("timesnet", 8, 3, seq_len=60).eval()
    x = torch.randn(2, 60, 8)
    mask = torch.ones(2, 60, dtype=torch.bool)
    with torch.no_grad():
        out = model(x, mask=mask)
    assert out.shape == (2, 3)
