"""The numpy helpers of ``tools/ort_video_loop.py`` that have no graph behind them: the
antialiased resize and the ``mask_downsample`` conv used by a mask prompt, against torch.

The loop itself (clicks, a box and a mask prompt through onnxruntime, three frames against
the torch predictor) is covered in ``tests/test_onnx_export.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from tools import ort_video_loop as loop

# (source h, w) -> (destination h, w): the shapes a mask prompt goes through at 512 and
# 1024 (video frame to S, S to S / 4), plus odd sizes in both directions.
RESIZE_CASES = [
    ((540, 960), (512, 512)),
    ((360, 640), (512, 512)),
    ((128, 128), (512, 512)),
    ((512, 512), (128, 128)),
    ((1024, 1024), (256, 256)),
    ((5, 7), (9, 4)),
    ((7, 5), (7, 5)),
]


@pytest.mark.parametrize(("src", "dst"), RESIZE_CASES)
def test_resize_antialias_matches_torch(src: tuple[int, int], dst: tuple[int, int]) -> None:
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1, 1, *src, generator=gen) * 10
    want = F.interpolate(x, size=dst, mode="bilinear", align_corners=False, antialias=True)
    got = loop.resize_antialias(x.numpy(), dst)
    assert got.shape == want.shape and got.dtype == np.float32
    assert np.abs(got - want.numpy()).max() < 2e-5


def test_resize_antialias_binary_mask_is_bilinear_when_enlarging() -> None:
    """Enlarging never mixes more than the two nearest source pixels, so the antialiased
    filter equals plain bilinear interpolation (the loop's ``upsample_bilinear``)."""
    gen = torch.Generator().manual_seed(1)
    m = (torch.rand(1, 1, 128, 128, generator=gen) > 0.5).float().numpy()
    got = loop.resize_antialias(m, (512, 512))
    plain = loop.upsample_bilinear(m, (512, 512))
    assert np.abs(got - plain).max() < 1e-6


def test_antialias_weights_are_the_torch_triangle_filter() -> None:
    w = loop._antialias_weights(8, 2)  # a 4x downsample, no border effects in the middle
    assert w.shape == (2, 8)
    assert np.allclose(w.sum(1), 1.0)
    # Interior taps of a 4x downsample: 1/32 * (1, 3, 5, 7, 7, 5, 3, 1) around the centre;
    # with 8 source pixels every tap is inside, row 0 covers pixels 0..5 renormalised.
    assert np.allclose(w[1], [0, 0, 1 / 28, 3 / 28, 5 / 28, 7 / 28, 7 / 28, 5 / 28])


def test_mask_downsample_matches_the_conv() -> None:
    gen = torch.Generator().manual_seed(2)
    weight = torch.randn(1, 1, 4, 4, generator=gen)
    bias = torch.randn(1, generator=gen)
    mask = (torch.rand(1, 1, 64, 64, generator=gen) > 0.5).float()
    want = F.conv2d(mask, weight, bias, stride=4)
    got = loop.mask_downsample(mask.numpy(), weight.numpy(), bias.numpy())
    assert got.shape == (1, 1, 16, 16)
    assert np.abs(got - want.numpy()).max() < 1e-5


def test_prompt_table() -> None:
    tags = [p.tag for p in loop.RUNS]
    assert tags == ["fwd", "rev", "box", "mask"]
    box = loop.RUNS[2]
    assert box.box is not None and box.points == () and box.mask_from is None
    mask = loop.RUNS[3]
    assert mask.mask_from == "fwd" and mask.box is None and mask.points == ()
