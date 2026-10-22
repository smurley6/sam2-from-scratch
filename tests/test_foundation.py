from __future__ import annotations

import torch

from sam2_scratch import configs
from sam2_scratch.common import MLP, LayerNorm2d, sine_pe_1d
from sam2_scratch.weights import load_checkpoint, slice_prefix
from tests.conftest import CHECKPOINTS, _require


def test_config_channels() -> None:
    assert configs.TINY.hiera.channel_list == (96, 192, 384, 768)
    assert configs.SMALL.hiera.stages == (1, 2, 11, 2)
    assert configs.TINY.image_embedding_size == 64


def test_mlp_matches_checkpoint_layout() -> None:
    mlp = MLP(256, 256, 32, 3)
    names = {n for n, _ in mlp.named_parameters()}
    assert names == {f"layers.{i}.{p}" for i in range(3) for p in ("weight", "bias")}
    out = mlp(torch.zeros(2, 256))
    assert out.shape == (2, 32)


def test_layernorm2d_normalises_channels() -> None:
    ln = LayerNorm2d(8)
    x = torch.randn(1, 8, 4, 4)
    y = ln(x)
    assert torch.allclose(y.mean(1), torch.zeros(1, 4, 4), atol=1e-5)


def test_sine_pe_shape() -> None:
    assert sine_pe_1d(torch.tensor([0.0, 0.5]), 64).shape == (2, 64)


def test_checkpoint_param_count() -> None:
    sd = load_checkpoint(_require(CHECKPOINTS["tiny"]))
    total = sum(v.numel() for v in sd.values())
    assert abs(total / 1e6 - 38.96) < 0.01
    assert len(slice_prefix(sd, "sam_prompt_encoder")) == 17
