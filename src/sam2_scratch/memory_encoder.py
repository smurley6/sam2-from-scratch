"""SAM 2 memory encoder: fuse a predicted mask with its frame's image embedding.

The output is the per-frame entry of the memory bank: a 64-channel map at the 64x64
embedding resolution plus its sinusoidal position encoding. The mask is brought down to
the embedding grid by a strided conv stack, added to a projection of the (unconditioned)
image embedding, mixed spatially by two ConvNeXt blocks and projected to the narrow
memory width, which keeps the bank cheap to attend over.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from sam2_scratch.common import LayerNorm2d
from sam2_scratch.configs import MemoryEncoderConfig
from sam2_scratch.position_encoding import PositionEmbeddingSine


class MaskDownSampler(nn.Module):
    """Mask logits ``[B, 1, 1024, 1024]`` to ``[B, 256, 64, 64]`` features.

    Each stage halves the resolution and multiplies the channels by ``stride**2`` (4), so
    no information is discarded by the striding: 1 -> 4 -> 16 -> 64 -> 256 channels over
    four stages for the total stride of 16. A final 1x1 conv mixes the channels.
    Indices inside ``encoder`` follow the checkpoint (conv, norm, act per stage).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        total_stride: int = 16,
    ) -> None:
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        if stride**num_layers != total_stride:
            raise ValueError("total_stride must be a power of stride")
        layers: list[nn.Module] = []
        in_ch, out_ch = 1, 1
        for _ in range(num_layers):
            out_ch = in_ch * stride**2
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
                LayerNorm2d(out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch
        layers.append(nn.Conv2d(out_ch, embed_dim, kernel_size=1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.encoder(x)
        return out


class CXBlock(nn.Module):
    """ConvNeXt block: depthwise 7x7 conv, LayerNorm, pointwise MLP, layer scale, residual.

    The pointwise MLP is written as ``nn.Linear`` on a channels-last view, as in the
    checkpoint (``pwconv1``/``pwconv2`` are Linear weights, not 1x1 conv weights).
    """

    def __init__(
        self,
        dim: int = 256,
        kernel_size: int = 7,
        padding: int = 3,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(self.dwconv(x))
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC for the Linear layers
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = self.gamma * x
        return residual + x.permute(0, 3, 1, 2)


class Fuser(nn.Module):
    """A stack of CX blocks applied after the mask and image features are summed."""

    def __init__(self, dim: int, num_layers: int, kernel_size: int, padding: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            CXBlock(dim, kernel_size=kernel_size, padding=padding) for _ in range(num_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MemoryEncoder(nn.Module):
    """Mask + image embedding -> one memory-bank entry (features and position encoding)."""

    def __init__(self, cfg: MemoryEncoderConfig) -> None:
        super().__init__()
        self.mask_downsampler = MaskDownSampler(
            embed_dim=cfg.in_dim,
            kernel_size=cfg.mask_down_kernel,
            stride=cfg.mask_down_stride,
            padding=cfg.mask_down_padding,
            total_stride=cfg.mask_down_total_stride,
        )
        self.pix_feat_proj = nn.Conv2d(cfg.in_dim, cfg.in_dim, kernel_size=1)
        self.fuser = Fuser(
            cfg.in_dim, cfg.fuser_layers, kernel_size=cfg.fuser_kernel, padding=cfg.fuser_padding
        )
        self.out_proj = nn.Conv2d(cfg.in_dim, cfg.out_dim, kernel_size=1)
        self.position_encoding = PositionEmbeddingSine(
            cfg.pos_num_feats, temperature=cfg.pos_temperature
        )

    def forward(
        self, pix_feat: Tensor, masks: Tensor, skip_mask_sigmoid: bool = False
    ) -> dict[str, Tensor | list[Tensor]]:
        """Encode ``masks [B, 1, 1024, 1024]`` with ``pix_feat [B, 256, 64, 64]``.

        Returns ``{"vision_features": [B, 64, 64, 64], "vision_pos_enc": [[B, 64, 64,
        64]]}``. ``skip_mask_sigmoid`` is set by the video loop, which already maps the
        logits through its own scaled sigmoid before calling in.
        """
        if not skip_mask_sigmoid:
            masks = torch.sigmoid(masks)
        mask_feat = self.mask_downsampler(masks)
        x = self.pix_feat_proj(pix_feat.to(mask_feat.device)) + mask_feat
        x = self.out_proj(self.fuser(x))
        pos = self.position_encoding(x).to(x.dtype)
        return {"vision_features": x, "vision_pos_enc": [pos]}
