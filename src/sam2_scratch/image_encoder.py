"""SAM 2 image encoder: Hiera trunk followed by a feature pyramid neck."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from .configs import HieraConfig, NeckConfig
from .hiera import Hiera
from .position_encoding import PositionEmbeddingSine
from .weights import load_prefix


class _Lateral(nn.Module):
    """Wrapper so the 1x1 conv sits at ``convs.{i}.conv`` like the checkpoint."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.conv(x)
        return out


class FpnNeck(nn.Module):
    """Projects every trunk stage to ``d_model`` channels with optional top-down fusion.

    ``convs`` are ordered coarsest stage first (``backbone_channel_list``), while inputs
    and outputs are ordered finest first. Level indices in ``fpn_top_down_levels`` refer
    to the output order; only those levels add the upsampled coarser output (summed, not
    averaged), the rest are purely lateral.
    """

    def __init__(self, cfg: NeckConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.convs = nn.ModuleList(_Lateral(c, cfg.d_model) for c in cfg.backbone_channel_list)
        self.position_encoding = PositionEmbeddingSine(
            cfg.pos_num_feats, temperature=cfg.pos_temperature
        )
        self.top_down_levels = set(cfg.fpn_top_down_levels)

    def forward(self, xs: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        n = len(self.convs)
        if len(xs) != n:
            raise ValueError(f"expected {n} trunk levels, got {len(xs)}")
        out: list[Tensor | None] = [None] * n
        pos: list[Tensor | None] = [None] * n
        prev: Tensor | None = None
        for i in range(n - 1, -1, -1):
            lateral = self.convs[n - 1 - i](xs[i])
            if i in self.top_down_levels and prev is not None:
                prev = lateral + F.interpolate(prev, scale_factor=2.0, mode="nearest")
            else:
                prev = lateral
            out[i] = prev
            pos[i] = self.position_encoding(prev).to(prev.dtype)
        return [t for t in out if t is not None], [p for p in pos if p is not None]


class ImageEncoder(nn.Module):
    """Trunk + neck. Drops the ``scalp`` coarsest levels, which SAM 2 never consumes."""

    def __init__(self, hiera_cfg: HieraConfig, neck_cfg: NeckConfig) -> None:
        super().__init__()
        if tuple(reversed(hiera_cfg.channel_list)) != neck_cfg.backbone_channel_list:
            raise ValueError("neck backbone_channel_list does not match the trunk widths")
        self.trunk = Hiera(hiera_cfg)
        self.neck = FpnNeck(neck_cfg)
        self.scalp = neck_cfg.scalp

    @classmethod
    def from_checkpoint(
        cls, state_dict: dict[str, Tensor], hiera_cfg: HieraConfig, neck_cfg: NeckConfig
    ) -> ImageEncoder:
        """Build the encoder and strictly load the ``image_encoder.`` checkpoint slice."""
        enc = cls(hiera_cfg, neck_cfg)
        load_prefix(enc, state_dict, "image_encoder")
        return enc.eval()

    def forward(self, x: Tensor) -> dict[str, Tensor | list[Tensor]]:
        features, pos = self.neck(self.trunk(x))
        if self.scalp > 0:
            features, pos = features[: -self.scalp], pos[: -self.scalp]
        return {"vision_features": features[-1], "vision_pos_enc": pos, "backbone_fpn": features}
