"""Hiera trunk (Ryali et al. 2023) as configured for SAM 2.1.

A hierarchical ViT: a strided patch embedding followed by four stages of transformer
blocks. Tokens stay channels-last ``[B, H, W, C]`` throughout. The first block of every
stage after the first halves the token grid by max-pooling its queries (and its
residual), while doubling width and head count. Most blocks attend inside
non-overlapping windows; a few listed blocks attend globally.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .common import MLP
from .configs import HieraConfig


def _max_pool(x: Tensor, stride: tuple[int, int]) -> Tensor:
    """2-D max pool of a channels-last map with kernel == stride (floor division)."""
    x = x.permute(0, 3, 1, 2)
    x = F.max_pool2d(x, kernel_size=stride, stride=stride)
    return x.permute(0, 2, 3, 1)


def window_partition(x: Tensor, window: int) -> tuple[Tensor, tuple[int, int]]:
    """Split ``[B, H, W, C]`` into ``[B * nW, window, window, C]`` windows.

    The grid is zero-padded on the bottom/right to a window multiple; the padded size
    is returned so the inverse can crop it off again.
    """
    b, h, w, c = x.shape
    pad_h = (window - h % window) % window
    pad_w = (window - w % window) % window
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w
    x = x.view(b, hp // window, window, wp // window, window, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window, window, c)
    return windows, (hp, wp)


def window_unpartition(
    windows: Tensor, window: int, padded_hw: tuple[int, int], hw: tuple[int, int]
) -> Tensor:
    """Inverse of :func:`window_partition`, cropping the padding away."""
    hp, wp = padded_hw
    h, w = hw
    b = windows.shape[0] // ((hp // window) * (wp // window))
    x = windows.view(b, hp // window, wp // window, window, window, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, -1)
    return x[:, :h, :w, :].contiguous()


class PatchEmbed(nn.Module):
    def __init__(self, cfg: HieraConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            3,
            cfg.embed_dim,
            kernel_size=cfg.patch_kernel,
            stride=cfg.patch_stride,
            padding=cfg.patch_padding,
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.proj(x)
        return out.permute(0, 2, 3, 1)


class MultiScaleAttention(nn.Module):
    """Multi-head self-attention whose queries may be pooled to a coarser grid.

    Keys and values keep the input resolution, so a pooled block still sees every token
    of its window while producing one output per pooled query.
    """

    def __init__(
        self, dim: int, dim_out: int, num_heads: int, q_stride: tuple[int, int] | None
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.q_stride = q_stride
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: Tensor) -> Tensor:
        b, h, w, _ = x.shape
        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, dim=2)  # each [B, HW, heads, head_dim]
        if self.q_stride is not None:
            q = _max_pool(q.reshape(b, h, w, -1), self.q_stride)
            h, w = q.shape[1], q.shape[2]
            q = q.reshape(b, h * w, self.num_heads, -1)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        out = out.transpose(1, 2).reshape(b, h, w, -1)
        projected: Tensor = self.proj(out)
        return projected


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float,
        q_stride: tuple[int, int] | None,
        window_size: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.q_stride = q_stride
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiScaleAttention(dim, dim_out, num_heads, q_stride)
        self.norm2 = nn.LayerNorm(dim_out, eps=1e-6)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, 2, activation=nn.GELU)
        # Only stage-transition blocks change width; their residual is projected from the
        # normalised input (not the raw one) and pooled like the queries.
        self.proj: nn.Linear | None = nn.Linear(dim, dim_out) if dim != dim_out else None

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.proj is not None:
            shortcut = self.proj(x)
            if self.q_stride is not None:
                shortcut = _max_pool(shortcut, self.q_stride)

        window = self.window_size
        if window > 0:
            # Partition on the input grid with this block's window. A q-pooling block
            # shrinks every window by the stride, so the output is reassembled with the
            # smaller window on the pooled grid (padded to that window's multiple).
            x, _ = window_partition(x, window)
        x = self.attn(x)
        if window > 0:
            if self.q_stride is not None:
                window = window // self.q_stride[0]
            h, w = shortcut.shape[1], shortcut.shape[2]
            padded = (h + (window - h % window) % window, w + (window - w % window) % window)
            x = window_unpartition(x, window, padded, (h, w))

        x = shortcut + x
        mlp_out: Tensor = self.mlp(self.norm2(x))
        return x + mlp_out


class Hiera(nn.Module):
    """Returns the four stage outputs as ``[B, C, H, W]``, highest resolution first."""

    def __init__(self, cfg: HieraConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, cfg.embed_dim, *cfg.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, cfg.embed_dim, cfg.window_spec[0], cfg.window_spec[0])
        )

        depth = sum(cfg.stages)
        self.stage_ends = [sum(cfg.stages[: i + 1]) - 1 for i in range(len(cfg.stages))]
        q_pool_blocks = {end + 1 for end in self.stage_ends[:-1][: cfg.q_pool]}

        blocks: list[MultiScaleBlock] = []
        dim, heads, stage = cfg.embed_dim, cfg.num_heads, 0
        for i in range(depth):
            # A transition block still windows its (pre-pooling) input with the previous
            # stage's window size, so the window is chosen before the stage advances.
            window = 0 if i in cfg.global_att_blocks else cfg.window_spec[stage]
            dim_out = dim
            if i - 1 in self.stage_ends:
                dim_out = int(dim * cfg.dim_mul)
                heads = int(heads * cfg.head_mul)
                stage += 1
            blocks.append(
                MultiScaleBlock(
                    dim=dim,
                    dim_out=dim_out,
                    num_heads=heads,
                    mlp_ratio=cfg.mlp_ratio,
                    q_stride=cfg.q_stride if i in q_pool_blocks else None,
                    window_size=window,
                )
            )
            dim = dim_out
        self.blocks = nn.ModuleList(blocks)

    def _pos_embed(self, h: int, w: int) -> Tensor:
        """Background embedding resized to the grid plus the window embedding tiled on it."""
        bkg = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        win = self.pos_embed_window
        wh, ww = win.shape[-2:]
        if h % wh or w % ww:
            raise ValueError(f"token grid {h}x{w} is not a multiple of the {wh}x{ww} window")
        pos = bkg + win.tile(1, 1, h // wh, w // ww)
        return pos.permute(0, 2, 3, 1)

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.patch_embed(x)
        x = x + self._pos_embed(x.shape[1], x.shape[2])
        outputs: list[Tensor] = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.stage_ends:
                outputs.append(x.permute(0, 3, 1, 2))
        return outputs
