"""Fixed 2-D sinusoidal position encoding for feature maps (DETR style).

The encoding depends only on the spatial size of the map, so it is computed once per
``(H, W, device)`` and reused; it carries no parameters and nothing is saved in the
state dict.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal encoding of normalised (y, x) pixel coordinates.

    ``num_pos_feats`` is the total number of output channels: half encode the row
    coordinate, half the column coordinate (rows first), and within each half sine and
    cosine alternate channel by channel.
    """

    def __init__(
        self,
        num_pos_feats: int,
        temperature: float = 10000,
        normalize: bool = True,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        if num_pos_feats % 4 != 0:
            raise ValueError("num_pos_feats must be divisible by 4 (2 axes x sin/cos)")
        if scale is not None and not normalize:
            raise ValueError("scale only applies when normalize=True")
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale
        self._cache: dict[tuple[int, int, str], Tensor] = {}

    def _encode(self, h: int, w: int, device: torch.device) -> Tensor:
        # Coordinates start at 1 so that normalisation by the last index maps the grid
        # onto (0, scale]; the epsilon guards against a zero-sized axis.
        y = torch.arange(1, h + 1, dtype=torch.float32, device=device)[:, None].expand(h, w)
        x = torch.arange(1, w + 1, dtype=torch.float32, device=device)[None, :].expand(h, w)
        if self.normalize:
            eps = 1e-6
            y = y / (y[-1:, :] + eps) * self.scale
            x = x / (x[:, -1:] + eps) * self.scale

        n = self.num_pos_feats
        dim_t = torch.arange(n, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / n)

        def interleave(coord: Tensor) -> Tensor:
            p = coord[:, :, None] / dim_t  # [H, W, n]
            return torch.stack((p[:, :, 0::2].sin(), p[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((interleave(y), interleave(x)), dim=2)  # [H, W, 2n]
        return pos.permute(2, 0, 1).contiguous()  # [2n, H, W]

    def forward(self, x: Tensor) -> Tensor:
        """Return a ``[B, num_pos_feats, H, W]`` float32 encoding for ``x`` of ``[B, C, H, W]``."""
        b, _, h, w = x.shape
        key = (h, w, str(x.device))
        pos = self._cache.get(key)
        if pos is None:
            pos = self._encode(h, w, x.device)
            self._cache[key] = pos
        return pos[None].expand(b, -1, -1, -1)
