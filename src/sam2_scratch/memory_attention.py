"""SAM 2 memory attention: condition the current frame's tokens on the memory bank.

Each layer runs self attention over the 64x64 current-frame tokens, cross attention from
those tokens into the memory bank (spatial memories of past frames followed by object
pointer tokens), and an MLP, all pre-norm with residuals. Position information enters
the attentions through 2-D rotary embeddings (RoPE) rather than additive encodings, so
that every memory frame, which shares the current frame's grid, is compared position by
position with the query grid. Object pointers have no spatial position and are left
un-rotated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sam2_scratch.configs import MemoryAttentionConfig


def axial_rope_cis(head_dim: int, end_x: int, end_y: int, theta: float = 10000.0) -> Tensor:
    """Unit rotations for 2-D axial RoPE as ``[end_x * end_y, head_dim // 2, 2]`` real
    (cos, sin) pairs, i.e. ``view_as_real`` of the complex ``exp(i * angle)`` table.

    Channels are rotated in adjacent pairs. The first half of the ``head_dim // 2`` pairs
    rotates with the token's x (column) index and the second half with its y (row) index,
    both with the frequency ladder ``theta ** (-4 i / head_dim)``. Tokens are in
    row-major order (``t = y * end_x + x``).
    """
    if head_dim % 4 != 0:
        raise ValueError("2-D RoPE needs head_dim divisible by 4")
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 4)[: head_dim // 4].float() / head_dim))
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = t % end_x
    t_y = torch.div(t, end_x, rounding_mode="floor")
    angles = torch.cat([torch.outer(t_x, freqs), torch.outer(t_y, freqs)], dim=-1)
    return torch.view_as_real(torch.polar(torch.ones_like(angles), angles))


def apply_rope(x: Tensor, cis: Tensor) -> Tensor:
    """Rotate adjacent channel pairs of ``x [B, H, N, D]`` by ``cis [N, D/2, 2]``.

    Treating each pair as a complex number and multiplying by the unit rotation is the
    reference formulation, and with complex arithmetic the result is bit-identical to
    it. MPS lacks complex multiplication, so there the same product is expanded into
    real arithmetic (equal up to float rounding).
    """
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    if x.device.type == "mps":
        x0, x1 = pairs[..., 0], pairs[..., 1]
        cos, sin = cis[..., 0], cis[..., 1]
        out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    else:
        out = torch.view_as_real(torch.view_as_complex(pairs) * torch.view_as_complex(cis))
    return out.flatten(-2).type_as(x)


class RoPEAttention(nn.Module):
    """Multi-head attention with 2-D axial RoPE on queries and keys.

    ``kv_in_dim`` lets keys and values come from the 64-channel memory bank.
    ``rope_k_repeat`` tiles the query grid's rotation table over the keys when they hold
    several frames of the same grid (cross attention into the spatial memories). The
    last ``num_k_exclude_rope`` keys passed to ``forward`` are object pointers, which
    carry a temporal rather than spatial position and are therefore not rotated.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        kv_in_dim: int | None = None,
        rope_theta: float = 10000.0,
        rope_k_repeat: bool = False,
        feat_sizes: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.num_heads = num_heads
        if embedding_dim % num_heads != 0:
            raise ValueError("num_heads must divide embedding_dim")
        self.head_dim = embedding_dim // num_heads
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, embedding_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.rope_theta = rope_theta
        self.rope_k_repeat = rope_k_repeat
        self.feat_sizes = feat_sizes
        # Rotation tables depend only on the grid, so they are cached per (grid, device).
        self._rope_cache: dict[tuple[int, int, str], Tensor] = {}

    def _rope_table(self, n_tokens: int, device: torch.device) -> Tensor:
        # The query grid is assumed square (as at every supported resolution); the
        # configured feat_sizes is used when it matches, which also covers non-square.
        h, w = self.feat_sizes
        if h * w != n_tokens:
            side = int(round(n_tokens**0.5))
            if side * side != n_tokens:
                raise ValueError(f"cannot infer a 2-D grid for {n_tokens} query tokens")
            h = w = side
        key = (h, w, str(device))
        cis = self._rope_cache.get(key)
        if cis is None:
            cis = axial_rope_cis(self.head_dim, end_x=w, end_y=h, theta=self.rope_theta)
            cis = cis.to(device)
            self._rope_cache[key] = cis
        return cis

    def _separate_heads(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0) -> Tensor:
        """Batch-first inputs: ``q [B, Nq, C]``, ``k``/``v [B, Nk, kv_in_dim]``."""
        q = self._separate_heads(self.q_proj(q))
        k = self._separate_heads(self.k_proj(k))
        v = self._separate_heads(self.v_proj(v))

        cis = self._rope_table(q.shape[-2], q.device)
        q = apply_rope(q, cis)
        num_k_rope = k.shape[-2] - num_k_exclude_rope
        if num_k_rope != q.shape[-2]:
            if not self.rope_k_repeat or num_k_rope % q.shape[-2] != 0:
                raise ValueError(
                    f"{num_k_rope} rotated keys do not tile {q.shape[-2]} query positions"
                )
            # Every memory frame shares the query grid, so the table is tiled per frame.
            cis_k = cis.repeat(num_k_rope // q.shape[-2], 1, 1)
        else:
            cis_k = cis
        k = torch.cat([apply_rope(k[:, :, :num_k_rope], cis_k), k[:, :, num_k_rope:]], dim=2)

        out = F.scaled_dot_product_attention(q, k, v)
        b, h, n, d = out.shape
        result: Tensor = self.out_proj(out.transpose(1, 2).reshape(b, n, h * d))
        return result


class MemoryAttentionLayer(nn.Module):
    """Pre-norm self attention, cross attention into memory, then a ReLU MLP.

    With the released config the position encodings are only added to the memory keys
    (``pos_enc_at_cross_attn_keys``); the queries rely on RoPE alone.
    """

    def __init__(self, cfg: MemoryAttentionConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.self_attn = RoPEAttention(
            d, cfg.num_heads, rope_theta=cfg.rope_theta, feat_sizes=cfg.feat_sizes
        )
        self.cross_attn_image = RoPEAttention(
            d,
            cfg.num_heads,
            kv_in_dim=cfg.kv_in_dim,
            rope_theta=cfg.rope_theta,
            rope_k_repeat=cfg.rope_k_repeat_in_cross_attn,
            feat_sizes=cfg.feat_sizes,
        )
        self.linear1 = nn.Linear(d, cfg.dim_feedforward)
        self.linear2 = nn.Linear(cfg.dim_feedforward, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.activation = nn.ReLU()
        self.pos_enc_at_attn = False
        self.pos_enc_at_cross_attn_keys = cfg.pos_enc_at_cross_attn_keys
        self.pos_enc_at_cross_attn_queries = cfg.pos_enc_at_cross_attn_queries

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        pos: Tensor,
        query_pos: Tensor,
        num_k_exclude_rope: int = 0,
    ) -> Tensor:
        """Batch-first: ``tgt``/``query_pos [B, HW, C]``, ``memory``/``pos [B, N, C_mem]``."""
        x = self.norm1(tgt)
        qk = x + query_pos if self.pos_enc_at_attn else x
        tgt = tgt + self.self_attn(qk, qk, x)

        x = self.norm2(tgt)
        tgt = tgt + self.cross_attn_image(
            x + query_pos if self.pos_enc_at_cross_attn_queries else x,
            memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            memory,
            num_k_exclude_rope=num_k_exclude_rope,
        )

        x = self.norm3(tgt)
        mlp_out: Tensor = self.linear2(self.activation(self.linear1(x)))
        return tgt + mlp_out


class MemoryAttention(nn.Module):
    """Stack of memory attention layers with a final LayerNorm."""

    def __init__(self, cfg: MemoryAttentionConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(MemoryAttentionLayer(cfg) for _ in range(cfg.num_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.pos_enc_at_input = cfg.pos_enc_at_input

    def forward(
        self,
        curr: Tensor,
        curr_pos: Tensor,
        memory: Tensor,
        memory_pos: Tensor,
        num_obj_ptr_tokens: int = 0,
    ) -> Tensor:
        """Sequence-first: ``curr``/``curr_pos [HW, B, C]``, ``memory``/``memory_pos [N, B,
        C_mem]`` with the ``num_obj_ptr_tokens`` pointer tokens last. Returns ``[HW, B, C]``.
        """
        output = curr
        if self.pos_enc_at_input:
            # The input position encoding is added at a small fixed weight: the RoPE in
            # every attention already carries the precise spatial signal.
            output = output + 0.1 * curr_pos
        output = output.transpose(0, 1)
        query_pos = curr_pos.transpose(0, 1)
        mem = memory.transpose(0, 1)
        mem_pos = memory_pos.transpose(0, 1)
        for layer in self.layers:
            output = layer(output, mem, mem_pos, query_pos, num_k_exclude_rope=num_obj_ptr_tokens)
        normed: Tensor = self.norm(output)
        return normed.transpose(0, 1)
