"""Two-way transformer of the SAM mask decoder.

Prompt tokens and image tokens attend to each other in both directions: tokens
self-attend, tokens read from the image, an MLP updates the tokens, and the image reads
back from the tokens. Positional encodings are re-added to queries and keys before every
attention rather than once at the input, so the values stay free of position signal.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from sam2_scratch.common import MLP


class Attention(nn.Module):
    """Multi-head attention with an optional reduced internal width.

    ``downsample_rate`` shrinks the projected width (256 -> 128 for the cross attentions)
    to make attention over the 4096 image tokens cheaper; ``kv_in_dim`` lets keys and
    values come from a space of a different width than the queries.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        kv_in_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        if self.internal_dim % num_heads != 0:
            raise ValueError("num_heads must divide embedding_dim // downsample_rate")
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, h, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n, h * d)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self._separate_heads(self.q_proj(q))
        k = self._separate_heads(self.k_proj(k))
        v = self._separate_heads(self.v_proj(v))
        out = F.scaled_dot_product_attention(q, k, v)
        result: Tensor = self.out_proj(self._recombine_heads(out))
        return result


class TwoWayAttentionBlock(nn.Module):
    """One token <-> image round: self attn, token->image, MLP, image->token.

    ``skip_first_layer_pe`` is set on the first block, whose inputs are the raw prompt
    embeddings that already contain the positional encoding; adding it again would
    double count it, so that block's self attention runs on the tokens as they are and
    replaces them instead of adding a residual.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, 2, activation=nn.ReLU)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q, q, queries)
        queries = self.norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(q, k, keys))

        queries = self.norm3(queries + self.mlp(queries))

        q = queries + query_pe
        k = keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(k, q, queries))
        return queries, keys


class TwoWayTransformer(nn.Module):
    """Stack of two-way blocks followed by a final token->image attention."""

    def __init__(
        self,
        depth: int = 2,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TwoWayAttentionBlock(
                embedding_dim,
                num_heads,
                mlp_dim=mlp_dim,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=i == 0,
            )
            for i in range(depth)
        )
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self, image_embedding: Tensor, image_pe: Tensor, point_embedding: Tensor
    ) -> tuple[Tensor, Tensor]:
        """``image_embedding``/``image_pe`` are ``[B, C, H, W]``; tokens ``[B, N, C]``.

        Returns ``(tokens [B, N, C], image tokens [B, H*W, C])``.
        """
        keys = image_embedding.flatten(2).permute(0, 2, 1)
        key_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries = point_embedding
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_embedding, key_pe)

        q = queries + point_embedding
        k = keys + key_pe
        queries = self.norm_final_attn(queries + self.final_attn_token_to_image(q, k, keys))
        return queries, keys
