"""SAM prompt encoder: points, boxes and mask prompts to sparse and dense embeddings.

Sparse prompts (points and box corners) become one 256-d token each: a random Fourier
encoding of the location plus a learned embedding for the prompt type. A dense mask
prompt is downscaled by a small conv stack to the 64x64 embedding grid; without one, a
learned "no mask" vector is broadcast over the grid.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

from sam2_scratch.common import LayerNorm2d

# Point label conventions shared with the predictor.
LABEL_PAD = -1
LABEL_NEG = 0
LABEL_POS = 1
LABEL_BOX_TL = 2
LABEL_BOX_BR = 3


class PositionEmbeddingRandom(nn.Module):
    """Random Fourier features over normalised (x, y) coordinates.

    A fixed Gaussian matrix projects 2-D coordinates to ``num_pos_feats`` frequencies;
    the encoding is ``cat(sin, cos)`` so the output has ``2 * num_pos_feats`` channels.
    The matrix is a buffer (not trained), but it is stored in the checkpoint.
    """

    positional_encoding_gaussian_matrix: Tensor

    def __init__(self, num_pos_feats: int = 128, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer(
            "positional_encoding_gaussian_matrix", scale * torch.randn((2, num_pos_feats))
        )

    def _pe_encoding(self, coords: Tensor) -> Tensor:
        """Encode coordinates already normalised to [0, 1]: ``[..., 2] -> [..., C]``."""
        # Map to [-1, 1] so the projection is centred on the image.
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: tuple[int, int]) -> Tensor:
        """Dense encoding of an ``(h, w)`` grid at pixel centres: ``[C, h, w]``."""
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        # cumsum - 0.5 gives the centre of each cell (0.5, 1.5, ...), then normalise.
        y_embed = (grid.cumsum(dim=0) - 0.5) / h
        x_embed = (grid.cumsum(dim=1) - 0.5) / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    def forward_with_coords(self, coords_input: Tensor, image_size: tuple[int, int]) -> Tensor:
        """Encode unnormalised ``(x, y)`` pixel coordinates in an ``(h, w)`` image."""
        coords = coords_input.clone()
        coords[..., 0] = coords[..., 0] / image_size[1]
        coords[..., 1] = coords[..., 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))


class PromptEncoder(nn.Module):
    """Encodes prompts for the SAM mask decoder.

    ``point_embeddings`` index the prompt type: 0 negative, 1 positive, 2 box top-left,
    3 box bottom-right. Padding points (label -1) carry ``not_a_point_embed`` and no
    positional signal.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        image_embedding_size: tuple[int, int] = (64, 64),
        input_image_size: tuple[int, int] = (1024, 1024),
        mask_in_chans: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings = 4
        self.point_embeddings = nn.ModuleList(
            nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)
        )
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        # Mask prompts arrive at 4x the embedding resolution (256x256) and are reduced by
        # two stride-2 convs before a 1x1 lift to the embedding width.
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> Tensor:
        """Positional encoding of the image embedding grid: ``[1, C, 64, 64]``."""
        pe: Tensor = self.pe_layer(self.image_embedding_size)
        return pe.unsqueeze(0)

    def _type_embed(self, label: int) -> Tensor:
        """Learned embedding for a point type, ``[1, C]``."""
        return cast(nn.Embedding, self.point_embeddings[label]).weight

    def _embed_points(self, points: Tensor, labels: Tensor, pad: bool) -> Tensor:
        # Shift to pixel centres so the encoding agrees with get_dense_pe's grid.
        points = points + 0.5
        if pad:
            # Without a box the decoder was trained with a trailing padding token.
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label.to(labels.dtype)], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)

        lab = labels.unsqueeze(-1)
        point_embedding = torch.where(
            lab == LABEL_PAD,
            torch.zeros_like(point_embedding) + self.not_a_point_embed.weight,
            point_embedding,
        )
        for label in range(self.num_point_embeddings):
            point_embedding = torch.where(
                lab == label, point_embedding + self._type_embed(label), point_embedding
            )
        return point_embedding

    def _embed_boxes(self, boxes: Tensor) -> Tensor:
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self._type_embed(LABEL_BOX_TL)[0]
        corner_embedding[:, 1, :] += self._type_embed(LABEL_BOX_BR)[0]
        return corner_embedding

    def _batch_size(
        self, points: tuple[Tensor, Tensor] | None, boxes: Tensor | None, masks: Tensor | None
    ) -> int:
        if points is not None:
            return int(points[0].shape[0])
        if boxes is not None:
            return int(boxes.shape[0])
        if masks is not None:
            return int(masks.shape[0])
        return 1

    def forward(
        self,
        points: tuple[Tensor, Tensor] | None = None,
        boxes: Tensor | None = None,
        masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(sparse [B, N, C], dense [B, C, 64, 64])``.

        ``points`` is ``(coords [B, P, 2], labels [B, P])`` in input-image pixels (1024
        space); ``boxes`` is ``[B, 4]`` xyxy in the same space; ``masks`` is logits
        ``[B, 1, 256, 256]``.
        """
        bs = self._batch_size(points, boxes, masks)
        device = self.no_mask_embed.weight.device
        sparse = torch.empty((bs, 0, self.embed_dim), device=device)
        if points is not None:
            coords, labels = points
            sparse = torch.cat([sparse, self._embed_points(coords, labels, boxes is None)], dim=1)
        if boxes is not None:
            sparse = torch.cat([sparse, self._embed_boxes(boxes)], dim=1)

        if masks is not None:
            dense = self.mask_downscaling(masks)
        else:
            h, w = self.image_embedding_size
            dense = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(bs, -1, h, w)
        return sparse, dense
