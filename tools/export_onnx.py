"""Export SAM 2 to ONNX for ONNX Runtime Web: the image half and the video (memory) half.

Image graphs, one per input resolution ``S`` (1024 and 512):

``image_encoder_{S}.onnx``
    Hiera trunk + FPN neck + the mask decoder's ``conv_s0`` / ``conv_s1`` projections.
    ``image [1, 3, S, S]`` (normalised) -> ``image_embed [1, 256, S/16, S/16]`` (with
    ``no_mem_embed`` added, ready for single-image decoding), ``high_res_feat_s0
    [1, 32, S/4, S/4]``, ``high_res_feat_s1 [1, 64, S/8, S/8]``.

``sam_head_{S}.onnx``
    Prompt encoder + mask decoder, returning all four mask tokens. The dense positional
    encoding of the embedding grid is baked in as a constant. See ``exports/README.md``
    for the input/output contract.

Video graphs (``tools/ort_video_loop.py`` is the loop that drives them):

``memory_encoder_{S}.onnx``
    Raw stride-16 features + high-res mask logits -> one memory-bank entry. The
    binarise / scaled-sigmoid mask preprocessing and the gated ``no_obj_embed_spatial``
    term run inside the graph.

``memory_attention_{S}.onnx``
    Current frame tokens + memory bank (dynamic length) -> memory-conditioned embedding,
    with 2-D RoPE in real arithmetic; the stride-16 position encoding is built in.

``object_pointer.onnx`` / ``object_pointer_tpos.onnx``
    Object pointer of a frame (projection + occlusion switch) and the temporal encoding
    of pointer tokens; identical at every ``S``, so exported once.

``constants_{S}.json``
    The small tensors and scalars the loop needs outside the graphs (``no_mem_embed``,
    ``maskmem_tpos_enc``, normalisation, thresholds), so the browser never reads the
    checkpoint.

Precision variants: ``image_encoder_{S}.int8.onnx`` (int8 MatMul weights, see
``quantize_encoder_int8`` for the two schemes), and ``sam_head_{S}.fp16.onnx``,
``memory_encoder_{S}.fp16.onnx``, ``memory_attention_{S}.fp16.onnx`` (fp16 weights and
compute, fp32 I/O).

Export path: the legacy TorchScript exporter (``dynamo=False``, opset 17), then onnxslim
to fold the traced shape arithmetic so every encoder shape is static.

Usage (from the repo root, venv active)::

    python tools/export_onnx.py --model tiny
    python tools/export_onnx.py --model tiny --sizes 512 --precisions fp32
    python tools/export_onnx.py --model tiny --int8-scheme dynamic
    python tools/export_onnx.py --model tiny --graphs memory --sizes 512

Files for a model other than ``tiny`` carry the model name, e.g.
``image_encoder_small_1024.onnx``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import onnx
import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:  # allow ``python tools/export_onnx.py`` without install
    sys.path.insert(0, str(ROOT / "src"))

from sam2_scratch.common import MLP, sine_pe_1d  # noqa: E402
from sam2_scratch.configs import CONFIGS, SAM2Config  # noqa: E402
from sam2_scratch.image_encoder import ImageEncoder  # noqa: E402
from sam2_scratch.mask_decoder import MaskDecoder  # noqa: E402
from sam2_scratch.memory_attention import (  # noqa: E402
    MemoryAttention,
    MemoryAttentionLayer,
    RoPEAttention,
    axial_rope_cis,
)
from sam2_scratch.memory_encoder import MemoryEncoder  # noqa: E402
from sam2_scratch.model import NO_OBJ_SCORE  # noqa: E402
from sam2_scratch.position_encoding import PositionEmbeddingSine  # noqa: E402
from sam2_scratch.prompt_encoder import LABEL_PAD, PromptEncoder  # noqa: E402
from sam2_scratch.weights import load_checkpoint, load_prefix  # noqa: E402

EXPORTS = ROOT / "exports"
CHECKPOINTS = {
    "tiny": ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt",
    "small": ROOT / "checkpoints" / "sam2.1_hiera_small.pt",
}
OPSET = 17
SIZES = (1024, 512)
PRECISIONS = ("fp32", "int8", "fp16")
# Constant subgraphs whose folded output is larger than this stay in the graph (the
# Hiera position embedding); onnxruntime folds them at session creation instead.
FOLD_LIMIT_BYTES = 1 << 20
# Same for the memory attention, whose RoPE and position grids (256 kB to 4 MB) are
# built from separable tables of a few kB.
GRID_FOLD_LIMIT_BYTES = 64 << 10
INT8_SCHEMES = ("weight-only", "dynamic")
# Hiera blocks whose MatMul weights stay fp32 in the int8 encoder. The first blocks run
# on the stride-4/8 grids that feed the mask decoder's skip connections and dominate the
# quantisation error (per-block sweep in exports/README.md).
INT8_FP32_BLOCKS = {"weight-only": (0, 1), "dynamic": (0, 1, 2, 3)}

ENCODER_INPUTS = ["image"]
ENCODER_OUTPUTS = ["image_embed", "high_res_feat_s0", "high_res_feat_s1"]
HEAD_INPUTS = [
    "image_embed",
    "high_res_feat_s0",
    "high_res_feat_s1",
    "point_coords",
    "point_labels",
    "mask_input",
    "has_mask_input",
]
HEAD_OUTPUTS = ["masks", "iou_predictions", "object_score_logits", "sam_tokens"]
MEMORY_ENCODER_INPUTS = ["vision_feat", "mask_logits", "object_score_logits", "binarize"]
MEMORY_ENCODER_OUTPUTS = ["maskmem_features", "maskmem_pos_enc"]
MEMORY_ATTENTION_INPUTS = ["curr", "memory", "memory_pos", "num_obj_ptr_tokens"]
MEMORY_ATTENTION_OUTPUTS = ["fused", "image_embed"]
OBJECT_POINTER_INPUTS = ["mask_token", "object_score_logits"]
OBJECT_POINTER_OUTPUTS = ["obj_ptr"]
POINTER_TPOS_INPUTS = ["t_diff", "t_diff_max"]
POINTER_TPOS_OUTPUTS = ["ptr_pos"]
GRAPH_SETS = ("image", "memory")
# Upper bound for one attention score matrix (fp32 bytes) in the memory attention graph.
# Queries are split into static chunks so a full memory bank at 1024 px (28 736 keys)
# never needs one 470 MB buffer; WebGPU's default maxStorageBufferBindingSize is 128 MiB.
ATTENTION_SCORE_BUDGET = 64 << 20

# Ops to keep out of the browser graphs: ones without an ORT Web WebGPU kernel (they fall
# back to the WASM CPU provider, with a GPU<->CPU copy at each boundary), plus control
# flow and data-dependent shapes, which break static planning even where a kernel exists.
WEBGPU_UNSUPPORTED = frozenset(
    {
        "ConvInteger",
        "DynamicQuantizeLinear",
        "If",
        "Loop",
        "MatMulInteger",
        "NonMaxSuppression",
        "NonZero",
        "Scan",
        "Unique",
    }
)


# --------------------------------------------------------------------------------------
# Torch side
# --------------------------------------------------------------------------------------


@dataclass
class TorchSam:
    """The reimplemented modules of one checkpoint, in eval mode on CPU (fp32)."""

    name: str
    cfg: SAM2Config
    encoder: ImageEncoder
    mask_decoder: MaskDecoder
    no_mem_embed: Tensor  # (1, 1, 256)
    state_dict: dict[str, Tensor] = field(repr=False)
    _prompt_encoders: dict[int, PromptEncoder] = field(default_factory=dict, repr=False)
    _memory: TorchMemory | None = field(default=None, repr=False)

    def memory(self) -> TorchMemory:
        """The video-side modules and learned tensors, loaded on first use."""
        if self._memory is None:
            self._memory = TorchMemory.from_state_dict(self.cfg, self.state_dict)
        return self._memory

    def vision_pos_tokens(self, image_size: int) -> Tensor:
        """Sine position encoding of the stride-16 level as tokens ``[1, HW, C]``."""
        g = image_size // self.cfg.backbone_stride
        pe = PositionEmbeddingSine(self.cfg.neck.pos_num_feats, self.cfg.neck.pos_temperature)
        pos: Tensor = pe(torch.zeros(1, self.cfg.hidden_dim, g, g))
        return pos.flatten(2).transpose(1, 2).contiguous()

    def prompt_encoder(self, image_size: int) -> PromptEncoder:
        """Prompt encoder for an ``image_size`` input (grid ``image_size / 16``)."""
        if image_size not in self._prompt_encoders:
            grid = image_size // self.cfg.backbone_stride
            pe = PromptEncoder(
                embed_dim=self.cfg.prompt_embed_dim,
                image_embedding_size=(grid, grid),
                input_image_size=(image_size, image_size),
                mask_in_chans=self.cfg.mask_in_chans,
            )
            load_prefix(pe, self.state_dict, "sam_prompt_encoder")
            self._prompt_encoders[image_size] = pe.eval()
        return self._prompt_encoders[image_size]

    @torch.no_grad()
    def encode(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Reference encoder outputs: ``(image_embed, high_res_feat_s0, high_res_feat_s1)``."""
        out = self.encoder(image)
        fpn = out["backbone_fpn"]
        assert isinstance(fpn, list)
        vision = out["vision_features"]
        assert isinstance(vision, Tensor)
        embed = vision + self.no_mem_embed.reshape(1, -1, 1, 1)
        return embed, self.mask_decoder.conv_s0(fpn[0]), self.mask_decoder.conv_s1(fpn[1])

    @torch.no_grad()
    def decode(
        self,
        image_size: int,
        feats: tuple[Tensor, Tensor, Tensor],
        point_coords: Tensor,
        point_labels: Tensor,
        mask_input: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Reference head outputs ``(masks, iou, object_score_logits, sam_tokens)``.

        Uses the modules' own API exactly as the image predictor does (box corners go
        through the point path, so the prompt encoder appends the padding token).
        """
        pe = self.prompt_encoder(image_size)
        sparse, dense = pe(points=(point_coords, point_labels), boxes=None, masks=mask_input)
        embed, s0, s1 = feats
        masks, iou, tokens, obj = self.mask_decoder.predict_masks(
            embed, pe.get_dense_pe(), sparse, dense, [s0, s1]
        )
        return masks, iou, obj, tokens


@dataclass
class TorchMemory:
    """Memory encoder, memory attention and the object-pointer parameters (fp32, CPU)."""

    memory_encoder: MemoryEncoder
    memory_attention: MemoryAttention
    obj_ptr_proj: MLP
    obj_ptr_tpos_proj: nn.Linear
    no_obj_ptr: Tensor  # (1, C)
    no_obj_embed_spatial: Tensor  # (1, C_mem)
    maskmem_tpos_enc: Tensor  # (num_maskmem, 1, 1, C_mem)

    @classmethod
    def from_state_dict(cls, cfg: SAM2Config, sd: dict[str, Tensor]) -> TorchMemory:
        enc = MemoryEncoder(cfg.memory_encoder)
        load_prefix(enc, sd, "memory_encoder")
        attn = MemoryAttention(cfg.memory_attention)
        load_prefix(attn, sd, "memory_attention")
        c, m = cfg.hidden_dim, cfg.mem_dim
        proj = MLP(c, c, c, 3)
        load_prefix(proj, sd, "obj_ptr_proj")
        tpos = nn.Linear(c, m)
        load_prefix(tpos, sd, "obj_ptr_tpos_proj")

        def param(key: str, shape: tuple[int, ...]) -> Tensor:
            t = sd[key].detach().clone().float()
            if tuple(t.shape) != shape:
                raise ValueError(f"{key}: shape {tuple(t.shape)} != {shape}")
            return t

        return cls(
            enc.eval(),
            attn.eval(),
            proj.eval(),
            tpos.eval(),
            param("no_obj_ptr", (1, c)),
            param("no_obj_embed_spatial", (1, m)),
            param("maskmem_tpos_enc", (cfg.num_maskmem, 1, 1, m)),
        )


def load_torch(model: str, checkpoint: Path | None = None) -> TorchSam:
    cfg = CONFIGS[model]
    sd = load_checkpoint(checkpoint or CHECKPOINTS[model])
    encoder = ImageEncoder.from_checkpoint(sd, cfg.hiera, cfg.neck)
    md = MaskDecoder(
        transformer_dim=cfg.prompt_embed_dim,
        num_multimask_outputs=cfg.num_multimask_outputs,
        iou_head_depth=cfg.iou_head_depth,
        iou_head_hidden_dim=cfg.iou_head_hidden_dim,
        transformer_depth=cfg.decoder_depth,
        transformer_heads=cfg.decoder_heads,
        transformer_mlp_dim=cfg.decoder_mlp_dim,
    )
    load_prefix(md, sd, "sam_mask_decoder")
    no_mem_embed = sd["no_mem_embed"].detach().clone().float()
    if tuple(no_mem_embed.shape) != (1, 1, cfg.hidden_dim):
        raise ValueError(f"unexpected no_mem_embed shape {tuple(no_mem_embed.shape)}")
    return TorchSam(model, cfg, encoder.eval(), md.eval(), no_mem_embed, sd)


# --------------------------------------------------------------------------------------
# Export wrappers
# --------------------------------------------------------------------------------------


class ImageEncoderGraph(nn.Module):
    """Image encoder plus the per-image work the SAM head needs, for one input size.

    The trunk computes its position embedding in the graph (bicubic resize of the 7x7
    background table plus the tiled 8x8 window table), exactly as ``Hiera.forward``
    does. The result depends only on constants, so onnxruntime folds it once at
    session creation; keeping the two small tables in the file instead of the folded
    ``[1, S/4, S/4, 96]`` tensor saves 25 MB at 1024.
    """

    no_mem_embed: Tensor

    def __init__(self, sam: TorchSam, image_size: int) -> None:
        super().__init__()
        if image_size % 32:
            raise ValueError(f"image_size {image_size} must be a multiple of 32")
        self.image_size = image_size
        self.encoder = sam.encoder
        self.conv_s0 = sam.mask_decoder.conv_s0
        self.conv_s1 = sam.mask_decoder.conv_s1
        self.register_buffer("no_mem_embed", sam.no_mem_embed.reshape(1, -1, 1, 1).clone())

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.encoder(image)
        fpn = out["backbone_fpn"]
        vision = out["vision_features"]
        assert isinstance(fpn, list) and isinstance(vision, Tensor)
        return vision + self.no_mem_embed, self.conv_s0(fpn[0]), self.conv_s1(fpn[1])


class SamHeadGraph(nn.Module):
    """Prompt encoder + mask decoder for one input size, all four mask tokens out.

    Points follow ``PromptEncoder`` with ``boxes=None``: coordinates are in S-pixel
    space, a trailing padding token is appended inside the graph, and box corners are
    ordinary points with labels 2 and 3. The mask prompt is blended in with
    ``has_mask_input`` so one graph covers both the mask and the no-mask path.
    """

    dense_pe: Tensor
    no_mask_dense: Tensor

    def __init__(self, sam: TorchSam, image_size: int) -> None:
        super().__init__()
        pe = sam.prompt_encoder(image_size)
        self.image_size = image_size
        self.pe_layer = pe.pe_layer
        self.mask_downscaling = pe.mask_downscaling
        self.mask_decoder = sam.mask_decoder
        self.num_point_embeddings = pe.num_point_embeddings
        self.not_a_point = nn.Parameter(pe.not_a_point_embed.weight.detach().clone())
        self.point_types = nn.Parameter(
            torch.cat([pe._type_embed(i) for i in range(pe.num_point_embeddings)]).detach()
        )
        with torch.no_grad():
            self.register_buffer("dense_pe", pe.get_dense_pe().detach().clone())
        self.register_buffer(
            "no_mask_dense", pe.no_mask_embed.weight.detach().reshape(1, -1, 1, 1).clone()
        )

    def _embed_points(self, coords: Tensor, labels: Tensor) -> Tensor:
        coords = coords + 0.5
        pad_coords = torch.zeros((1, 1, 2), dtype=coords.dtype)
        pad_labels = torch.full((1, 1), LABEL_PAD, dtype=labels.dtype)
        coords = torch.cat([coords, pad_coords], dim=1)
        labels = torch.cat([labels, pad_labels], dim=1)
        # Square input, so normalising both axes by S matches forward_with_coords.
        emb = self.pe_layer._pe_encoding(coords / float(self.image_size))
        lab = labels.unsqueeze(-1)
        emb = torch.where(lab == LABEL_PAD, self.not_a_point, emb)
        for label in range(self.num_point_embeddings):
            emb = torch.where(lab == label, emb + self.point_types[label], emb)
        return emb

    def forward(
        self,
        image_embed: Tensor,
        high_res_feat_s0: Tensor,
        high_res_feat_s1: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        mask_input: Tensor,
        has_mask_input: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        sparse = self._embed_points(point_coords, point_labels)
        has = has_mask_input.reshape(1, 1, 1, 1)
        dense = self.mask_downscaling(mask_input) * has + self.no_mask_dense * (1.0 - has)
        masks, iou, tokens, obj = self.mask_decoder.predict_masks(
            image_embed, self.dense_pe, sparse, dense, [high_res_feat_s0, high_res_feat_s1]
        )
        return masks, iou, obj, tokens


def _step(x: Tensor) -> Tensor:
    """``(x > 0)`` as 0.0 / 1.0, written as Where (a bool->float Cast is not retyped by
    the fp16 converter)."""
    return torch.where(x > 0, 1.0, 0.0)


class MemoryEncoderGraph(nn.Module):
    """One memory-bank entry, as ``SAM2.encode_memory`` computes it, for one input size.

    ``mask_logits`` are the high-res ``[1, 1, S, S]`` logits of the chosen mask.
    ``binarize`` (1 for frames whose mask came from clicks, 0 for tracked frames) selects
    ``logits > 0`` instead of ``sigmoid(logits)``; either is then mapped to
    ``20 * m - 10``. The learned ``no_obj_embed_spatial`` is added when
    ``object_score_logits <= 0``. ``maskmem_pos_enc`` is a constant of ``S``.
    """

    no_obj: Tensor

    def __init__(self, sam: TorchSam, image_size: int) -> None:
        super().__init__()
        mem = sam.memory()
        self.image_size = image_size
        self.memory_encoder = mem.memory_encoder
        self.scale = sam.cfg.sigmoid_scale_for_mem_enc
        self.bias = sam.cfg.sigmoid_bias_for_mem_enc
        self.add_no_obj = sam.cfg.no_obj_embed_spatial
        self.register_buffer("no_obj", mem.no_obj_embed_spatial.reshape(1, -1, 1, 1).clone())

    def forward(
        self,
        vision_feat: Tensor,
        mask_logits: Tensor,
        object_score_logits: Tensor,
        binarize: Tensor,
    ) -> tuple[Tensor, Tensor]:
        b = binarize.reshape(1, 1, 1, 1)
        hard = _step(mask_logits)
        mask = b * hard + (1.0 - b) * torch.sigmoid(mask_logits)
        mask = mask * self.scale + self.bias
        out = self.memory_encoder(vision_feat, mask, skip_mask_sigmoid=True)
        features = out["vision_features"]
        pos = out["vision_pos_enc"]
        assert isinstance(features, Tensor) and isinstance(pos, list)
        if self.add_no_obj:
            is_obj = _step(object_score_logits).reshape(1, 1, 1, 1)
            features = features + (1.0 - is_obj) * self.no_obj
        return features, pos[-1]


def _separable_grid(table_y: Tensor, table_x: Tensor, x_first: bool = False) -> Tensor:
    """``[H, Cy]`` (per row) and ``[W, Cx]`` (per column) -> ``[H * W, Cy + Cx]`` row-major,
    channels ``[y | x]`` (``[x | y]`` with ``x_first``).

    Pure data movement (Expand / Concat / Reshape): the stored tables stay small, the
    fp16 conversion only rounds stored values, and onnxruntime folds the full grid once
    at session creation.
    """
    h, cy = table_y.shape
    w, cx = table_x.shape
    ys = table_y[:, None, :].expand(h, w, cy)
    xs = table_x[None, :, :].expand(h, w, cx)
    parts = [xs, ys] if x_first else [ys, xs]
    return torch.cat(parts, dim=-1).reshape(h * w, cy + cx)


class _RealRoPEAttention(nn.Module):
    """``RoPEAttention`` for export: real-arithmetic RoPE and query-chunked attention.

    Rotating an adjacent channel pair ``(x0, x1)`` by ``(cos, sin)`` is the reference's
    complex product written out: ``(x0 cos - x1 sin, x0 sin + x1 cos)``. With
    ``rope_k_repeat`` the keys before the trailing ``num_k_exclude_rope`` pointer tokens
    are whole frames of the query grid; they are viewed as ``[.., frames, HW, D]`` so one
    table broadcasts over any number of memory frames. The ``cos`` / ``sin [HW, D/2]``
    tables are passed in (one copy per graph, see ``rope_tables``).
    """

    def __init__(self, attn: RoPEAttention, grid: int, max_keys: int, budget: int) -> None:
        super().__init__()
        self.q_proj, self.k_proj = attn.q_proj, attn.k_proj
        self.v_proj, self.out_proj = attn.v_proj, attn.out_proj
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.rope_k_repeat = attn.rope_k_repeat
        self.rope_theta = attn.rope_theta
        self.hw = grid * grid
        # Largest power-of-two split of the queries whose score matrix fits the budget.
        self.q_chunk = self.hw
        while self.q_chunk % 2 == 0 and self.q_chunk * max_keys * 4 > budget:
            self.q_chunk //= 2

    def _heads(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)

    @staticmethod
    def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Rotate the channel pairs of ``x [..., HW, D]`` by ``cos`` / ``sin [HW, D/2]``."""
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        x0, x1 = pairs[..., 0], pairs[..., 1]
        out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
        return out.reshape(x.shape)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        num_k_exclude_rope: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        q = self._heads(self.q_proj(q))
        k = self._heads(self.k_proj(k))
        v = self._heads(self.v_proj(v))
        q = self._rotate(q, cos, sin)
        if self.rope_k_repeat:
            end = k.shape[2] - num_k_exclude_rope.reshape(())
            b, h = k.shape[0], k.shape[1]
            k_rope = k[:, :, :end].reshape(b, h, -1, self.hw, self.head_dim)
            k_rope = self._rotate(k_rope, cos, sin).reshape(b, h, -1, self.head_dim)
            k = torch.cat([k_rope, k[:, :, end:]], dim=2)
        else:
            k = self._rotate(k, cos, sin)

        q = q * (1.0 / math.sqrt(self.head_dim))
        kt = k.transpose(-2, -1)
        n = q.shape[2]
        chunks = [
            torch.softmax(q[:, :, i : i + self.q_chunk] @ kt, dim=-1) @ v
            for i in range(0, n, self.q_chunk)
        ]
        out = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)
        b, h, n, d = out.shape
        result: Tensor = self.out_proj(out.transpose(1, 2).reshape(b, n, h * d))
        return result


class _ExportMemoryLayer(nn.Module):
    """``MemoryAttentionLayer`` (SAM 2.1 settings) on ``_RealRoPEAttention``."""

    def __init__(self, layer: MemoryAttentionLayer, grid: int, max_keys: int, budget: int) -> None:
        super().__init__()
        if layer.pos_enc_at_attn or layer.pos_enc_at_cross_attn_queries:
            raise NotImplementedError("query position encodings are not used by SAM 2.1")
        self.self_attn = _RealRoPEAttention(layer.self_attn, grid, grid * grid, budget)
        self.cross_attn = _RealRoPEAttention(layer.cross_attn_image, grid, max_keys, budget)
        self.linear1, self.linear2 = layer.linear1, layer.linear2
        self.norm1, self.norm2, self.norm3 = layer.norm1, layer.norm2, layer.norm3
        self.activation = layer.activation
        self.pos_at_keys = layer.pos_enc_at_cross_attn_keys

    def forward(
        self, tgt: Tensor, memory: Tensor, pos: Tensor, num_ptr: Tensor, cos: Tensor, sin: Tensor
    ) -> Tensor:
        x = self.norm1(tgt)
        tgt = tgt + self.self_attn(x, x, x, num_ptr, cos, sin)
        x = self.norm2(tgt)
        keys = memory + pos if self.pos_at_keys else memory
        tgt = tgt + self.cross_attn(x, keys, memory, num_ptr, cos, sin)
        x = self.norm3(tgt)
        mlp_out: Tensor = self.linear2(self.activation(self.linear1(x)))
        return tgt + mlp_out


class MemoryAttentionGraph(nn.Module):
    """``MemoryAttention`` for one input size, batch-first, bank length dynamic.

    ``curr [1, HW, 256]`` are the raw stride-16 tokens (no ``no_mem_embed``); ``memory`` /
    ``memory_pos [1, N, 64]`` hold the spatial memories (whole frames of ``HW`` tokens)
    followed by ``num_obj_ptr_tokens`` pointer tokens. The input position term
    ``0.1 * curr_pos`` is built in the graph from separable tables. Outputs the fused
    tokens and the same values transposed to ``[1, 256, S/16, S/16]`` for the SAM head.
    """

    pos_y: Tensor
    pos_x: Tensor
    cos_y: Tensor
    cos_x: Tensor
    sin_y: Tensor
    sin_x: Tensor

    def __init__(
        self, sam: TorchSam, image_size: int, budget: int = ATTENTION_SCORE_BUDGET
    ) -> None:
        super().__init__()
        cfg = sam.cfg
        attn = sam.memory().memory_attention
        self.grid = image_size // cfg.backbone_stride
        self.image_size = image_size
        tokens_per_ptr = cfg.hidden_dim // cfg.mem_dim
        max_keys = cfg.num_maskmem * self.grid**2 + cfg.max_obj_ptrs_in_encoder * tokens_per_ptr
        layers = [cast(MemoryAttentionLayer, layer) for layer in attn.layers]
        self.layers = nn.ModuleList(
            _ExportMemoryLayer(layer, self.grid, max_keys, budget) for layer in layers
        )
        self.norm = attn.norm
        self.pos_weight = 0.1 if attn.pos_enc_at_input else 0.0
        # The sine encoding is [y-part | x-part] over channels: keep one column / row.
        pos = sam.vision_pos_tokens(image_size).reshape(self.grid, self.grid, -1)
        half = pos.shape[-1] // 2
        self.register_buffer("pos_y", (self.pos_weight * pos[:, 0, :half]).clone())
        self.register_buffer("pos_x", (self.pos_weight * pos[0, :, half:]).clone())
        # Every attention rotates with the same table (one grid, head_dim and theta).
        export_layers = [cast(_ExportMemoryLayer, layer) for layer in self.layers]
        attns = [a for layer in export_layers for a in (layer.self_attn, layer.cross_attn)]
        spec = {(a.head_dim, float(a.rope_theta)) for a in attns}
        if len(spec) != 1:
            raise ValueError(f"attentions disagree on RoPE settings: {spec}")
        head_dim, theta = spec.pop()
        cis = axial_rope_cis(head_dim, end_x=self.grid, end_y=self.grid, theta=theta)
        # cis[t], t = y * grid + x: the first D/4 pairs depend on x only, the rest on y.
        quarter = head_dim // 4
        table = cis.reshape(self.grid, self.grid, 2 * quarter, 2)
        for i, name in enumerate(("cos", "sin")):
            self.register_buffer(f"{name}_x", table[0, :, :quarter, i].clone())
            self.register_buffer(f"{name}_y", table[:, 0, quarter:, i].clone())

    def forward(
        self, curr: Tensor, memory: Tensor, memory_pos: Tensor, num_obj_ptr_tokens: Tensor
    ) -> tuple[Tensor, Tensor]:
        output = curr
        if self.pos_weight:
            output = output + _separable_grid(self.pos_y, self.pos_x)[None]
        cos = _separable_grid(self.cos_y, self.cos_x, x_first=True)
        sin = _separable_grid(self.sin_y, self.sin_x, x_first=True)
        for layer in self.layers:
            output = layer(output, memory, memory_pos, num_obj_ptr_tokens, cos, sin)
        fused: Tensor = self.norm(output)
        embed = fused.transpose(1, 2).reshape(1, -1, self.grid, self.grid)
        return fused, embed


class ObjectPointerGraph(nn.Module):
    """``obj_ptr_proj`` on the chosen mask token, then the hard occlusion switch
    (``fixed_no_obj_ptr``): ``is_obj * ptr + (1 - is_obj) * no_obj_ptr``."""

    no_obj_ptr: Tensor

    def __init__(self, sam: TorchSam) -> None:
        super().__init__()
        mem = sam.memory()
        self.proj = mem.obj_ptr_proj
        self.register_buffer("no_obj_ptr", mem.no_obj_ptr.clone())

    def forward(self, mask_token: Tensor, object_score_logits: Tensor) -> Tensor:
        is_obj = _step(object_score_logits)
        ptr: Tensor = self.proj(mask_token)
        return is_obj * ptr + (1 - is_obj) * self.no_obj_ptr


class PointerTposGraph(nn.Module):
    """Temporal encoding of object pointers: ``obj_ptr_tpos_proj(sine(t_diff / t_max))``.

    ``t_diff [K]`` are the signed frame distances (as float), ``t_diff_max [1]`` is
    ``min(num_frames, max_obj_ptrs_in_encoder) - 1``. Output ``[K, 64]``: row ``i`` is the
    position of all four 64-channel tokens of pointer ``i``.
    """

    def __init__(self, sam: TorchSam) -> None:
        super().__init__()
        self.proj = sam.memory().obj_ptr_tpos_proj
        self.dim = sam.cfg.hidden_dim

    def forward(self, t_diff: Tensor, t_diff_max: Tensor) -> Tensor:
        out: Tensor = self.proj(sine_pe_1d(t_diff / t_diff_max, dim=self.dim))
        return out


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------


def file_stem(kind: str, model: str, size: int) -> str:
    """``image_encoder_1024`` for tiny, ``image_encoder_small_1024`` otherwise."""
    return f"{kind}_{size}" if model == "tiny" else f"{kind}_{model}_{size}"


def head_dummy_inputs(sam: TorchSam, size: int, num_points: int = 3) -> tuple[Tensor, ...]:
    grid = size // sam.cfg.backbone_stride
    gen = torch.Generator().manual_seed(0)
    return (
        torch.randn(1, 256, grid, grid, generator=gen),
        torch.randn(1, 32, 4 * grid, 4 * grid, generator=gen),
        torch.randn(1, 64, 2 * grid, 2 * grid, generator=gen),
        torch.rand(1, num_points, 2, generator=gen) * size,
        torch.tensor([[1, 0, 2, 3, -1][:num_points]], dtype=torch.int32),
        torch.zeros(1, 1, 4 * grid, 4 * grid),
        torch.zeros(1),
    )


def simplify(
    model: onnx.ModelProto, fold_limit: int | None = None, max_passes: int = 10
) -> onnx.ModelProto:
    """Fold the traced shape arithmetic away with onnxslim, repeated to a fixed point.

    The legacy exporter records every ``x.shape`` use (window padding, head splits,
    attention scale) as Shape/Gather/Div chains; one slim pass only folds what shape
    inference could resolve so far, so it is repeated until the node count settles.
    Gemm fusion is skipped so the encoder's Linear layers stay MatMul nodes, which is
    what onnxruntime's dynamic quantiser targets. Constants that would fold to more than
    ``fold_limit`` bytes are left as subgraphs.
    """
    import onnxslim

    count = -1
    for _ in range(max_passes):
        model = onnxslim.slim(model, skip_fusion_patterns=["FusionGemm"], size_threshold=fold_limit)
        if len(model.graph.node) == count:
            break
        count = len(model.graph.node)
    return model


def _set_output_shapes(model: onnx.ModelProto, shapes: dict[str, list[int]]) -> None:
    """Pin output dims that shape inference leaves symbolic (slices of a dynamic axis)."""
    for out in model.graph.output:
        if out.name in shapes:
            dims = out.type.tensor_type.shape.dim
            for dim, value in zip(dims, shapes[out.name], strict=True):
                dim.Clear()
                dim.dim_value = value


def _torch_export(
    module: nn.Module,
    args: tuple[Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None,
    output_shapes: dict[str, list[int]],
    metadata: dict[str, str],
    fold_limit: int | None = None,
    torch_fold: bool = True,
) -> None:
    """Trace ``module`` to ONNX, simplify, pin output shapes, stamp metadata, save.

    ``torch_fold=False`` skips the exporter's own constant folding, which ignores size:
    the memory attention builds its RoPE / position grids from small separable tables,
    and only onnxslim (bounded by ``fold_limit``) may fold them. Weight transposes the
    exporter would have folded are then folded by ``_fold_weight_transposes``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        # Shape arithmetic on static sizes is traced as constants; that is intended.
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        torch.onnx.export(
            module,
            args,
            buffer,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            do_constant_folding=torch_fold,
            dynamo=False,
        )
    model = simplify(onnx.load_from_string(buffer.getvalue()), fold_limit)
    if not torch_fold:
        _fold_weight_transposes(model)
        model = simplify(model, fold_limit)
    _set_output_shapes(model, output_shapes)
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def _fold_weight_transposes(model: onnx.ModelProto) -> None:
    """Replace ``Transpose(initializer)`` nodes by the transposed initializer."""
    from onnx import numpy_helper

    graph = model.graph
    inits = {init.name: init for init in graph.initializer}
    uses = Counter(name for node in graph.node for name in node.input)
    keep: list[onnx.NodeProto] = []
    for node in graph.node:
        src = node.input[0] if node.input else ""
        if node.op_type != "Transpose" or src not in inits:
            keep.append(node)
            continue
        perm = [a.ints for a in node.attribute if a.name == "perm"]
        array = numpy_helper.to_array(inits[src])
        folded = array.transpose(*perm[0]) if perm else array.T
        graph.initializer.append(numpy_helper.from_array(folded.copy(), node.output[0]))
        if uses[src] == 1:
            graph.initializer.remove(inits[src])
    del graph.node[:]
    graph.node.extend(keep)


def _metadata(sam: TorchSam, size: int, graph: str) -> dict[str, str]:
    return {"graph": graph, "model": f"sam2.1_hiera_{sam.name}", "image_size": str(size)}


@torch.no_grad()
def export_image_encoder(sam: TorchSam, size: int, path: Path) -> Path:
    graph = ImageEncoderGraph(sam, size).eval()
    dummy = (torch.randn(1, 3, size, size, generator=torch.Generator().manual_seed(0)),)
    g = size // sam.cfg.backbone_stride
    shapes = {
        "image_embed": [1, sam.cfg.hidden_dim, g, g],
        "high_res_feat_s0": [1, sam.cfg.hidden_dim // 8, 4 * g, 4 * g],
        "high_res_feat_s1": [1, sam.cfg.hidden_dim // 4, 2 * g, 2 * g],
    }
    _torch_export(
        graph,
        dummy,
        path,
        ENCODER_INPUTS,
        ENCODER_OUTPUTS,
        None,
        shapes,
        _metadata(sam, size, "image_encoder"),
        fold_limit=FOLD_LIMIT_BYTES,
    )
    return path


@torch.no_grad()
def export_sam_head(sam: TorchSam, size: int, path: Path) -> Path:
    graph = SamHeadGraph(sam, size).eval()
    dynamic = {"point_coords": {1: "num_points"}, "point_labels": {1: "num_points"}}
    g = size // sam.cfg.backbone_stride
    m = sam.mask_decoder.num_mask_tokens
    shapes = {
        "masks": [1, m, 4 * g, 4 * g],
        "iou_predictions": [1, m],
        "object_score_logits": [1, 1],
        "sam_tokens": [1, m, sam.cfg.hidden_dim],
    }
    _torch_export(
        graph,
        head_dummy_inputs(sam, size),
        path,
        HEAD_INPUTS,
        HEAD_OUTPUTS,
        dynamic,
        shapes,
        _metadata(sam, size, "sam_head"),
    )
    return path


@torch.no_grad()
def export_memory_encoder(sam: TorchSam, size: int, path: Path) -> Path:
    graph = MemoryEncoderGraph(sam, size).eval()
    g = size // sam.cfg.backbone_stride
    gen = torch.Generator().manual_seed(0)
    dummy = (
        torch.randn(1, sam.cfg.hidden_dim, g, g, generator=gen),
        torch.randn(1, 1, size, size, generator=gen) * 8,
        torch.ones(1, 1),
        torch.zeros(1),
    )
    m = sam.cfg.mem_dim
    shapes = {"maskmem_features": [1, m, g, g], "maskmem_pos_enc": [1, m, g, g]}
    _torch_export(
        graph,
        dummy,
        path,
        MEMORY_ENCODER_INPUTS,
        MEMORY_ENCODER_OUTPUTS,
        None,
        shapes,
        _metadata(sam, size, "memory_encoder"),
    )
    return path


@torch.no_grad()
def export_memory_attention(sam: TorchSam, size: int, path: Path) -> Path:
    graph = MemoryAttentionGraph(sam, size).eval()
    g = size // sam.cfg.backbone_stride
    hw, c, m = g * g, sam.cfg.hidden_dim, sam.cfg.mem_dim
    gen = torch.Generator().manual_seed(0)
    n_ptr = 8  # two pointers of four tokens behind two spatial frames
    dummy = (
        torch.randn(1, hw, c, generator=gen),
        torch.randn(1, 2 * hw + n_ptr, m, generator=gen),
        torch.randn(1, 2 * hw + n_ptr, m, generator=gen),
        torch.tensor([n_ptr], dtype=torch.int64),
    )
    dynamic = {"memory": {1: "num_memory"}, "memory_pos": {1: "num_memory"}}
    shapes = {"fused": [1, hw, c], "image_embed": [1, c, g, g]}
    _torch_export(
        graph,
        dummy,
        path,
        MEMORY_ATTENTION_INPUTS,
        MEMORY_ATTENTION_OUTPUTS,
        dynamic,
        shapes,
        _metadata(sam, size, "memory_attention"),
        fold_limit=GRID_FOLD_LIMIT_BYTES,
        torch_fold=False,
    )
    return path


@torch.no_grad()
def export_object_pointer(sam: TorchSam, path: Path) -> Path:
    graph = ObjectPointerGraph(sam).eval()
    c = sam.cfg.hidden_dim
    dummy = (torch.randn(1, c, generator=torch.Generator().manual_seed(0)), torch.ones(1, 1))
    meta = {"graph": "object_pointer", "model": f"sam2.1_hiera_{sam.name}"}
    _torch_export(
        graph,
        dummy,
        path,
        OBJECT_POINTER_INPUTS,
        OBJECT_POINTER_OUTPUTS,
        None,
        {"obj_ptr": [1, c]},
        meta,
    )
    return path


@torch.no_grad()
def export_pointer_tpos(sam: TorchSam, path: Path) -> Path:
    graph = PointerTposGraph(sam).eval()
    dummy = (torch.arange(4, dtype=torch.float32), torch.tensor([15.0]))
    meta = {"graph": "object_pointer_tpos", "model": f"sam2.1_hiera_{sam.name}"}
    _torch_export(
        graph,
        dummy,
        path,
        POINTER_TPOS_INPUTS,
        POINTER_TPOS_OUTPUTS,
        {"t_diff": {0: "num_ptrs"}, "ptr_pos": {0: "num_ptrs"}},
        {},
        meta,
    )
    return path


def _floats(t: Tensor) -> Any:
    """Exact JSON form of a float32 tensor (each value round-trips through float64)."""
    return t.detach().float().tolist()


def video_constants(sam: TorchSam, size: int) -> dict[str, Any]:
    """Everything the video loop needs besides the graphs, for input size ``size``."""
    cfg = sam.cfg
    mem = sam.memory()
    md = sam.mask_decoder
    g = size // cfg.backbone_stride
    return {
        "model": f"sam2.1_hiera_{sam.name}",
        "image_size": size,
        "embed_grid": g,
        "mask_size": 4 * g,
        "hidden_dim": cfg.hidden_dim,
        "mem_dim": cfg.mem_dim,
        "tokens_per_obj_ptr": cfg.hidden_dim // cfg.mem_dim,
        "num_maskmem": cfg.num_maskmem,
        "max_obj_ptrs_in_encoder": cfg.max_obj_ptrs_in_encoder,
        "max_cond_frames_in_attn": cfg.max_cond_frames_in_attn,
        "memory_temporal_stride_for_eval": cfg.memory_temporal_stride_for_eval,
        "only_obj_ptrs_in_the_past_for_eval": cfg.only_obj_ptrs_in_the_past_for_eval,
        "pixel_mean": list(cfg.pixel_mean),
        "pixel_std": list(cfg.pixel_std),
        "multimask_min_pt_num": cfg.multimask_min_pt_num,
        "multimask_max_pt_num": cfg.multimask_max_pt_num,
        "stability_delta": md.dynamic_multimask_stability_delta,
        "stability_thresh": md.dynamic_multimask_stability_thresh,
        "no_obj_score": NO_OBJ_SCORE,
        "mask_input_clamp": 32.0,
        "sigmoid_scale_for_mem_enc": cfg.sigmoid_scale_for_mem_enc,
        "sigmoid_bias_for_mem_enc": cfg.sigmoid_bias_for_mem_enc,
        # [256]; image_embed = raw + no_mem_embed (the encoder graph already adds it).
        "no_mem_embed": _floats(sam.no_mem_embed.reshape(-1)),
        # [num_maskmem][64]; spatial memory with temporal slot t_pos uses row
        # num_maskmem - t_pos - 1 (conditioning frames: t_pos 0 -> last row).
        "maskmem_tpos_enc": _floats(mem.maskmem_tpos_enc.reshape(cfg.num_maskmem, -1)),
        # Folded into the graphs; listed so the loop can be audited without the checkpoint.
        "no_obj_ptr": _floats(mem.no_obj_ptr.reshape(-1)),
        "no_obj_embed_spatial": _floats(mem.no_obj_embed_spatial.reshape(-1)),
    }


def write_constants(sam: TorchSam, size: int, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(video_constants(sam, size)) + "\n")
    return path


def encoder_block_matmuls(path: Path, blocks: Sequence[int]) -> list[str]:
    """Names of the MatMul nodes inside the listed Hiera blocks of an encoder graph."""
    model = onnx.load(str(path), load_external_data=False)
    tags = tuple(f"/trunk/blocks.{i}/" for i in blocks)
    return [
        n.name for n in model.graph.node if n.op_type == "MatMul" and any(t in n.name for t in tags)
    ]


def quantize_encoder_int8(src: Path, dst: Path, scheme: str = "weight-only") -> Path:
    """The shipped int8 encoder (see exports/README.md for the measured trade-offs).

    ``weight-only`` (default): MatMul weights int8 with block-wise scales (64 rows),
    DequantizeLinear in the graph, compute fp32; Hiera blocks 0-1 and every Conv stay
    fp32. Plain ONNX ops only, so it runs unchanged on ORT Web's WebGPU provider.

    ``dynamic``: onnxruntime ``quantize_dynamic``, per-channel int8 MatMul weights with
    uint8 activations quantised at run time (MatMulInteger); Hiera blocks 0-3 stay fp32.
    MatMulInteger / DynamicQuantizeLinear have no WebGPU kernels.
    """
    if scheme == "weight-only":
        keep = encoder_block_matmuls(src, INT8_FP32_BLOCKS["weight-only"])
        return quantize_weights_int8(src, dst, ("MatMul",), 64, keep)
    if scheme == "dynamic":
        keep = encoder_block_matmuls(src, INT8_FP32_BLOCKS["dynamic"])
        return quantize_dynamic_int8(src, dst, per_channel=True, nodes_to_exclude=keep)
    raise ValueError(f"unknown int8 scheme {scheme!r}")


def quantize_dynamic_int8(
    src: Path,
    dst: Path,
    op_types: Sequence[str] = ("MatMul",),
    per_channel: bool = False,
    weight_type: str = "QInt8",
    nodes_to_exclude: Sequence[str] = (),
) -> Path:
    """Dynamic (weight int8, activation uint8 at run time) quantisation with onnxruntime.

    Only MatMuls whose B input is a constant are touched (attention score / value
    products between two activations stay fp32). Conv is left out by default because
    ``ConvInteger`` exists only for uint8 weights in onnxruntime; ``weight_type="QUInt8"``
    makes it quantisable.
    """
    import logging

    from onnxruntime.quantization import QuantType, quantize_dynamic

    logging.getLogger().setLevel(logging.ERROR)  # the "run pre-processing" advice is noise
    quantize_dynamic(
        str(src),
        str(dst),
        op_types_to_quantize=list(op_types),
        per_channel=per_channel,
        weight_type=QuantType[weight_type],
        nodes_to_exclude=list(nodes_to_exclude),
        extra_options={"MatMulConstBOnly": True},
    )
    return dst


def quantize_weights_int8(
    src: Path,
    dst: Path,
    op_types: Sequence[str] = ("MatMul", "Conv"),
    block_size: int | None = 64,
    nodes_to_exclude: Sequence[str] = (),
    min_elements: int = 4096,
) -> Path:
    """Weight-only int8: constant weights stored as int8 plus fp32 scales.

    Each quantised weight is rebuilt in the graph by ``DequantizeLinear`` and feeds the
    original fp32 MatMul / Conv, so only storage is int8; activations and compute stay
    fp32 and every op is a plain ONNX op (no MatMulInteger / DynamicQuantizeLinear).
    Quantisation is symmetric (zero point 0, scale ``max|w| / 127``):

    * MatMul ``B [K, N]`` with ``block_size``: one scale per output column per block of
      ``block_size`` input rows. Stored as ``[block_size, K / block_size * N]`` so a
      per-axis DequantizeLinear (opset 13) applies the scales, followed by a constant
      Reshape / Transpose / Reshape back to ``[K, N]``.
    * MatMul without ``block_size``, and Conv: one scale per output channel.

    Weights of nodes in ``nodes_to_exclude`` and weights smaller than ``min_elements``
    stay fp32.
    """
    import numpy as np
    from onnx import helper, numpy_helper

    model = onnx.load(str(src))
    graph = model.graph
    inits = {init.name: init for init in graph.initializer}
    new_nodes: list[onnx.NodeProto] = []
    new_inits: list[onnx.TensorProto] = []
    done: set[str] = set()

    def add_dq(name: str, q: np.ndarray, scale: np.ndarray, axis: int, out: str) -> None:
        new_inits.extend(
            [
                numpy_helper.from_array(q, f"{name}_int8"),
                numpy_helper.from_array(scale.astype(np.float32), f"{name}_scale"),
                numpy_helper.from_array(np.zeros(scale.shape, np.int8), f"{name}_zero"),
            ]
        )
        new_nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [f"{name}_int8", f"{name}_scale", f"{name}_zero"],
                [out],
                name=f"{name}_dequantize",
                axis=axis,
            )
        )

    def reshape(name: str, src_name: str, shape: list[int], out: str) -> None:
        new_inits.append(numpy_helper.from_array(np.array(shape, np.int64), f"{out}_shape"))
        new_nodes.append(helper.make_node("Reshape", [src_name, f"{out}_shape"], [out]))

    for node in graph.node:
        if node.op_type not in op_types or node.name in nodes_to_exclude:
            continue
        name = node.input[1]
        if name in done or name not in inits:
            continue
        w = numpy_helper.to_array(inits[name]).astype(np.float32)
        if w.size < min_elements or w.ndim != (2 if node.op_type == "MatMul" else 4):
            continue
        graph.initializer.remove(inits[name])
        done.add(name)
        if node.op_type == "MatMul" and block_size and w.shape[0] % block_size == 0:
            k, n = w.shape
            blocks = k // block_size
            wr = w.reshape(blocks, block_size, n)
            amax = np.abs(wr).max(axis=1)  # [blocks, n]
            scale = np.where(amax > 0, amax / 127.0, 1.0)
            q = np.clip(np.rint(wr / scale[:, None, :]), -127, 127).astype(np.int8)
            stored = np.ascontiguousarray(q.transpose(1, 0, 2).reshape(block_size, blocks * n))
            add_dq(name, stored, scale.reshape(-1), 1, f"{name}_dq")
            reshape(name, f"{name}_dq", [block_size, blocks, n], f"{name}_dq3")
            new_nodes.append(
                helper.make_node("Transpose", [f"{name}_dq3"], [f"{name}_t"], perm=[1, 0, 2])
            )
            reshape(name, f"{name}_t", [k, n], name)
        else:
            axis = 1 if node.op_type == "MatMul" else 0
            reduce = tuple(i for i in range(w.ndim) if i != axis)
            amax = np.abs(w).max(axis=reduce)
            scale = np.where(amax > 0, amax / 127.0, 1.0)
            shape = [1] * w.ndim
            shape[axis] = -1
            q = np.clip(np.rint(w / scale.reshape(shape)), -127, 127).astype(np.int8)
            add_dq(name, q, scale, axis, name)

    graph.initializer.extend(new_inits)
    nodes = new_nodes + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(dst))
    return dst


def _isolate_outputs(model: onnx.ModelProto) -> None:
    """Give every graph output its own Identity so no output is also an internal edge.

    The fp16 converter inserts a Cast in front of each fp32 output; an output that also
    feeds other nodes (``sam_tokens`` feeds the hypernetwork MLPs) would otherwise be
    read by them as fp32 and fail type checking.
    """
    for out in model.graph.output:
        inner = f"{out.name}__inner"
        for node in model.graph.node:
            node.output[:] = [inner if o == out.name else o for o in node.output]
            node.input[:] = [inner if i == out.name else i for i in node.input]
        model.graph.node.append(onnx.helper.make_node("Identity", [inner], [out.name]))


def _retype_float_casts(model: onnx.ModelProto, names: set[str]) -> None:
    """Point the graph's own ``Cast(to=float)`` nodes (``names``) at float16.

    The converter retypes tensors but not the ``to`` attribute of Casts already in the
    graph (e.g. ``.to(x.dtype)`` on the memory encoder's position encoding); the casts it
    inserts itself (I/O boundary, blocked ops such as Resize) are left alone.
    """
    for node in model.graph.node:
        if node.op_type != "Cast" or node.name not in names:
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == onnx.TensorProto.FLOAT:
                attr.i = onnx.TensorProto.FLOAT16


def convert_fp16(src: Path, dst: Path) -> Path:
    """fp16 weights and compute with fp32 inputs/outputs (casts at the graph boundary)."""
    from onnxconverter_common import float16

    model = onnx.load(str(src))
    _isolate_outputs(model)
    own_casts = {n.name for n in model.graph.node if n.op_type == "Cast"}
    del model.graph.value_info[:]  # stale fp32 annotations; re-inferred below
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # tiny weights flushed to 1e-7
        half = float16.convert_float_to_float16(model, keep_io_types=True)
    _retype_float_casts(half, own_casts)
    half = onnx.shape_inference.infer_shapes(half, strict_mode=True)
    onnx.checker.check_model(half, full_check=True)
    onnx.save(half, str(dst))
    return dst


convert_head_fp16 = convert_fp16


def op_counts(path: Path) -> Counter[str]:
    model = onnx.load(str(path), load_external_data=False)
    return Counter(node.op_type for node in model.graph.node)


def describe(path: Path) -> str:
    counts = op_counts(path)
    flagged = sorted(op for op in counts if op in WEBGPU_UNSUPPORTED)
    size_mb = path.stat().st_size / 1e6
    ops = ", ".join(f"{op}x{n}" for op, n in sorted(counts.items()))
    note = f"  not on WebGPU: {', '.join(flagged)}" if flagged else ""
    return f"{path.name}: {size_mb:.1f} MB, {sum(counts.values())} nodes [{ops}]{note}"


def export_all(
    model: str,
    out_dir: Path,
    sizes: Sequence[int] = SIZES,
    precisions: Sequence[str] = PRECISIONS,
    sam: TorchSam | None = None,
    int8_scheme: str = "weight-only",
    graphs: Sequence[str] = GRAPH_SETS,
) -> list[Path]:
    """Export every requested graph; returns the written paths."""
    sam = sam or load_torch(model)
    written: list[Path] = []
    if "image" in graphs:
        written += _export_image_graphs(sam, model, out_dir, sizes, precisions, int8_scheme)
    if "memory" in graphs:
        written += export_memory_graphs(sam, model, out_dir, sizes, precisions)
    return written


def export_memory_graphs(
    sam: TorchSam,
    model: str,
    out_dir: Path,
    sizes: Sequence[int] = SIZES,
    precisions: Sequence[str] = PRECISIONS,
) -> list[Path]:
    """Memory encoder / attention per size (fp32 + fp16), pointer graphs, constants."""
    suffix = "" if model == "tiny" else f"_{model}"
    ptr = export_object_pointer(sam, out_dir / f"object_pointer{suffix}.onnx")
    tpos = export_pointer_tpos(sam, out_dir / f"object_pointer_tpos{suffix}.onnx")
    written = [ptr, tpos]
    for size in sizes:
        t0 = time.perf_counter()
        menc = out_dir / f"{file_stem('memory_encoder', model, size)}.onnx"
        mattn = out_dir / f"{file_stem('memory_attention', model, size)}.onnx"
        export_memory_encoder(sam, size, menc)
        export_memory_attention(sam, size, mattn)
        written += [menc, mattn]
        print(f"[{model} {size}] exported memory graphs in {time.perf_counter() - t0:.1f} s")
        if "fp16" in precisions:
            written += [convert_fp16(p, p.with_suffix(".fp16.onnx")) for p in (menc, mattn)]
        written.append(
            write_constants(sam, size, out_dir / f"{file_stem('constants', model, size)}.json")
        )
    return written


def _export_image_graphs(
    sam: TorchSam,
    model: str,
    out_dir: Path,
    sizes: Sequence[int],
    precisions: Sequence[str],
    int8_scheme: str,
) -> list[Path]:
    written: list[Path] = []
    for size in sizes:
        enc = out_dir / f"{file_stem('image_encoder', model, size)}.onnx"
        head = out_dir / f"{file_stem('sam_head', model, size)}.onnx"
        t0 = time.perf_counter()
        export_image_encoder(sam, size, enc)
        export_sam_head(sam, size, head)
        written += [enc, head]
        print(f"[{model} {size}] exported fp32 graphs in {time.perf_counter() - t0:.1f} s")
        if "int8" in precisions:
            int8 = enc.with_suffix(".int8.onnx")
            written.append(quantize_encoder_int8(enc, int8, int8_scheme))
        if "fp16" in precisions:
            written.append(convert_head_fp16(head, head.with_suffix(".fp16.onnx")))
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=sorted(CONFIGS), default="tiny")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--precisions", nargs="+", choices=PRECISIONS, default=list(PRECISIONS))
    parser.add_argument("--int8-scheme", choices=INT8_SCHEMES, default="weight-only")
    parser.add_argument("--graphs", nargs="+", choices=GRAPH_SETS, default=list(GRAPH_SETS))
    parser.add_argument("--out", type=Path, default=EXPORTS)
    args = parser.parse_args(argv)
    paths = export_all(
        args.model,
        args.out,
        args.sizes,
        args.precisions,
        int8_scheme=args.int8_scheme,
        graphs=args.graphs,
    )
    for path in paths:
        if path.suffix == ".onnx":
            print(describe(path))
        else:
            print(f"{path.name}: {path.stat().st_size / 1e3:.1f} kB")


if __name__ == "__main__":
    main()
