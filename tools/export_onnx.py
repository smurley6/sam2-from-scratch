"""Export the image half of SAM 2 to ONNX for ONNX Runtime Web.

Two graphs per input resolution ``S`` (1024 and 512):

``image_encoder_{S}.onnx``
    Hiera trunk + FPN neck + the mask decoder's ``conv_s0`` / ``conv_s1`` projections.
    ``image [1, 3, S, S]`` (normalised) -> ``image_embed [1, 256, S/16, S/16]`` (with
    ``no_mem_embed`` added, ready for single-image decoding), ``high_res_feat_s0
    [1, 32, S/4, S/4]``, ``high_res_feat_s1 [1, 64, S/8, S/8]``.

``sam_head_{S}.onnx``
    Prompt encoder + mask decoder, returning all four mask tokens. The dense positional
    encoding of the embedding grid is baked in as a constant. See ``exports/README.md``
    for the input/output contract.

Precision variants: ``image_encoder_{S}.int8.onnx`` (int8 MatMul weights, see
``quantize_encoder_int8`` for the two schemes) and ``sam_head_{S}.fp16.onnx`` (fp16
weights and compute, fp32 I/O).

Export path: the legacy TorchScript exporter (``dynamo=False``, opset 17), then onnxslim
to fold the traced shape arithmetic so every encoder shape is static.

Usage (from the repo root, venv active)::

    python tools/export_onnx.py --model tiny
    python tools/export_onnx.py --model tiny --sizes 512 --precisions fp32
    python tools/export_onnx.py --model tiny --int8-scheme dynamic

Files for a model other than ``tiny`` carry the model name, e.g.
``image_encoder_small_1024.onnx``.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import onnx
import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:  # allow ``python tools/export_onnx.py`` without install
    sys.path.insert(0, str(ROOT / "src"))

from sam2_scratch.configs import CONFIGS, SAM2Config  # noqa: E402
from sam2_scratch.image_encoder import ImageEncoder  # noqa: E402
from sam2_scratch.mask_decoder import MaskDecoder  # noqa: E402
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
) -> None:
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
            do_constant_folding=True,
            dynamo=False,
        )
    model = simplify(onnx.load_from_string(buffer.getvalue()), fold_limit)
    _set_output_shapes(model, output_shapes)
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


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


def convert_head_fp16(src: Path, dst: Path) -> Path:
    """fp16 weights and compute with fp32 inputs/outputs (casts at the graph boundary)."""
    from onnxconverter_common import float16

    model = onnx.load(str(src))
    _isolate_outputs(model)
    del model.graph.value_info[:]  # stale fp32 annotations; re-inferred below
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # tiny weights flushed to 1e-7
        half = float16.convert_float_to_float16(model, keep_io_types=True)
    half = onnx.shape_inference.infer_shapes(half, strict_mode=True)
    onnx.checker.check_model(half, full_check=True)
    onnx.save(half, str(dst))
    return dst


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
) -> list[Path]:
    """Export every requested graph; returns the written paths."""
    sam = sam or load_torch(model)
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
    parser.add_argument("--out", type=Path, default=EXPORTS)
    args = parser.parse_args(argv)
    paths = export_all(
        args.model, args.out, args.sizes, args.precisions, int8_scheme=args.int8_scheme
    )
    for path in paths:
        print(describe(path))


if __name__ == "__main__":
    main()
