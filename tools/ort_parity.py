"""Parity of the exported ONNX graphs (onnxruntime, CPU provider) against the torch modules.

For each input size the three images in ``data/images`` are preprocessed exactly as the
image predictor does, encoded by torch and by every exported encoder variant, and every
prompt of ``reference_dumps/<model>/manifest.json`` is decoded by torch and by every head
variant. Each prompt is decoded twice: once without a mask prompt and once with the torch
model's best low-res mask fed back as ``mask_input`` (the refinement path).

Reported per graph and precision:

* ``max|err|``: max abs error vs torch fp32 over every output of that graph (encoder:
  the three feature maps; head: masks, IoU predictions, object score, SAM tokens; the
  head is fed torch's encoder features so its error is isolated).
* ``IoU``: mask IoU at logit threshold 0 of the best multimask candidate (highest
  torch-predicted IoU among tokens 1..3) against torch, at the low-res S/4 resolution.
  Encoder rows are end to end: that encoder's features decoded by the fp32 ONNX head.
* ``flips``: prompts where the ONNX graph's own best-candidate choice differs from torch.
* ``ms``: median ORT CPU latency of one ``run`` call.

Video rows (``--parts video``):

* ``memory_encoder_{S}`` / ``memory_attention_{S}``: replay of the memory calls at
  frames 1, 2, 7, 20, 47 of the bedroom forward run, recorded by the official package at
  1024 (``video_bedroom_fwd.pt``) and by this repo's torch predictor at 512 (no dump
  exists there). ``max|err|`` is over the graph outputs (memory features before the
  bfloat16 storage rounding, position encoding; fused tokens). Memory-attention IoU is
  the tracking-frame head call (fp32 head, padding point) on the graph's embedding vs on
  the reference embedding.
* ``object_pointer`` / ``object_pointer_tpos``: against the torch modules.
* ``video loop fwd/rev``: ``tools/ort_video_loop.py`` end to end, per-frame low-res mask
  IoU vs the dump (1024) or the torch predictor (512), for the fp32 graphs and the
  shipped precision set; ``ms`` is the median per tracked frame including the encoder.

Usage (from the repo root, venv active)::

    python tools/ort_parity.py --model tiny
    python tools/ort_parity.py --model tiny --sizes 512 --images truck --json parity.json
    python tools/ort_parity.py --parts video --sizes 512
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:  # allow ``python tools/ort_parity.py`` without install
        sys.path.insert(0, str(_p))

from tools.export_onnx import (  # noqa: E402
    EXPORTS,
    SIZES,
    TorchSam,
    file_stem,
    load_torch,
)

from sam2_scratch.common import sine_pe_1d  # noqa: E402
from sam2_scratch.image_predictor import ImagePredictor  # noqa: E402
from sam2_scratch.prompt_encoder import LABEL_BOX_BR, LABEL_BOX_TL  # noqa: E402

IMAGES = ("truck", "cars", "groceries")
DUMPS = ROOT / "reference_dumps"
IMAGE_DIR = ROOT / "data" / "images"

# Acceptance thresholds (docs/PROGRAM.md T6 and the export task).
FP32_MAX_ABS = 1e-3
FP32_IOU = 0.999
FP16_IOU = 0.99
INT8_IOU = 0.98
# End-to-end video loop with the shipped precision set (int8 encoder, fp16 head, ...).
SHIPPED_LOOP_IOU = {1024: 0.98, 512: 0.97}

Arrays = dict[str, np.ndarray]


@dataclass
class Prompt:
    name: str
    coords: np.ndarray  # [1, P, 2] float32, S-pixel space
    labels: np.ndarray  # [1, P] int32


@dataclass
class Stat:
    """Running max abs error, IoUs and argmax flips for one (graph, precision) row."""

    max_abs: float = 0.0
    ious: list[float] = field(default_factory=list)
    flips: int = 0
    has_error: bool = True

    def add_error(self, got: Arrays, want: Arrays) -> None:
        for key, ref in want.items():
            self.max_abs = max(self.max_abs, float(np.abs(got[key] - ref).max()))

    def add_masks(self, got: Arrays, want: Arrays) -> None:
        best = 1 + int(np.argmax(want["iou_predictions"][0, 1:]))
        mine = 1 + int(np.argmax(got["iou_predictions"][0, 1:]))
        self.flips += int(best != mine)
        self.ious.append(mask_iou(got["masks"][0, best], want["masks"][0, best]))


@dataclass
class Row:
    graph: str
    precision: str
    size: int
    file_mb: float
    max_abs: float | None
    iou_min: float | None
    iou_mean: float | None
    flips: int | None  # None where no argmax choice is compared (loop rows)
    cases: int
    latency_ms: float | None
    passed: bool | None
    criterion: str
    note: str = ""


def mask_iou(a: np.ndarray, b: np.ndarray, threshold: float = 0.0) -> float:
    """IoU of two logit masks binarised at ``threshold``; 1.0 when both are empty."""
    pa, pb = a > threshold, b > threshold
    union = int((pa | pb).sum())
    return 1.0 if union == 0 else int((pa & pb).sum()) / union


def load_image(name: str) -> np.ndarray:
    return np.array(Image.open(IMAGE_DIR / f"{name}.jpg").convert("RGB"))


def preprocess(sam: TorchSam, image: np.ndarray, size: int) -> torch.Tensor:
    """The image predictor's resize + normalisation at ``size``."""
    predictor = ImagePredictor(
        sam.prompt_encoder(size),
        sam.mask_decoder,
        image_size=size,
        pixel_mean=sam.cfg.pixel_mean,
        pixel_std=sam.cfg.pixel_std,
    )
    return predictor.preprocess(image)


def manifest_prompts(model: str, image: str, size: int) -> tuple[tuple[int, int], list[Prompt]]:
    """Manifest prompts in S-pixel space, converted like ``ImagePredictor.predict``.

    A box becomes two corner points (labels 2, 3) placed before any clicks; the padding
    token is appended inside the head graph, as the prompt encoder does with no boxes.
    """
    manifest = json.loads((DUMPS / model / "manifest.json").read_text())
    info = manifest["images"][image]
    h, w = info["orig_hw"]
    scale = np.array([size / w, size / h], dtype=np.float32)
    prompts: list[Prompt] = []
    for p in info["prompts"]:
        coords: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        if "box" in p:
            coords.append(np.asarray(p["box"], dtype=np.float32).reshape(2, 2))
            labels.append(np.array([LABEL_BOX_TL, LABEL_BOX_BR], dtype=np.int32))
        if "points" in p:
            coords.append(np.asarray(p["points"], dtype=np.float32).reshape(-1, 2))
            labels.append(np.asarray(p["labels"], dtype=np.int32))
        c = (np.concatenate(coords) * scale).astype(np.float32)[None]
        lbl = np.concatenate(labels).astype(np.int32)[None]
        prompts.append(Prompt(p["name"], c, lbl))
    return (h, w), prompts


def cpu_session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def run(sess: ort.InferenceSession, feeds: Arrays) -> Arrays:
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds), strict=True))


def latency_ms(sess: ort.InferenceSession, feeds: Arrays, runs: int) -> float:
    sess.run(None, feeds)  # warm-up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def head_feeds(feats: Arrays, prompt: Prompt, mask: np.ndarray | None, size: int) -> Arrays:
    g4 = size // 4
    return {
        **feats,
        "point_coords": prompt.coords,
        "point_labels": prompt.labels,
        "mask_input": mask if mask is not None else np.zeros((1, 1, g4, g4), np.float32),
        "has_mask_input": np.array([1.0 if mask is not None else 0.0], np.float32),
    }


def torch_head(
    sam: TorchSam,
    size: int,
    feats: tuple[torch.Tensor, ...],
    prompt: Prompt,
    mask: np.ndarray | None,
) -> Arrays:
    assert len(feats) == 3
    out = sam.decode(
        size,
        (feats[0], feats[1], feats[2]),
        torch.from_numpy(prompt.coords),
        torch.from_numpy(prompt.labels),
        None if mask is None else torch.from_numpy(mask),
    )
    keys = ("masks", "iou_predictions", "object_score_logits", "sam_tokens")
    return {k: v.numpy() for k, v in zip(keys, out, strict=True)}


def graph_paths(exports: Path, model: str, size: int) -> dict[tuple[str, str], Path]:
    enc = exports / f"{file_stem('image_encoder', model, size)}.onnx"
    head = exports / f"{file_stem('sam_head', model, size)}.onnx"
    paths = {
        ("image_encoder", "fp32"): enc,
        ("image_encoder", "int8"): enc.with_suffix(".int8.onnx"),
        ("sam_head", "fp32"): head,
        ("sam_head", "fp16"): head.with_suffix(".fp16.onnx"),
    }
    return {k: v for k, v in paths.items() if v.exists()}


@torch.no_grad()
def evaluate(
    model: str,
    size: int,
    exports: Path = EXPORTS,
    images: Sequence[str] = IMAGES,
    sam: TorchSam | None = None,
    latency_runs: int = 5,
) -> list[Row]:
    """Compare every exported variant at ``size`` with torch; one Row per variant."""
    sam = sam or load_torch(model)
    paths = graph_paths(exports, model, size)
    if ("image_encoder", "fp32") not in paths or ("sam_head", "fp32") not in paths:
        raise FileNotFoundError(f"fp32 graphs for {model} {size} missing under {exports}")
    sessions = {k: cpu_session(p) for k, p in paths.items()}
    encoders = [k for k in sessions if k[0] == "image_encoder"]
    heads = [k for k in sessions if k[0] == "sam_head"]
    head32 = sessions[("sam_head", "fp32")]
    stats: dict[tuple[str, str], Stat] = {k: Stat() for k in sessions}
    pipeline = ("pipeline", "int8+fp16")
    run_pipeline = ("image_encoder", "int8") in sessions and ("sam_head", "fp16") in sessions
    if run_pipeline:
        stats[pipeline] = Stat(has_error=False)

    enc_feeds: Arrays = {}
    last_head_feeds: Arrays = {}
    for name in images:
        _, prompts = manifest_prompts(model, name, size)
        x = preprocess(sam, load_image(name), size)
        ref_t = sam.encode(x)
        ref_feats = dict(
            zip(
                ("image_embed", "high_res_feat_s0", "high_res_feat_s1"),
                (t.numpy() for t in ref_t),
                strict=True,
            )
        )
        enc_feeds = {"image": x.numpy()}
        onnx_feats = {k: run(sessions[k], enc_feeds) for k in encoders}
        for k in encoders:
            stats[k].add_error(onnx_feats[k], ref_feats)

        for prompt in prompts:
            ref = torch_head(sam, size, ref_t, prompt, None)
            best = 1 + int(np.argmax(ref["iou_predictions"][0, 1:]))
            refine = np.clip(ref["masks"][:, best : best + 1], -32.0, 32.0).astype(np.float32)
            for mask in (None, refine):
                want = ref if mask is None else torch_head(sam, size, ref_t, prompt, mask)
                for k in heads:
                    last_head_feeds = head_feeds(ref_feats, prompt, mask, size)
                    got = run(sessions[k], last_head_feeds)
                    stats[k].add_error(got, want)
                    stats[k].add_masks(got, want)
                for k in encoders:
                    got = run(head32, head_feeds(onnx_feats[k], prompt, mask, size))
                    stats[k].add_masks(got, want)
                if run_pipeline:
                    feeds = head_feeds(onnx_feats[("image_encoder", "int8")], prompt, mask, size)
                    got = run(sessions[("sam_head", "fp16")], feeds)
                    stats[pipeline].add_masks(got, want)

    rows: list[Row] = []
    for key, stat in stats.items():
        graph, precision = key
        if key == pipeline:
            file_mb = sum(paths[k].stat().st_size for k in paths if k[1] in ("int8", "fp16"))
            lat = None
        else:
            file_mb = paths[key].stat().st_size
            feeds = enc_feeds if graph == "image_encoder" else last_head_feeds
            runs = latency_runs if graph == "image_encoder" else 4 * latency_runs
            lat = latency_ms(sessions[key], feeds, runs)
        label = graph if graph == "pipeline" else file_stem(graph, model, size)
        rows.append(_row(label, precision, size, file_mb / 1e6, stat, lat))
    return rows


def _row(graph: str, precision: str, size: int, mb: float, stat: Stat, lat: float | None) -> Row:
    iou_min = min(stat.ious)
    if precision == "fp32":
        passed: bool | None = stat.max_abs < FP32_MAX_ABS and iou_min >= FP32_IOU
        criterion = f"max|err| < {FP32_MAX_ABS:g}, IoU >= {FP32_IOU}"
    elif precision == "fp16":
        passed, criterion = iou_min >= FP16_IOU, f"IoU >= {FP16_IOU}"
    else:
        passed, criterion = iou_min >= INT8_IOU, f"IoU >= {INT8_IOU}"
    return Row(
        graph=graph,
        precision=precision,
        size=size,
        file_mb=round(mb, 2),
        max_abs=stat.max_abs if stat.has_error else None,
        iou_min=iou_min,
        iou_mean=float(np.mean(stat.ious)),
        flips=stat.flips,
        cases=len(stat.ious),
        latency_ms=lat,
        passed=passed,
        criterion=criterion,
    )


# --------------------------------------------------------------------------------------
# Video half: memory graphs against recorded calls, and the ORT loop end to end
# --------------------------------------------------------------------------------------


@dataclass
class MemoryEncoderCall:
    frame: int
    feeds: Arrays  # the memory_encoder graph inputs
    want: Arrays  # maskmem_features (fp32, with the no-object term), maskmem_pos_enc


@dataclass
class MemoryAttentionCall:
    frame: int
    feeds: Arrays  # the memory_attention graph inputs
    want: np.ndarray  # fused tokens [1, HW, 256]
    image: np.ndarray  # the frame's encoder input [1, 3, S, S], for the head IoU


def _encoder_call(
    sam: TorchSam,
    frame: int,
    vision_feat: torch.Tensor,
    high_res: torch.Tensor,
    score: torch.Tensor,
    want_feats: torch.Tensor,
    want_pos: torch.Tensor,
    binarize: bool,
) -> MemoryEncoderCall:
    """A memory-encoder graph call from the reference's inputs.

    The reference encoder sees ``20 * sigmoid(high_res) - 10`` (``20 * (high_res > 0) -
    10`` when binarised) where ``high_res`` is the bilinear upsample of the frame's low-res
    logits; the graph is fed that ``high_res``. ``want_feats`` excludes the no-object
    term, which is added here as the model does.
    """
    no_obj = sam.memory().no_obj_embed_spatial.reshape(1, -1, 1, 1)
    is_obj = (score > 0).float().reshape(1, 1, 1, 1)
    feats = want_feats + (1 - is_obj) * no_obj
    return MemoryEncoderCall(
        frame,
        {
            "vision_feat": vision_feat.numpy(),
            "mask_logits": high_res.numpy(),
            "object_score_logits": score.numpy(),
            "binarize": np.array([1.0 if binarize else 0.0], np.float32),
        },
        {"maskmem_features": feats.numpy(), "maskmem_pos_enc": want_pos.numpy()},
    )


def _upsample4(low_res: torch.Tensor) -> torch.Tensor:
    s = 4 * low_res.shape[-1]
    return F.interpolate(low_res, size=(s, s), mode="bilinear", align_corners=False)


def _synthetic_encoder_calls(sam: TorchSam, base: MemoryEncoderCall) -> list[MemoryEncoderCall]:
    """Branches the bedroom clip never reaches (the object is visible on every frame):
    a binarised clicked mask, and an absent object with each mask mode."""
    cfg = sam.cfg
    pix = torch.from_numpy(base.feeds["vision_feat"])
    high_res = torch.from_numpy(base.feeds["mask_logits"])
    calls = []
    for binarize, score in ((True, 2.0), (False, -2.0), (True, -2.0)):
        mask = (high_res > 0).float() if binarize else torch.sigmoid(high_res)
        mask = mask * cfg.sigmoid_scale_for_mem_enc + cfg.sigmoid_bias_for_mem_enc
        out = sam.memory().memory_encoder(pix, mask, skip_mask_sigmoid=True)
        feats, pos = out["vision_features"], out["vision_pos_enc"]
        assert isinstance(feats, torch.Tensor) and isinstance(pos, list)
        calls.append(
            _encoder_call(
                sam, base.frame, pix, high_res, torch.tensor([[score]]), feats, pos[0], binarize
            )
        )
    return calls


def _attention_call(
    frame: int, kw: dict[str, Any], out: torch.Tensor, image: np.ndarray
) -> MemoryAttentionCall:
    curr = kw["curr"][0] if isinstance(kw["curr"], list) else kw["curr"]
    return MemoryAttentionCall(
        frame,
        {
            "curr": curr.transpose(0, 1).contiguous().numpy(),
            "memory": kw["memory"].transpose(0, 1).contiguous().numpy(),
            "memory_pos": kw["memory_pos"].transpose(0, 1).contiguous().numpy(),
            "num_obj_ptr_tokens": np.array([int(kw["num_obj_ptr_tokens"])], np.int64),
        },
        out.transpose(0, 1).contiguous().numpy(),
        image,
    )


def dumped_memory_calls(
    sam: TorchSam, model: str = "tiny", tag: str = "fwd"
) -> tuple[list[MemoryEncoderCall], list[MemoryAttentionCall]]:
    """The memory calls recorded by ``tools/dump_reference.py`` (1024 px, frames 1, 2, 7,
    20, 47 of the bedroom run)."""
    dump = torch.load(
        DUMPS / model / f"video_bedroom_{tag}.pt", map_location="cpu", weights_only=False
    )
    enc: list[MemoryEncoderCall] = []
    for frame, call in dump["memory_encoder_calls"].items():
        out = dump["frames"][frame]
        pix, _ = call["args"]
        ref = call["output"]
        enc.append(
            _encoder_call(
                sam,
                frame,
                pix,
                _upsample4(out["pred_masks"]),
                out["object_score_logits"],
                ref["vision_features"],
                ref["vision_pos_enc"][0],
                binarize=False,
            )
        )
    attn = [
        _attention_call(f, c["kwargs"], c["output"], dump["input_frames"][f : f + 1].numpy())
        for f, c in dump["memory_attention_calls"].items()
    ]
    return enc, attn


@torch.no_grad()
def recorded_memory_calls(
    sam: TorchSam, size: int, frames_hooked: Sequence[int] = (1, 2, 7, 20, 47)
) -> tuple[list[MemoryEncoderCall], list[MemoryAttentionCall]]:
    """The same calls recorded from this repo's torch ``VideoPredictor`` at ``size``
    (bedroom forward run), for sizes without an official dump."""
    import dataclasses

    from tools import ort_video_loop as loop

    from sam2_scratch.configs import TINY
    from sam2_scratch.model import SAM2
    from sam2_scratch.video_predictor import VideoPredictor

    model = SAM2.from_checkpoint(loop.CHECKPOINT, dataclasses.replace(TINY, image_size=size))
    predictor = VideoPredictor(model)
    state = predictor.init_state(loop.raw_frames())
    first = loop.RUNS[0]
    frame0, reverse = first.frame, first.reverse
    predictor.add_points(state, frame0, 1, [list(q) for q in first.points], list(first.labels))

    current = {"frame": frame0}
    attn: list[MemoryAttentionCall] = []
    enc_inputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    enc_outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def attn_hook(_m: Any, _args: Any, kwargs: dict[str, Any], output: torch.Tensor) -> None:
        f = current["frame"]
        if f in frames_hooked:
            image = state.images[f : f + 1].numpy()
            attn.append(_attention_call(f, kwargs, output.clone(), image))

    def enc_hook(_m: Any, args: Any, kwargs: dict[str, Any], output: dict[str, Any]) -> None:
        f = current["frame"]
        if f in frames_hooked:
            enc_inputs[f] = (args[0].clone(), args[1].clone())
            enc_outputs[f] = (
                output["vision_features"].clone(),
                output["vision_pos_enc"][0].clone(),
            )

    handles = [
        model.memory_attention.register_forward_hook(attn_hook, with_kwargs=True),
        model.memory_encoder.register_forward_hook(enc_hook, with_kwargs=True),
    ]
    order = iter(range(frame0 + 1, state.num_frames))
    try:
        for _ in predictor.propagate(state, reverse=reverse):
            current["frame"] = next(order, -1)  # the frame the generator tracks next
    finally:
        for h in handles:
            h.remove()
    out_dict = state.objects[1].output["non_cond_frame_outputs"]
    enc = [
        _encoder_call(
            sam,
            f,
            enc_inputs[f][0],
            _upsample4(out_dict[f].pred_masks),
            out_dict[f].object_score_logits,
            *enc_outputs[f],
            binarize=False,
        )
        for f in sorted(enc_inputs)
    ]
    return enc, attn


def video_graph_paths(exports: Path, model: str, size: int) -> dict[tuple[str, str], Path]:
    menc = exports / f"{file_stem('memory_encoder', model, size)}.onnx"
    mattn = exports / f"{file_stem('memory_attention', model, size)}.onnx"
    paths = {
        ("memory_encoder", "fp32"): menc,
        ("memory_encoder", "fp16"): menc.with_suffix(".fp16.onnx"),
        ("memory_attention", "fp32"): mattn,
        ("memory_attention", "fp16"): mattn.with_suffix(".fp16.onnx"),
    }
    return {k: v for k, v in paths.items() if v.exists()}


def _pad_head_masks(
    head: ort.InferenceSession, feats: Arrays, embed: np.ndarray, size: int
) -> Arrays:
    """The tracking-frame head call (one padding point, no mask) on ``embed``."""
    pad = Prompt("pad", np.zeros((1, 1, 2), np.float32), np.full((1, 1), -1, np.int32))
    return run(head, head_feeds({**feats, "image_embed": embed}, pad, None, size))


@torch.no_grad()
def evaluate_video_graphs(
    model: str,
    size: int,
    exports: Path = EXPORTS,
    sam: TorchSam | None = None,
    latency_runs: int = 5,
) -> list[Row]:
    """Memory encoder / attention / pointer graphs against recorded reference calls."""
    sam = sam or load_torch(model)
    paths = video_graph_paths(exports, model, size)
    if ("memory_attention", "fp32") not in paths:
        raise FileNotFoundError(f"memory graphs for {model} {size} missing under {exports}")
    if size == 1024:
        enc_calls, attn_calls = dumped_memory_calls(sam, model)
        source = "official dump"
    else:
        enc_calls, attn_calls = recorded_memory_calls(sam, size)
        source = f"torch at {size}"
    enc_calls += _synthetic_encoder_calls(sam, enc_calls[0])
    encoder = cpu_session(exports / f"{file_stem('image_encoder', model, size)}.onnx")
    head = cpu_session(exports / f"{file_stem('sam_head', model, size)}.onnx")
    rows: list[Row] = []
    frames = ",".join(str(c.frame) for c in attn_calls)
    synthetic = "; encoder also 3 synthetic (clicked mask, object absent)"
    for (graph, precision), path in paths.items():
        sess = cpu_session(path)
        stat = Stat()
        if graph == "memory_encoder":
            for enc_call in enc_calls:
                stat.add_error(run(sess, enc_call.feeds), enc_call.want)
            feeds = enc_calls[-1].feeds
            ious: list[float] | None = None
        else:
            for attn_call in attn_calls:
                got = run(sess, attn_call.feeds)
                stat.add_error({"fused": got["fused"]}, {"fused": attn_call.want})
                g = size // 16
                feats = run(encoder, {"image": attn_call.image})
                want_embed = attn_call.want.transpose(0, 2, 1).reshape(1, -1, g, g)
                want = _pad_head_masks(head, feats, want_embed.astype(np.float32), size)
                mine = _pad_head_masks(head, feats, got["image_embed"], size)
                stat.add_masks(mine, want)
            feeds = attn_calls[-1].feeds
            ious = stat.ious
        lat = latency_ms(sess, feeds, latency_runs)
        rows.append(
            _video_row(
                file_stem(graph, model, size),
                precision,
                size,
                path.stat().st_size / 1e6,
                stat.max_abs,
                ious,
                stat.flips,
                len(enc_calls if ious is None else attn_calls),
                lat,
                f"{source}, frames {frames}" + (synthetic if graph == "memory_encoder" else ""),
            )
        )
    return rows


def _video_row(
    graph: str,
    precision: str,
    size: int,
    mb: float,
    max_abs: float | None,
    ious: list[float] | None,
    flips: int | None,
    cases: int,
    lat: float | None,
    note: str,
    iou_floor: float | None = None,
) -> Row:
    iou_min = min(ious) if ious else None
    if precision == "fp32":
        passed: bool | None = (max_abs is None or max_abs < FP32_MAX_ABS) and (
            iou_min is None or iou_min >= FP32_IOU
        )
        criterion = (
            f"IoU >= {FP32_IOU}"
            if max_abs is None
            else f"max|err| < {FP32_MAX_ABS:g}" + (f", IoU >= {FP32_IOU}" if ious else "")
        )
    elif iou_floor is not None or iou_min is not None:
        floor = iou_floor if iou_floor is not None else FP16_IOU
        passed, criterion = (iou_min is not None and iou_min >= floor), f"IoU >= {floor}"
    else:
        passed, criterion = None, "reported"
    return Row(
        graph=graph,
        precision=precision,
        size=size,
        file_mb=round(mb, 2),
        max_abs=max_abs,
        iou_min=iou_min,
        iou_mean=float(np.mean(ious)) if ious else None,
        flips=flips,
        cases=cases,
        latency_ms=lat,
        passed=passed,
        criterion=criterion,
        note=note,
    )


@torch.no_grad()
def evaluate_pointer_graphs(
    model: str, exports: Path = EXPORTS, sam: TorchSam | None = None
) -> list[Row]:
    """Object-pointer graphs against the torch modules (random tokens, both occlusion
    states; every temporal distance of a 48-frame clip at the full and a capped window)."""
    sam = sam or load_torch(model)
    mem = sam.memory()
    suffix = "" if model == "tiny" else f"_{model}"
    rows: list[Row] = []
    ptr_path = exports / f"object_pointer{suffix}.onnx"
    tpos_path = exports / f"object_pointer_tpos{suffix}.onnx"
    gen = torch.Generator().manual_seed(0)
    ptr_sess, tpos_sess = cpu_session(ptr_path), cpu_session(tpos_path)
    stat = Stat()
    for i in range(16):
        token = torch.randn(1, sam.cfg.hidden_dim, generator=gen) * 2
        score = torch.tensor([[3.0 if i % 2 else -3.0]])
        is_obj = (score > 0).float()
        want = is_obj * mem.obj_ptr_proj(token) + (1 - is_obj) * mem.no_obj_ptr
        got = run(ptr_sess, {"mask_token": token.numpy(), "object_score_logits": score.numpy()})
        stat.add_error(got, {"obj_ptr": want.numpy()})
    rows.append(
        _video_row(
            "object_pointer",
            "fp32",
            0,
            ptr_path.stat().st_size / 1e6,
            stat.max_abs,
            None,
            0,
            16,
            latency_ms(
                ptr_sess, {"mask_token": token.numpy(), "object_score_logits": score.numpy()}, 20
            ),
            "random tokens, object present / absent",
        )
    )
    stat = Stat()
    t = torch.arange(48)
    for t_max in (15, 2):
        want = mem.obj_ptr_tpos_proj(sine_pe_1d(t / t_max, dim=sam.cfg.hidden_dim))
        feeds = {"t_diff": t.float().numpy(), "t_diff_max": np.array([t_max], np.float32)}
        stat.add_error(run(tpos_sess, feeds), {"ptr_pos": want.numpy()})
    rows.append(
        _video_row(
            "object_pointer_tpos",
            "fp32",
            0,
            tpos_path.stat().st_size / 1e6,
            stat.max_abs,
            None,
            0,
            96,
            latency_ms(tpos_sess, feeds, 20),
            "t_diff 0..47, t_max 15 and 2",
        )
    )
    return rows


def evaluate_video_loop(
    size: int, exports: Path = EXPORTS, precision_sets: Sequence[str] = ("fp32", "shipped")
) -> tuple[list[Row], dict[str, Any]]:
    """End-to-end rows from ``tools/ort_video_loop.py`` (forward and reverse runs)."""
    from tools import ort_video_loop as loop

    rows: list[Row] = []
    details: dict[str, Any] = {}
    references: dict[str, loop.Reference] = {}
    for name in precision_sets:
        precisions = loop.PRECISION_SETS[name]
        result = loop.evaluate(size, precisions, exports, references=references)
        details[name] = result
        mb = sum((exports / f).stat().st_size for f in result["files"].values()) / 1e6
        floor = FP32_IOU if name == "fp32" else SHIPPED_LOOP_IOU[size]
        for tag, r in result["runs"].items():
            t = r["timings"]
            per_frame = t["tracked_frame_total"]["median_ms"] + result["image_encoder_ms"]
            label = "+".join(precisions[k] for k in loop.PRECISION_SETS["fp32"])
            rows.append(
                _video_row(
                    f"video loop {tag}",
                    name,
                    size,
                    mb,
                    None,
                    [v["iou"] for v in r["per_frame"].values()],
                    None,
                    r["frames"],
                    per_frame,
                    f"{label}; worst frame {r['worst_frame']}, max|logit err| "
                    f"{r['logit_err_max']:.2g}, max|score err| {r['score_err_max']:.2g}; "
                    f"vs {result['reference']}",
                    iou_floor=floor,
                )
            )
    return rows, details


def format_table(rows: Sequence[Row]) -> str:
    header = (
        f"{'graph':<24} {'prec':<9} {'MB':>6} {'max|err|':>9} {'IoU min':>8} {'IoU mean':>8}"
        f" {'flips':>7} {'ORT ms':>7}  result"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        err = f"{r.max_abs:.2e}" if r.max_abs is not None else "-"
        lat = f"{r.latency_ms:.1f}" if r.latency_ms is not None else "-"
        verdict = "info" if r.passed is None else "PASS" if r.passed else "FAIL"
        i_min = f"{r.iou_min:.4f}" if r.iou_min is not None else "-"
        i_mean = f"{r.iou_mean:.4f}" if r.iou_mean is not None else "-"
        note = f"  [{r.note}]" if r.note else ""
        flips = "-" if r.flips is None else r.flips
        lines.append(
            f"{r.graph:<24} {r.precision:<9} {r.file_mb:>6.1f} {err:>9} {i_min:>8}"
            f" {i_mean:>8} {f'{flips}/{r.cases}':>7} {lat:>7}  {verdict} ({r.criterion}){note}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=("tiny", "small"), default="tiny")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--images", nargs="+", default=list(IMAGES))
    parser.add_argument("--exports", type=Path, default=EXPORTS)
    parser.add_argument("--latency-runs", type=int, default=5)
    parser.add_argument("--json", type=Path, help="also write the rows as JSON here")
    parser.add_argument(
        "--parts", nargs="+", choices=("image", "video"), default=["image", "video"]
    )
    parser.add_argument(
        "--skip-loop", action="store_true", help="video graphs only, no end-to-end loop"
    )
    args = parser.parse_args(argv)

    sam = load_torch(args.model)
    rows: list[Row] = []
    loops: dict[str, Any] = {}
    for size in args.sizes:
        if "image" in args.parts:
            rows += evaluate(args.model, size, args.exports, args.images, sam, args.latency_runs)
        if "video" in args.parts:
            if args.model != "tiny":
                raise SystemExit("the video rows are defined for the shipped tiny model only")
            rows += evaluate_video_graphs(args.model, size, args.exports, sam, args.latency_runs)
            if not args.skip_loop:
                loop_rows, loops[str(size)] = evaluate_video_loop(size, args.exports)
                rows += loop_rows
    if "video" in args.parts:
        rows += evaluate_pointer_graphs(args.model, args.exports, sam)
    print(f"ORT {ort.__version__} CPU provider, model {args.model}, images {args.images}")
    print(format_table(rows))
    if args.json:
        payload: dict[str, Any] = {"model": args.model, "rows": [asdict(r) for r in rows]}
        if loops:
            payload["video_loop"] = loops
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if all(r.passed is not False for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
