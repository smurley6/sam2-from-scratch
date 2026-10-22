"""SAM 2 video tracking driven purely through onnxruntime: the blueprint for the browser.

``OrtVideoTracker`` reproduces ``sam2_scratch.video_predictor.VideoPredictor`` (one
object) with the exported graphs, ``numpy`` and the ``constants_{S}.json`` file only: no
torch and no checkpoint inside the loop. Each method maps onto one step of the recipe in
``exports/README.md``; the TypeScript demo follows the same steps.

Per tracked frame::

    image_encoder -> (raw tokens) -> object_pointer_tpos -> memory_attention -> sam_head
        -> object_pointer -> memory_encoder -> memory bank

The command line runs the bedroom clip forward from a click on frame 0 and in reverse
from a click on frame 30 (the prompts of ``tools/dump_reference.py``), and compares every
frame with a reference: the official package's dump at 1024, or this repo's torch
``VideoPredictor`` run at the same size otherwise (torch is imported for that comparison
only, after the loop has run)::

    python tools/ort_video_loop.py                       # fp32 graphs, 1024, vs the dump
    python tools/ort_video_loop.py --size 512 --precision shipped --json loop512.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:  # the torch reference (comparison only) lives in src
        sys.path.insert(0, str(_p))
EXPORTS = ROOT / "exports"
VIDEO_DIR = ROOT / "data" / "videos" / "bedroom"
DUMPS = ROOT / "reference_dumps"
CHECKPOINT = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
FRAME_SUFFIXES = (".jpg", ".jpeg", ".png")

# The two runs of tools/dump_reference.py: (tag, frame, points (x, y) in video pixels,
# labels, reverse).
RUNS: tuple[tuple[str, int, list[list[float]], list[int], bool], ...] = (
    ("fwd", 0, [[210.0, 350.0]], [1], False),
    ("rev", 30, [[210.0, 350.0]], [1], True),
)

# Graph precision per role. "shipped" is the browser set (see exports/README.md).
PRECISION_SETS: dict[str, dict[str, str]] = {
    "fp32": {
        "image_encoder": "fp32",
        "sam_head": "fp32",
        "memory_encoder": "fp32",
        "memory_attention": "fp32",
    },
    "shipped": {
        "image_encoder": "int8",
        "sam_head": "fp16",
        "memory_encoder": "fp16",
        "memory_attention": "fp16",
    },
}
SUFFIX = {"fp32": ".onnx", "int8": ".int8.onnx", "fp16": ".fp16.onnx"}
LABEL_PAD = -1

Arrays = dict[str, np.ndarray]


# --------------------------------------------------------------------------------------
# Graphs and constants
# --------------------------------------------------------------------------------------


def session(path: Path) -> ort.InferenceSession:
    """CPU session; int8 weights are dequantised once at load (identical outputs)."""
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.add_session_config_entry("session.disable_quant_qdq", "1")
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


@dataclass
class Graphs:
    """One onnxruntime session per exported graph, for one input size."""

    image_encoder: ort.InferenceSession
    sam_head: ort.InferenceSession
    memory_encoder: ort.InferenceSession
    memory_attention: ort.InferenceSession
    object_pointer: ort.InferenceSession
    object_pointer_tpos: ort.InferenceSession
    files: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def load(cls, exports: Path, size: int, precisions: dict[str, str]) -> Graphs:
        files = {
            role: exports / f"{role}_{size}{SUFFIX[precisions[role]]}"
            for role in ("image_encoder", "sam_head", "memory_encoder", "memory_attention")
        }
        files["object_pointer"] = exports / "object_pointer.onnx"
        files["object_pointer_tpos"] = exports / "object_pointer_tpos.onnx"
        missing = [str(p) for p in files.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"run tools/export_onnx.py first; missing {missing}")
        sessions = {role: session(p) for role, p in files.items()}
        return cls(**sessions, files=files)


@dataclass
class Constants:
    """``constants_{S}.json`` with the tensors as float32 arrays."""

    raw: dict[str, Any]
    no_mem_embed: np.ndarray  # [256]
    maskmem_tpos_enc: np.ndarray  # [num_maskmem, 64]

    @classmethod
    def load(cls, path: Path) -> Constants:
        raw = json.loads(path.read_text())
        return cls(
            raw,
            np.asarray(raw["no_mem_embed"], np.float32),
            np.asarray(raw["maskmem_tpos_enc"], np.float32),
        )

    def __getattr__(self, name: str) -> Any:
        raw = self.__dict__.get("raw", {})
        if name in raw:
            return raw[name]
        raise AttributeError(name)


def run(sess: ort.InferenceSession, feeds: Arrays) -> Arrays:
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds), strict=True))


# --------------------------------------------------------------------------------------
# numpy helpers (each is a few lines of TypeScript)
# --------------------------------------------------------------------------------------


def load_video_frames(
    frames: Path | Sequence[np.ndarray],
    size: int,
    mean: Sequence[float],
    std: Sequence[float],
    max_frames: int | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """``video_predictor.load_frames`` in numpy: PIL bicubic to ``size x size`` (aspect
    not kept), ``uint8 / 255`` in float64, then float32 ``(x - mean) / std``, CHW."""
    if isinstance(frames, Path):
        paths = sorted(
            (p for p in frames.iterdir() if p.suffix.lower() in FRAME_SUFFIXES),
            key=lambda p: (int(p.stem), p.name) if p.stem.isdigit() else (0, p.name),
        )
        pil = [Image.open(p).convert("RGB") for p in paths[:max_frames]]
    else:
        pil = [Image.fromarray(a) for a in frames[:max_frames]]
    width, height = pil[0].size
    out = np.empty((len(pil), 3, size, size), np.float32)
    m = np.asarray(mean, np.float32)[:, None, None]
    s = np.asarray(std, np.float32)[:, None, None]
    for i, im in enumerate(pil):
        x = np.asarray(im.resize((size, size), Image.Resampling.BICUBIC)) / 255.0
        out[i] = (x.transpose(2, 0, 1).astype(np.float32) - m) / s
    return out, (height, width)


def _linear_taps(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source taps of bilinear resizing with half-pixel centres (``align_corners=False``)."""
    scale = n_in / n_out
    src = np.maximum((np.arange(n_out, dtype=np.float32) + 0.5) * np.float32(scale) - 0.5, 0)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    return i0, i1, (src - i0).astype(np.float32)


def upsample_bilinear(x: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """``F.interpolate(x, size, mode="bilinear", align_corners=False)`` for ``[..., h, w]``."""
    y0, y1, wy = _linear_taps(x.shape[-2], size[0])
    x0, x1, wx = _linear_taps(x.shape[-1], size[1])
    rows = x[..., y0, :] * (1 - wy)[:, None] + x[..., y1, :] * wy[:, None]
    out: np.ndarray = rows[..., x0] * (1 - wx) + rows[..., x1] * wx
    return out.astype(np.float32)


def round_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 to the nearest bfloat16 (ties to even), returned as float32.

    The reference stores memory features as bfloat16; this reproduces that storage.
    """
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    u = (u + (np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1)))) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


def stability_score(logits: np.ndarray, delta: float) -> float:
    """``count(logit > delta) / count(logit > -delta)``; 1.0 when the denominator is 0."""
    inter = int(np.count_nonzero(logits > delta))
    union = int(np.count_nonzero(logits > -delta))
    return 1.0 if union == 0 else float(np.float32(inter) / np.float32(union))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = a > 0, b > 0
    union = int((pa | pb).sum())
    return 1.0 if union == 0 else int((pa & pb).sum()) / union


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------


@dataclass
class Features:
    """Encoder outputs of one frame plus the raw tokens the memory graphs take."""

    image_embed: np.ndarray  # [1, 256, g, g], no_mem_embed added (encoder output)
    high_res_feat_s0: np.ndarray  # [1, 32, 4g, 4g]
    high_res_feat_s1: np.ndarray  # [1, 64, 2g, 2g]
    vision_feat: np.ndarray  # [1, 256, g, g] raw = image_embed - no_mem_embed
    curr: np.ndarray  # [1, g*g, 256] raw tokens, row-major over the grid


@dataclass
class FrameOutput:
    """What the memory bank keeps for one frame (``model.FrameOutput``)."""

    pred_masks: np.ndarray  # [1, 1, 4g, 4g] chosen low-res logits (NO_OBJ_SCORE if absent)
    obj_ptr: np.ndarray  # [1, 256]
    object_score_logits: np.ndarray  # [1, 1]
    maskmem_features: np.ndarray | None = None  # [1, 64, g, g], bfloat16-rounded


@dataclass
class Timings:
    per_graph: dict[str, list[float]] = field(default_factory=dict)
    per_frame: list[float] = field(default_factory=list)

    def add(self, graph: str, ms: float) -> None:
        self.per_graph.setdefault(graph, []).append(ms)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            g: {"median_ms": round(statistics.median(v), 2), "calls": len(v)}
            for g, v in self.per_graph.items()
        }
        if self.per_frame:
            out["tracked_frame_total"] = {
                "median_ms": round(statistics.median(self.per_frame), 1),
                "frames": len(self.per_frame),
            }
        return out


class OrtVideoTracker:
    """One object tracked through one video with the exported graphs.

    Mirrors ``VideoPredictor.add_points`` / ``propagate`` and ``SAM2.track_step``; the
    method docstrings name the torch function each step reproduces.
    """

    def __init__(
        self,
        graphs: Graphs,
        consts: Constants,
        frames: np.ndarray,
        video_hw: tuple[int, int],
        bf16_memory: bool = True,
        encoder_cache: dict[int, Features] | None = None,
    ) -> None:
        self.g = graphs
        self.c = consts
        self.frames = frames
        self.video_hw = video_hw
        self.num_frames = int(frames.shape[0])
        self.size = int(consts.image_size)
        self.grid = int(consts.embed_grid)
        self.bf16_memory = bf16_memory
        self.cond: dict[int, FrameOutput] = {}
        self.non_cond: dict[int, FrameOutput] = {}
        self.pending: dict[int, FrameOutput] = {}  # prompted, memory not yet encoded
        self.tracked: dict[int, bool] = {}  # frame -> reverse flag of its last tracking
        self.points: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.maskmem_pos: np.ndarray | None = None  # [g*g, 64] tokens, cached once
        self._cache = encoder_cache  # optional cross-run cache (parity runs only)
        self._last: tuple[int, Features] | None = None
        self.timings = Timings()

    def _timed(self, name: str, sess: ort.InferenceSession, feeds: Arrays) -> Arrays:
        t0 = time.perf_counter()
        out = run(sess, feeds)
        self.timings.add(name, (time.perf_counter() - t0) * 1e3)
        return out

    # ---------------------------------------------------------------- step 1: encode

    def features(self, frame: int) -> Features:
        """``VideoPredictor._features``: encode once per frame (one-frame cache)."""
        if self._cache is not None and frame in self._cache:
            return self._cache[frame]
        if self._last is not None and self._last[0] == frame:
            return self._last[1]
        out = self._timed(
            "image_encoder", self.g.image_encoder, {"image": self.frames[frame][None]}
        )
        embed = out["image_embed"]
        raw = (embed - self.c.no_mem_embed[None, :, None, None]).astype(np.float32)
        curr = np.ascontiguousarray(raw.reshape(1, raw.shape[1], -1).transpose(0, 2, 1))
        feats = Features(embed, out["high_res_feat_s0"], out["high_res_feat_s1"], raw, curr)
        self._last = (frame, feats)
        if self._cache is not None:
            self._cache[frame] = feats
        return feats

    # ---------------------------------------------------------------- step 2: SAM head

    def sam_head(
        self,
        image_embed: np.ndarray,
        feats: Features,
        coords: np.ndarray,
        labels: np.ndarray,
        mask_input: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``SAM2.sam_heads`` (head graph + output selection + object pointer).

        Returns ``(low-res logits [1, 1, m, m], obj_ptr [1, 256], score [1, 1])``.
        """
        m = int(self.c.mask_size)
        out = self._timed(
            "sam_head",
            self.g.sam_head,
            {
                "image_embed": image_embed,
                "high_res_feat_s0": feats.high_res_feat_s0,
                "high_res_feat_s1": feats.high_res_feat_s1,
                "point_coords": coords.astype(np.float32),
                "point_labels": labels.astype(np.int32),
                "mask_input": np.zeros((1, 1, m, m), np.float32)
                if mask_input is None
                else mask_input.astype(np.float32),
                "has_mask_input": np.array([0.0 if mask_input is None else 1.0], np.float32),
            },
        )
        masks, ious = out["masks"][0], out["iou_predictions"][0]
        tokens, score = out["sam_tokens"][0], out["object_score_logits"]
        num_points = int((labels != LABEL_PAD).sum())
        best_multi = 1 + int(np.argmax(ious[1:]))  # first maximum, as torch.argmax
        if self.c.multimask_min_pt_num <= num_points <= self.c.multimask_max_pt_num:
            chosen, token = masks[best_multi], tokens[best_multi]
        else:
            # Single-mask call: token 0 unless unstable; the pointer always uses token 0.
            stable = stability_score(masks[0], self.c.stability_delta) >= self.c.stability_thresh
            chosen, token = (masks[0] if stable else masks[best_multi]), tokens[0]
        if not float(score[0, 0]) > 0:
            chosen = np.full_like(chosen, self.c.no_obj_score)
        ptr = self._timed(
            "object_pointer",
            self.g.object_pointer,
            {"mask_token": token[None].astype(np.float32), "object_score_logits": score},
        )["obj_ptr"]
        return chosen[None, None].astype(np.float32), ptr, score

    # ---------------------------------------------------------------- step 3: memory

    def encode_memory(self, frame: int, out: FrameOutput, from_clicks: bool) -> None:
        """``SAM2.encode_memory`` + ``VideoPredictor._store_memory``."""
        feats = self.features(frame)
        high_res = upsample_bilinear(out.pred_masks, (self.size, self.size))
        mem = self._timed(
            "memory_encoder",
            self.g.memory_encoder,
            {
                "vision_feat": feats.vision_feat,
                "mask_logits": high_res,
                "object_score_logits": out.object_score_logits,
                "binarize": np.array([1.0 if from_clicks else 0.0], np.float32),
            },
        )
        f = mem["maskmem_features"]
        out.maskmem_features = round_bf16(f) if self.bf16_memory else f
        if self.maskmem_pos is None:
            pos = mem["maskmem_pos_enc"][0]
            self.maskmem_pos = np.ascontiguousarray(pos.reshape(pos.shape[0], -1).T)

    def _ptr_pos(self, t_diffs: list[int], max_ptrs: int) -> np.ndarray:
        return self._timed(
            "object_pointer_tpos",
            self.g.object_pointer_tpos,
            {
                "t_diff": np.asarray(t_diffs, np.float32),
                "t_diff_max": np.array([max_ptrs - 1], np.float32),
            },
        )["ptr_pos"]

    def memory_bank(self, frame: int, reverse: bool) -> tuple[np.ndarray, np.ndarray, int]:
        """The ``memory`` / ``memory_pos`` / ``num_obj_ptr_tokens`` that
        ``SAM2.condition_on_memory`` hands to the memory attention."""
        c = self.c
        n_mem = int(c.num_maskmem)
        stride = int(c.memory_temporal_stride_for_eval)
        selected, unselected = select_closest_cond_frames(
            frame, self.cond, int(c.max_cond_frames_in_attn)
        )
        # (a) every selected conditioning frame (t_pos 0), (b) the num_maskmem - 1 most
        # recent frames in tracking direction (t_pos 1 = oldest ... 6 = previous frame).
        slots: list[tuple[int, FrameOutput | None]] = [(0, o) for o in selected.values()]
        for t_pos in range(1, n_mem):
            t_rel = n_mem - t_pos
            if t_rel == 1:
                prev = frame + 1 if reverse else frame - 1
            elif reverse:
                prev = -(-(frame + 2) // stride) * stride + (t_rel - 2) * stride
            else:
                prev = ((frame - 2) // stride) * stride - (t_rel - 2) * stride
            out = self.non_cond.get(prev)
            slots.append((t_pos, out if out is not None else unselected.get(prev)))
        assert self.maskmem_pos is not None
        memory: list[np.ndarray] = []
        memory_pos: list[np.ndarray] = []
        for t_pos, prev_out in slots:
            if prev_out is None or prev_out.maskmem_features is None:
                continue
            f = prev_out.maskmem_features[0]
            memory.append(f.reshape(f.shape[0], -1).T)
            memory_pos.append(self.maskmem_pos + c.maskmem_tpos_enc[n_mem - t_pos - 1])

        # (c) object pointers: past conditioning frames, then up to max_ptrs - 1 recent
        # tracked frames, each with its signed distance to this frame.
        max_ptrs = min(self.num_frames, int(c.max_obj_ptrs_in_encoder))
        sign = -1 if reverse else 1
        if c.only_obj_ptrs_in_the_past_for_eval:
            ptr_cond = {
                t: o for t, o in selected.items() if (t >= frame if reverse else t <= frame)
            }
        else:
            ptr_cond = selected
        ptrs: list[tuple[int, np.ndarray]] = [
            ((frame - t) * sign, o.obj_ptr) for t, o in ptr_cond.items()
        ]
        for t_diff in range(1, max_ptrs):
            t = frame + t_diff if reverse else frame - t_diff
            if t < 0 or t >= self.num_frames:
                break
            out = self.non_cond.get(t, unselected.get(t))
            if out is not None:
                ptrs.append((t_diff, out.obj_ptr))
        n_ptr_tokens = 0
        if ptrs:
            split = int(c.tokens_per_obj_ptr)
            pos = self._ptr_pos([d for d, _ in ptrs], max_ptrs)  # [K, 64]
            # Pointer k becomes tokens 4k .. 4k+3 (its 256 channels in four 64-chunks),
            # all four with pointer k's temporal encoding.
            memory.append(np.concatenate([p.reshape(split, -1) for _, p in ptrs]))
            memory_pos.append(np.repeat(pos, split, axis=0))
            n_ptr_tokens = split * len(ptrs)
        bank = np.concatenate(memory)[None].astype(np.float32)
        bank_pos = np.concatenate(memory_pos)[None].astype(np.float32)
        return bank, bank_pos, n_ptr_tokens

    def condition(self, frame: int, is_init: bool, reverse: bool) -> np.ndarray:
        """``SAM2.condition_on_memory``: the SAM head's ``image_embed`` for this frame."""
        feats = self.features(frame)
        if is_init:
            return feats.image_embed  # raw + no_mem_embed, straight from the encoder
        memory, memory_pos, n_ptr = self.memory_bank(frame, reverse)
        out = self._timed(
            "memory_attention",
            self.g.memory_attention,
            {
                "curr": feats.curr,
                "memory": memory,
                "memory_pos": memory_pos,
                "num_obj_ptr_tokens": np.array([n_ptr], np.int64),
            },
        )
        return out["image_embed"]

    # ---------------------------------------------------------------- prompts / propagate

    def add_click(
        self, frame: int, points: Sequence[Sequence[float]], labels: Sequence[int]
    ) -> np.ndarray:
        """``VideoPredictor.add_points`` with ``clear_old_points=True`` and no box.

        ``points`` are (x, y) in video pixels. Returns the low-res logits.
        """
        h, w = self.video_hw
        xy = np.asarray(points, np.float32).reshape(1, -1, 2)
        coords = (xy / np.array([w, h], np.float32) * np.float32(self.size)).astype(np.float32)
        labs = np.asarray(labels, np.int32).reshape(1, -1)
        self.points[frame] = (coords, labs)
        prior = self.pending.get(frame) or self.cond.get(frame) or self.non_cond.get(frame)
        clamp = float(self.c.mask_input_clamp)
        mask_input = None if prior is None else np.clip(prior.pred_masks, -clamp, clamp)
        # A frame tracked before is a correction, conditioned on memory in its direction.
        is_init = frame not in self.tracked
        reverse = False if is_init else self.tracked[frame]
        embed = self.condition(frame, is_init, reverse)
        low, ptr, score = self.sam_head(embed, self.features(frame), coords, labs, mask_input)
        self.pending[frame] = FrameOutput(low, ptr, score)
        return low

    def _preflight(self) -> None:
        """``VideoPredictor._preflight``: encode prompted frames, make them conditioning."""
        for frame, out in sorted(self.pending.items()):
            if out.maskmem_features is None:
                self.encode_memory(frame, out, from_clicks=True)
            self.cond[frame] = out
            self.non_cond.pop(frame, None)
        self.pending.clear()
        if not self.cond:
            raise ValueError("add a click before propagating")

    def propagate(
        self, start: int | None = None, reverse: bool = False
    ) -> Iterator[tuple[int, FrameOutput]]:
        """``VideoPredictor.propagate`` for one object (whole clip from ``start``)."""
        self._preflight()
        if start is None:
            start = min(self.cond)
        order = range(start, -1, -1) if reverse else range(start, self.num_frames)
        m = int(self.c.mask_size)
        for frame in order:
            out = self.cond.get(frame)
            if out is None:
                self.features(frame)  # timed on its own; per_frame is everything after it
                t0 = time.perf_counter()
                embed = self.condition(frame, False, reverse)
                pad_coords = np.zeros((1, 1, 2), np.float32)
                pad_labels = np.full((1, 1), LABEL_PAD, np.int32)
                low, ptr, score = self.sam_head(
                    embed, self.features(frame), pad_coords, pad_labels, None
                )
                out = FrameOutput(low, ptr, score)
                self.encode_memory(frame, out, from_clicks=False)
                self.non_cond[frame] = out
                self.timings.per_frame.append((time.perf_counter() - t0) * 1e3)
            assert out.pred_masks.shape[-1] == m
            self.tracked[frame] = reverse
            yield frame, out

    def video_res(self, low_res: np.ndarray) -> np.ndarray:
        """Low-res logits -> video-resolution logits (threshold at 0 for display)."""
        return upsample_bilinear(low_res, self.video_hw)


def select_closest_cond_frames(
    frame: int, cond: dict[int, FrameOutput], max_cond_frames: int
) -> tuple[dict[int, FrameOutput], dict[int, FrameOutput]]:
    """``model.select_closest_cond_frames`` (all frames selected for the released -1)."""
    if max_cond_frames == -1 or len(cond) <= max_cond_frames:
        return dict(cond), {}
    selected: dict[int, FrameOutput] = {}
    before = max((t for t in cond if t < frame), default=None)
    if before is not None:
        selected[before] = cond[before]
    after = min((t for t in cond if t >= frame), default=None)
    if after is not None:
        selected[after] = cond[after]
    rest = sorted((t for t in cond if t not in selected), key=lambda t: abs(t - frame))
    for t in rest[: max_cond_frames - len(selected)]:
        selected[t] = cond[t]
    return selected, {t: v for t, v in cond.items() if t not in selected}


def track(
    graphs: Graphs,
    consts: Constants,
    frames: np.ndarray,
    video_hw: tuple[int, int],
    click_frame: int,
    points: Sequence[Sequence[float]],
    labels: Sequence[int],
    reverse: bool,
    bf16_memory: bool = True,
    encoder_cache: dict[int, Features] | None = None,
) -> tuple[dict[int, FrameOutput], Timings]:
    """Click once on ``click_frame`` and propagate through the clip in one direction."""
    tracker = OrtVideoTracker(graphs, consts, frames, video_hw, bf16_memory, encoder_cache)
    tracker.add_click(click_frame, points, labels)
    outputs = dict(tracker.propagate(reverse=reverse))
    return outputs, tracker.timings


# --------------------------------------------------------------------------------------
# References and comparison (torch is used here only, never inside the loop)
# --------------------------------------------------------------------------------------


@dataclass
class Reference:
    """Per-frame outputs of a reference run: low-res logits, scores, pointers."""

    pred_masks: dict[int, np.ndarray]
    object_score_logits: dict[int, np.ndarray]
    obj_ptr: dict[int, np.ndarray]
    input_frames: np.ndarray | None = None


def dump_reference(tag: str) -> Reference:
    """The official package's run from ``reference_dumps/tiny/video_bedroom_{tag}.pt``."""
    import torch

    dump = torch.load(
        DUMPS / "tiny" / f"video_bedroom_{tag}.pt", map_location="cpu", weights_only=False
    )
    frames = dump["frames"]
    return Reference(
        {f: v["pred_masks"].numpy() for f, v in frames.items()},
        {f: v["object_score_logits"].numpy() for f, v in frames.items()},
        {f: v["obj_ptr"].numpy() for f, v in frames.items()},
        dump["input_frames"].numpy(),
    )


def torch_reference(
    size: int,
    frames: Sequence[np.ndarray],
    click_frame: int,
    points: Sequence[Sequence[float]],
    labels: Sequence[int],
    reverse: bool,
    model: Any = None,
) -> Reference:
    """This repo's torch ``VideoPredictor`` (exact parity with the official package at
    1024) run at ``size`` on HWC uint8 ``frames``."""
    import dataclasses

    import torch

    from sam2_scratch.configs import TINY
    from sam2_scratch.model import SAM2
    from sam2_scratch.video_predictor import VideoPredictor

    if model is None:
        model = SAM2.from_checkpoint(CHECKPOINT, dataclasses.replace(TINY, image_size=size))
    predictor = VideoPredictor(model)
    with torch.no_grad():
        state = predictor.init_state(list(frames))
        predictor.add_points(state, click_frame, 1, points, labels)
        ref = Reference({}, {}, {})
        for frame, per_obj, _ in predictor.propagate(state, reverse=reverse):
            out = state.objects[1].output
            fo = out["non_cond_frame_outputs"].get(frame) or out["cond_frame_outputs"][frame]
            ref.pred_masks[frame] = per_obj[1].numpy()
            ref.object_score_logits[frame] = fo.object_score_logits.numpy()
            ref.obj_ptr[frame] = fo.obj_ptr.numpy()
    return ref


def compare(outputs: dict[int, FrameOutput], ref: Reference) -> dict[str, Any]:
    """Per-frame low-res IoU, max abs logit / score / pointer error against ``ref``."""
    if sorted(outputs) != sorted(ref.pred_masks):
        raise ValueError("the loop and the reference visited different frames")
    per_frame = {}
    for f in sorted(outputs):
        o = outputs[f]
        per_frame[f] = {
            "iou": mask_iou(o.pred_masks, ref.pred_masks[f]),
            "logit_err": float(np.abs(o.pred_masks - ref.pred_masks[f]).max()),
            "score": float(o.object_score_logits[0, 0]),
            "score_err": float(np.abs(o.object_score_logits - ref.object_score_logits[f]).max()),
            "ptr_err": float(np.abs(o.obj_ptr - ref.obj_ptr[f]).max()),
        }
    ious = {f: v["iou"] for f, v in per_frame.items()}
    worst = min(ious, key=lambda f: ious[f])
    return {
        "frames": len(per_frame),
        "iou_min": min(ious.values()),
        "iou_mean": float(np.mean(list(ious.values()))),
        "worst_frame": worst,
        "logit_err_max": max(v["logit_err"] for v in per_frame.values()),
        "score_err_max": max(v["score_err"] for v in per_frame.values()),
        "ptr_err_max": max(v["ptr_err"] for v in per_frame.values()),
        "per_frame": per_frame,
    }


def raw_frames(folder: Path = VIDEO_DIR, max_frames: int | None = None) -> list[np.ndarray]:
    paths = sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in FRAME_SUFFIXES),
        key=lambda p: int(p.stem),
    )
    return [np.array(Image.open(p).convert("RGB")) for p in paths[:max_frames]]


def evaluate(
    size: int,
    precisions: dict[str, str],
    exports: Path = EXPORTS,
    max_frames: int | None = None,
    bf16_memory: bool = True,
    runs: Sequence[tuple[str, int, list[list[float]], list[int], bool]] = RUNS,
    references: dict[str, Reference] | None = None,
) -> dict[str, Any]:
    """Run every prompt of ``runs`` through the ORT loop and compare with the reference
    (the dump at 1024 on the full clip, else the torch predictor at ``size``)."""
    graphs = Graphs.load(exports, size, precisions)
    consts = Constants.load(exports / f"constants_{size}.json")
    clip = raw_frames(VIDEO_DIR, max_frames)
    frames, hw = load_video_frames(
        list(clip), size, consts.pixel_mean, consts.pixel_std, max_frames
    )
    use_dump = size == 1024 and max_frames is None
    cache: dict[int, Features] = {}
    result: dict[str, Any] = {
        "size": size,
        "precisions": precisions,
        "files": {k: v.name for k, v in graphs.files.items()},
        "bf16_memory": bf16_memory,
        "num_frames": int(frames.shape[0]),
        "reference": "official sam2 dump" if use_dump else f"torch VideoPredictor at {size}",
        "runs": {},
    }
    references = references if references is not None else {}
    model = None
    for tag, frame, points, labels, reverse in runs:
        if frame >= frames.shape[0]:
            continue
        outputs, timings = track(
            graphs, consts, frames, hw, frame, points, labels, reverse, bf16_memory, cache
        )
        if tag not in references:
            if use_dump:
                references[tag] = dump_reference(tag)
            else:
                if model is None:
                    import dataclasses

                    from sam2_scratch.configs import TINY
                    from sam2_scratch.model import SAM2

                    model = SAM2.from_checkpoint(
                        CHECKPOINT, dataclasses.replace(TINY, image_size=size)
                    )
                references[tag] = torch_reference(size, clip, frame, points, labels, reverse, model)
        ref = references[tag]
        entry = compare(outputs, ref)
        if ref.input_frames is not None:
            entry["frame_load_err"] = float(np.abs(frames - ref.input_frames).max())
        entry["timings"] = timings.summary()
        result["runs"][tag] = entry
        enc = entry["timings"].get("image_encoder")
        if enc is not None and "image_encoder_ms" not in result:
            # Later runs reuse the cached encoder outputs, so record the cost once.
            result["image_encoder_ms"] = enc["median_ms"]
    return result


def _print(result: dict[str, Any]) -> None:
    print(
        f"S={result['size']} {result['precisions']} bf16={result['bf16_memory']} "
        f"vs {result['reference']}"
    )
    for tag, r in result["runs"].items():
        print(
            f"  {tag}: {r['frames']} frames, IoU min {r['iou_min']:.4f} (frame "
            f"{r['worst_frame']}) mean {r['iou_mean']:.4f}, max|logit err| "
            f"{r['logit_err_max']:.3g}, max|score err| {r['score_err_max']:.3g}, "
            f"max|ptr err| {r['ptr_err_max']:.3g}"
        )
        t = r["timings"]
        parts = [f"{g} {v['median_ms']:.1f}" for g, v in t.items() if g != "tracked_frame_total"]
        total = t.get("tracked_frame_total", {}).get("median_ms")
        print(f"    median ms: {', '.join(parts)}; tracked frame (excl. encoder) {total}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--size", type=int, choices=(1024, 512), default=1024)
    parser.add_argument("--precision", choices=sorted(PRECISION_SETS), default="fp32")
    for role in PRECISION_SETS["fp32"]:
        parser.add_argument(f"--{role.replace('_', '-')}", choices=("fp32", "fp16", "int8"))
    parser.add_argument("--max-frames", type=int, help="track only the first N frames")
    parser.add_argument("--no-bf16", action="store_true", help="keep fp32 memory features")
    parser.add_argument("--exports", type=Path, default=EXPORTS)
    parser.add_argument("--json", type=Path, help="write the summary here")
    args = parser.parse_args(argv)
    precisions = dict(PRECISION_SETS[args.precision])
    for role in precisions:
        override = getattr(args, role)
        if override:
            precisions[role] = override
    result = evaluate(
        args.size, precisions, args.exports, args.max_frames, bf16_memory=not args.no_bf16
    )
    _print(result)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
