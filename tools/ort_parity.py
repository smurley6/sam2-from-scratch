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

Usage (from the repo root, venv active)::

    python tools/ort_parity.py --model tiny
    python tools/ort_parity.py --model tiny --sizes 512 --images truck --json parity.json
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
    iou_min: float
    iou_mean: float
    flips: int
    cases: int
    latency_ms: float | None
    passed: bool | None
    criterion: str


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


def format_table(rows: Sequence[Row]) -> str:
    header = (
        f"{'graph':<24} {'prec':<9} {'MB':>6} {'max|err|':>9} {'IoU min':>8} {'IoU mean':>8}"
        f" {'flips':>7} {'ORT ms':>7}  result"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        err = f"{r.max_abs:.2e}" if r.max_abs is not None else "-"
        lat = f"{r.latency_ms:.1f}" if r.latency_ms is not None else "-"
        verdict = "PASS" if r.passed else "FAIL"
        lines.append(
            f"{r.graph:<24} {r.precision:<9} {r.file_mb:>6.1f} {err:>9} {r.iou_min:>8.4f}"
            f" {r.iou_mean:>8.4f} {f'{r.flips}/{r.cases}':>7} {lat:>7}  {verdict} ({r.criterion})"
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
    args = parser.parse_args(argv)

    sam = load_torch(args.model)
    rows: list[Row] = []
    for size in args.sizes:
        rows += evaluate(args.model, size, args.exports, args.images, sam, args.latency_runs)
    print(f"ORT {ort.__version__} CPU provider, model {args.model}, images {args.images}")
    print(format_table(rows))
    if args.json:
        payload: dict[str, Any] = {"model": args.model, "rows": [asdict(r) for r in rows]}
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if all(r.passed for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
