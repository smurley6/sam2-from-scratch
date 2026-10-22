"""Semi-supervised VOS evaluation on DAVIS 2017 val with the official evaluator.

Protocol (the one SAM 2 reports): every object is prompted with its ground-truth mask on
the first frame only, tracked forward independently, and the per-pixel winner across
objects is written as a palette PNG. ``davis2017-evaluation`` then computes J and F.

Usage::

    python tools/eval_davis.py --model tiny --device mps
    python tools/eval_davis.py --model tiny --num-maskmem 1 --tag mem1
    python tools/eval_davis.py --model tiny --no-obj-ptrs --tag noptr
    python tools/eval_davis.py --model tiny --image-size 512 --tag 512

Results: ``results/davis/<model>_<tag>/`` (PNGs, gitignored) and
``docs/results/davis_<model>_<tag>.json`` (tracked summary).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sam2_scratch import configs
from sam2_scratch.model import SAM2
from sam2_scratch.video_predictor import VideoPredictor, non_overlapping

ROOT = Path(__file__).resolve().parent.parent
DAVIS = ROOT / "data" / "davis" / "DAVIS"
CKPT = {
    "tiny": ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt",
    "small": ROOT / "checkpoints" / "sam2.1_hiera_small.pt",
}


def load_palette_mask(path: Path) -> tuple[np.ndarray, list[int]]:
    im = Image.open(path)
    assert im.mode == "P", f"{path} is not a palette PNG"
    arr = np.array(im)
    return arr, im.getpalette() or []


def run_sequence(
    predictor: VideoPredictor, seq: str, out_dir: Path, device: torch.device
) -> tuple[int, int, float]:
    frames_dir = DAVIS / "JPEGImages" / "480p" / seq
    first_mask, palette = load_palette_mask(DAVIS / "Annotations" / "480p" / seq / "00000.png")
    obj_ids = [int(i) for i in np.unique(first_mask) if i != 0]
    out_dir.mkdir(parents=True, exist_ok=True)

    state = predictor.init_state(frames_dir)
    for obj_id in obj_ids:
        predictor.add_mask(state, 0, obj_id, first_mask == obj_id)

    t0 = time.perf_counter()
    n_frames = 0
    for frame_idx, _per_obj, video_res_logits in predictor.propagate(state):
        # video_res_logits: [num_obj, 1, H, W] in obj_ids order
        logits = non_overlapping(video_res_logits)[:, 0]  # [num_obj, H, W]
        best = torch.argmax(logits, dim=0)
        fg = logits.max(dim=0).values > 0
        label = torch.zeros_like(best, dtype=torch.uint8)
        ids = torch.tensor(obj_ids, dtype=torch.uint8, device=best.device)
        label[fg] = ids[best[fg]]
        im = Image.fromarray(label.cpu().numpy().astype(np.uint8), mode="P")
        im.putpalette(palette)
        im.save(out_dir / f"{frame_idx:05d}.png")
        n_frames += 1
    return n_frames, len(obj_ids), time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=CKPT, default="tiny")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--num-maskmem", type=int, default=None)
    ap.add_argument("--no-obj-ptrs", action="store_true")
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--sequences", nargs="*", default=None, help="subset for smoke runs")
    ap.add_argument("--skip-inference", action="store_true", help="only re-run the evaluator")
    args = ap.parse_args()

    cfg = configs.CONFIGS[args.model]
    overrides: dict[str, Any] = {}
    if args.num_maskmem is not None:
        overrides["num_maskmem"] = args.num_maskmem
    if args.no_obj_ptrs:
        overrides["use_obj_ptrs_in_encoder"] = False
    if args.image_size is not None:
        overrides["image_size"] = args.image_size
    cfg = dataclasses.replace(cfg, **overrides)

    res_dir = ROOT / "results" / "davis" / f"{args.model}_{args.tag}"
    seqs = args.sequences or (DAVIS / "ImageSets" / "2017" / "val.txt").read_text().split()

    timing: dict[str, Any] = {}
    if not args.skip_inference:
        device = torch.device(args.device)
        model = SAM2.from_checkpoint(CKPT[args.model], cfg).to(device).eval()
        predictor = VideoPredictor(model)
        total_frames = total_obj_frames = 0
        total_time = 0.0
        with torch.inference_mode():
            for i, seq in enumerate(seqs):
                n, k, dt = run_sequence(predictor, seq, res_dir / seq, device)
                total_frames += n
                total_obj_frames += n * k
                total_time += dt
                print(f"[{i + 1}/{len(seqs)}] {seq}: {n} frames x {k} objects in {dt:.1f}s")
        timing = {
            "frames": total_frames,
            "object_frames": total_obj_frames,
            "seconds": round(total_time, 1),
            "object_frames_per_second": round(total_obj_frames / max(total_time, 1e-9), 2),
            "device": args.device,
        }

    from davis2017.evaluation import DAVISEvaluation

    ev = DAVISEvaluation(
        davis_root=str(DAVIS),
        task="semi-supervised",
        gt_set="val",
        sequences=args.sequences or "all",
    )
    metrics = ev.evaluate(str(res_dir))
    j_mean = float(np.mean(metrics["J"]["M"]))
    f_mean = float(np.mean(metrics["F"]["M"]))
    summary = {
        "model": args.model,
        "tag": args.tag,
        "config_overrides": overrides,
        "J&F": round((j_mean + f_mean) / 2 * 100, 2),
        "J": round(j_mean * 100, 2),
        "F": round(f_mean * 100, 2),
        "J_recall": round(float(np.mean(metrics["J"]["R"])) * 100, 2),
        "F_recall": round(float(np.mean(metrics["F"]["R"])) * 100, 2),
        "per_object": {
            seq: {"J": round(float(j) * 100, 2), "F": round(float(f) * 100, 2)}
            for seq, j, f in zip(
                metrics["J"]["M_per_object"].keys(),
                metrics["J"]["M_per_object"].values(),
                metrics["F"]["M_per_object"].values(),
                strict=True,
            )
        },
        "timing": timing,
    }
    out = ROOT / "docs" / "results" / f"davis_{args.model}_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_object"}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
