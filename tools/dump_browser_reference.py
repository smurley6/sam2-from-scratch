"""Ground truth for the TypeScript port (``web/``): outputs of the pure-onnxruntime loop
on the demo sample media with the shipped 512 graph set (int8 encoder, fp32 everything
else), so the Node parity test can compare against a loop that is itself verified
against the torch model.

Writes ``exports/browser_reference/``::

    clip_blackswan/frames_u8.bin      [T, 512, 512, 3] uint8, PIL-bicubic resized frames
    clip_blackswan/masks.bin          [T, 128, 128] float32 low-res logits, forward run
    clip_blackswan/masks_box.bin      [T, 128, 128] the same from a box on frame 0
    clip_blackswan/masks_mask.bin     [T, 128, 128] the same from a mask prompt on frame 0
    clip_blackswan/mask_prompt_u8.bin [H, W] uint8, that mask prompt (the click run's own
                                      frame-0 prediction at video resolution)
    clip_blackswan/meta.json          click, box, scores, obj_ptr norms, shapes
    clip_bmx-trees/frames_u8.bin      [T, 512, 512, 3] uint8
    clip_bmx-trees/masks.bin          [2, T, 128, 128] two objects (rider, bike) tracked
                                      independently, sharing one encoder pass per frame
    clip_bmx-trees/nonoverlap.bin     [2, T, 128, 128] the same after `non_overlapping`
    clip_bmx-trees/meta.json          clicks per object, scores, shapes
    image_truck/image_u8.bin          [512, 512, 3] uint8
    image_truck/prompts.json          prompts and head outputs (masks in masks.bin)

Usage: ``PYTHONPATH=src python tools/dump_browser_reference.py [--only bmx-trees]``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from ort_video_loop import Constants, Features, Graphs, OrtVideoTracker, run  # noqa: E402

SIZE = 512
PRECISIONS = {
    "image_encoder": "int8",
    "sam_head": "fp32",
    "memory_encoder": "fp32",
    "memory_attention": "fp32",
}
OUT = ROOT / "exports" / "browser_reference"
CLIPS = ROOT / "bundle" / "clips"
# Clicks in 640x360 video pixels: blackswan, one object; bmx-trees, the rider (shirt and
# shorts) and the bike (both wheels), two independent memory banks over one encoder pass.
CLICK = {"frame": 0, "points": [[330, 190]], "labels": [1]}  # on the swan
# The box run draws this margin (video pixels) around the click run's frame-0 mask.
BOX_MARGIN = 8
BMX_CLICKS = [
    {"frame": 0, "points": [[340, 150], [335, 185]], "labels": [1, 1]},
    {"frame": 0, "points": [[290, 235], [395, 235]], "labels": [1, 1]},
]
IMAGE = ROOT / "bundle" / "images" / "truck.jpg"
IMAGE_PROMPTS = [
    {"name": "one_point", "points": [[356, 267]], "labels": [1]},
    {"name": "pos_neg", "points": [[356, 267], [800, 444]], "labels": [1, 0]},
    {"name": "box", "box": [302, 427, 498, 622]},
]


def resize_u8(im: Image.Image) -> np.ndarray:
    return np.array(im.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BICUBIC))


def normalise(u8: np.ndarray, consts: Constants) -> np.ndarray:
    x = (u8 / 255.0 - np.asarray(consts.pixel_mean)) / np.asarray(consts.pixel_std)
    return np.ascontiguousarray(x.transpose(2, 0, 1)).astype(np.float32)


def load_clip(name: str, consts: Constants) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    files = sorted((CLIPS / name).glob("*.jpg"))
    u8 = np.stack([resize_u8(Image.open(f)) for f in files])
    w, h = Image.open(files[0]).size
    frames = np.stack([normalise(f, consts) for f in u8])
    return u8, frames, (h, w)


def run_clip(
    tracker: OrtVideoTracker,
) -> tuple[np.ndarray, list[float], list[float]]:
    """Propagate forward; ``([T, m, m]`` low-res logits, scores, pointer norms)``."""
    masks, scores, ptr_norm = [], [], []
    for _frame, fo in tracker.propagate():
        masks.append(fo.pred_masks[0, 0])
        scores.append(float(fo.object_score_logits[0, 0]))
        ptr_norm.append(float(np.linalg.norm(fo.obj_ptr)))
    return np.stack(masks).astype(np.float32), scores, ptr_norm


def dump_clip(graphs: Graphs, consts: Constants) -> None:
    """Three forward runs on blackswan over one encoder cache: the click, a box drawn
    around the click's frame-0 mask, and that mask itself as a mask prompt."""
    out = OUT / "clip_blackswan"
    out.mkdir(parents=True, exist_ok=True)
    u8, frames, (h, w) = load_clip("blackswan", consts)
    cache: dict[int, Features] = {}
    tracker = OrtVideoTracker(graphs, consts, frames, (h, w), encoder_cache=cache)
    tracker.add_click(CLICK["frame"], CLICK["points"], CLICK["labels"])
    masks_arr, scores, ptr_norm = run_clip(tracker)

    # The click's own frame-0 prediction at video resolution: a box around it, and the
    # mask prompt (what the demo's "use mask as prompt" hands in).
    frame0 = tracker.video_res(masks_arr[CLICK["frame"]][None, None])[0, 0] > 0
    ys, xs = np.nonzero(frame0)
    box = [
        float(max(int(xs.min()) - BOX_MARGIN, 0)),
        float(max(int(ys.min()) - BOX_MARGIN, 0)),
        float(min(int(xs.max()) + BOX_MARGIN, w - 1)),
        float(min(int(ys.max()) + BOX_MARGIN, h - 1)),
    ]
    box_tracker = OrtVideoTracker(graphs, consts, frames, (h, w), encoder_cache=cache)
    box_tracker.add_click(CLICK["frame"], [], [], box=box)
    masks_box, scores_box, _ = run_clip(box_tracker)
    mask_tracker = OrtVideoTracker(graphs, consts, frames, (h, w), encoder_cache=cache)
    mask_tracker.add_mask(CLICK["frame"], frame0)
    masks_mask, scores_mask, _ = run_clip(mask_tracker)

    u8.tofile(out / "frames_u8.bin")
    masks_arr.tofile(out / "masks.bin")
    masks_box.tofile(out / "masks_box.bin")
    masks_mask.tofile(out / "masks_mask.bin")
    frame0.astype(np.uint8).tofile(out / "mask_prompt_u8.bin")
    area = [float((m > 0).mean()) for m in masks_arr]
    meta = {
        "size": SIZE,
        "video_hw": [h, w],
        "num_frames": len(frames),
        "click": CLICK,
        "box": {"frame": CLICK["frame"], "box": box},
        "mask_prompt": {"frame": CLICK["frame"], "shape": [h, w], "file": "mask_prompt_u8.bin"},
        "frames_u8_shape": list(u8.shape),
        "masks_shape": list(masks_arr.shape),
        "object_score_logits": scores,
        "object_score_logits_box": scores_box,
        "object_score_logits_mask": scores_mask,
        "obj_ptr_norm": ptr_norm,
        "mask_area_fraction": area,
        "mask_area_fraction_box": [float((m > 0).mean()) for m in masks_box],
        "mask_area_fraction_mask": [float((m > 0).mean()) for m in masks_mask],
        "precisions": PRECISIONS,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    iou = [
        float(((a > 0) & (b > 0)).sum() / max(1, ((a > 0) | (b > 0)).sum()))
        for a, b in zip(masks_arr, masks_mask, strict=True)
    ]
    print(
        f"clip: frames {len(frames)} area frame0 {area[0]:.3f} min {min(area):.3f}; "
        f"box {box}; mask run vs click run IoU min {min(iou):.3f}"
    )


def dump_multi_clip(graphs: Graphs, consts: Constants) -> None:
    """Two objects on bmx-trees: one ``OrtVideoTracker`` (memory bank) per object, both
    reading the same encoder cache, so every frame is encoded once. ``nonoverlap.bin`` is
    the torch ``non_overlapping`` of the two runs, the display rule the browser mirrors."""
    from sam2_scratch.video_predictor import non_overlapping  # torch: display rule only

    out = OUT / "clip_bmx-trees"
    out.mkdir(parents=True, exist_ok=True)
    u8, frames, (h, w) = load_clip("bmx-trees", consts)
    cache: dict[int, Features] = {}
    masks, scores = [], []
    for click in BMX_CLICKS:
        tracker = OrtVideoTracker(graphs, consts, frames, (h, w), encoder_cache=cache)
        tracker.add_click(click["frame"], click["points"], click["labels"])
        run_masks, run_scores = [], []
        for _frame, fo in tracker.propagate():
            run_masks.append(fo.pred_masks[0, 0])
            run_scores.append(float(fo.object_score_logits[0, 0]))
        masks.append(np.stack(run_masks))
        scores.append(run_scores)
        encodes = len(tracker.timings.per_graph.get("image_encoder", []))
        print(f"bmx-trees object {len(masks)}: encoder calls {encodes}")
    masks_arr = np.stack(masks).astype(np.float32)  # [K, T, m, m]
    # non_overlapping takes one frame's [num_obj, 1, H, W]; applied per frame.
    nonoverlap = np.stack(
        [
            non_overlapping(torch.from_numpy(masks_arr[:, t])[:, None])[:, 0].numpy()
            for t in range(masks_arr.shape[1])
        ],
        axis=1,
    ).astype(np.float32)
    u8.tofile(out / "frames_u8.bin")
    masks_arr.tofile(out / "masks.bin")
    nonoverlap.tofile(out / "nonoverlap.bin")
    area = [[float((m > 0).mean()) for m in obj] for obj in masks_arr]
    overlap = [
        float(
            ((masks_arr[0, t] > 0) & (masks_arr[1, t] > 0)).sum()
            / max(1, ((masks_arr[0, t] > 0) | (masks_arr[1, t] > 0)).sum())
        )
        for t in range(masks_arr.shape[1])
    ]
    meta = {
        "size": SIZE,
        "video_hw": [h, w],
        "num_frames": len(frames),
        "objects": BMX_CLICKS,
        "frames_u8_shape": list(u8.shape),
        "masks_shape": list(masks_arr.shape),
        "object_score_logits": scores,
        "mask_area_fraction": area,
        "pairwise_iou": overlap,
        "precisions": PRECISIONS,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"bmx-trees: frames {len(frames)} areas frame0 {area[0][0]:.3f} / {area[1][0]:.3f} "
        f"min {min(area[0]):.3f} / {min(area[1]):.3f} max pairwise IoU {max(overlap):.3f}"
    )


def dump_image(graphs: Graphs, consts: Constants) -> None:
    out = OUT / "image_truck"
    out.mkdir(parents=True, exist_ok=True)
    im = Image.open(IMAGE)
    w, h = im.size
    u8 = resize_u8(im)
    u8.tofile(out / "image_u8.bin")
    x = normalise(u8, consts)[None]
    enc = run(graphs.image_encoder, {"image": x})
    results = []
    all_masks = []
    for p in IMAGE_PROMPTS:
        pts = [list(map(float, q)) for q in p.get("points", [])]
        lbl = list(p.get("labels", []))
        if "box" in p:
            x0, y0, x1, y1 = p["box"]
            pts = [[x0, y0], [x1, y1], *pts]
            lbl = [2, 3, *lbl]
        coords = np.asarray(pts, np.float32).reshape(1, -1, 2)
        coords[..., 0] *= SIZE / w
        coords[..., 1] *= SIZE / h
        labels = np.asarray(lbl, np.int32).reshape(1, -1)
        m = int(consts.mask_size)
        head = run(
            graphs.sam_head,
            {
                "image_embed": enc["image_embed"],
                "high_res_feat_s0": enc["high_res_feat_s0"],
                "high_res_feat_s1": enc["high_res_feat_s1"],
                "point_coords": coords,
                "point_labels": labels,
                "mask_input": np.zeros((1, 1, m, m), np.float32),
                "has_mask_input": np.zeros((1,), np.float32),
            },
        )
        all_masks.append(head["masks"][0])  # [4, m, m]
        results.append(
            {
                "name": p["name"],
                "point_coords_512": coords[0].tolist(),
                "point_labels": labels[0].tolist(),
                "iou_predictions": head["iou_predictions"][0].tolist(),
                "object_score_logits": float(head["object_score_logits"][0, 0]),
                "mask_area_fraction": [float((mm > 0).mean()) for mm in head["masks"][0]],
            }
        )
    np.stack(all_masks).astype(np.float32).tofile(out / "masks.bin")
    (out / "prompts.json").write_text(
        json.dumps(
            {
                "size": SIZE,
                "image_hw": [h, w],
                "masks_shape": [
                    len(IMAGE_PROMPTS),
                    4,
                    int(consts.mask_size),
                    int(consts.mask_size),
                ],
                "prompts": results,
                "precisions": PRECISIONS,
            },
            indent=2,
        )
    )
    for r in results:
        print("image:", r["name"], "iou", [round(v, 3) for v in r["iou_predictions"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--only",
        choices=["blackswan", "bmx-trees", "image"],
        action="append",
        help="write only these outputs (default: all)",
    )
    args = parser.parse_args()
    graphs = Graphs.load(ROOT / "exports", SIZE, PRECISIONS)
    consts = Constants.load(ROOT / "exports" / f"constants_{SIZE}.json")
    wanted = set(args.only or ["blackswan", "bmx-trees", "image"])
    if "blackswan" in wanted:
        dump_clip(graphs, consts)
    if "bmx-trees" in wanted:
        dump_multi_clip(graphs, consts)
    if "image" in wanted:
        dump_image(graphs, consts)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
