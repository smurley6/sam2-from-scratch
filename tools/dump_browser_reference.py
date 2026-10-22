"""Ground truth for the TypeScript port (``web/``): outputs of the pure-onnxruntime loop
on the demo sample media with the shipped 512 graph set (int8 encoder, fp32 everything
else), so the Node parity test can compare against a loop that is itself verified
against the torch model.

Writes ``exports/browser_reference/``::

    clip_blackswan/frames_u8.bin      [T, 512, 512, 3] uint8, PIL-bicubic resized frames
    clip_blackswan/masks.bin          [T, 128, 128] float32 low-res logits, forward run
    clip_blackswan/meta.json          click, scores, obj_ptr norms, shapes
    image_truck/image_u8.bin          [512, 512, 3] uint8
    image_truck/prompts.json          prompts and head outputs (masks in masks.bin)

Usage: ``PYTHONPATH=src python tools/dump_browser_reference.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from ort_video_loop import Constants, Graphs, OrtVideoTracker, run  # noqa: E402

SIZE = 512
PRECISIONS = {
    "image_encoder": "int8",
    "sam_head": "fp32",
    "memory_encoder": "fp32",
    "memory_attention": "fp32",
}
OUT = ROOT / "exports" / "browser_reference"
CLIP = ROOT / "bundle" / "clips" / "blackswan"
CLICK = {"frame": 0, "points": [[330, 190]], "labels": [1]}  # on the swan, 640x360 pixels
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


def dump_clip(graphs: Graphs, consts: Constants) -> None:
    out = OUT / "clip_blackswan"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(CLIP.glob("*.jpg"))
    u8 = np.stack([resize_u8(Image.open(f)) for f in files])
    w, h = Image.open(files[0]).size
    frames = np.stack([normalise(f, consts) for f in u8])
    tracker = OrtVideoTracker(graphs, consts, frames, (h, w))
    tracker.add_click(CLICK["frame"], CLICK["points"], CLICK["labels"])
    masks, scores, ptr_norm = [], [], []
    for _frame, fo in tracker.propagate():
        masks.append(fo.pred_masks[0, 0])
        scores.append(float(fo.object_score_logits[0, 0]))
        ptr_norm.append(float(np.linalg.norm(fo.obj_ptr)))
    masks_arr = np.stack(masks).astype(np.float32)
    u8.tofile(out / "frames_u8.bin")
    masks_arr.tofile(out / "masks.bin")
    area = [float((m > 0).mean()) for m in masks_arr]
    meta = {
        "size": SIZE,
        "video_hw": [h, w],
        "num_frames": len(files),
        "click": CLICK,
        "frames_u8_shape": list(u8.shape),
        "masks_shape": list(masks_arr.shape),
        "object_score_logits": scores,
        "obj_ptr_norm": ptr_norm,
        "mask_area_fraction": area,
        "precisions": PRECISIONS,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"clip: frames {len(files)} area frame0 {area[0]:.3f} min {min(area):.3f}")


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
    graphs = Graphs.load(ROOT / "exports", SIZE, PRECISIONS)
    consts = Constants.load(ROOT / "exports" / f"constants_{SIZE}.json")
    dump_clip(graphs, consts)
    dump_image(graphs, consts)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
