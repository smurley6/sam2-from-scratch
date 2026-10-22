"""Dump intermediate tensors from the official ``sam2`` package for parity tests.

Everything runs on CPU in float32 so the dumps are deterministic. The reimplementation
in ``src/sam2_scratch`` never imports ``sam2``; the tests load these files instead.

Usage (from the repo root, venv active)::

    python tools/dump_reference.py --model tiny
    python tools/dump_reference.py --model small --skip-video

Output layout (``reference_dumps/<model>/``)::

    manifest.json              model config summary, prompt definitions, file list
    image_<name>.pt            per-image tensors (see ``dump_image``)
    video_bedroom_fwd.pt       forward propagation from a click on frame 0
    video_bedroom_rev.pt       reverse propagation from a click on frame 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "checkpoints/sam2.1_hiera_small.pt"),
}

# Prompts in original-image pixel coordinates (x, y). Labels: 1 positive, 0 negative.
IMAGE_PROMPTS: dict[str, list[dict[str, Any]]] = {
    "truck": [
        {"name": "one_point", "points": [[500, 375]], "labels": [1]},
        {"name": "two_points", "points": [[500, 375], [1125, 625]], "labels": [1, 1]},
        {"name": "pos_neg", "points": [[500, 375], [1125, 625]], "labels": [1, 0]},
        {"name": "box", "box": [425, 600, 700, 875]},
        {"name": "box_and_neg", "box": [425, 600, 700, 875], "points": [[575, 750]], "labels": [0]},
    ],
    "cars": [
        {"name": "one_point", "points": [[1100, 900]], "labels": [1]},
        {"name": "box", "box": [600, 500, 1700, 1300]},
    ],
    "groceries": [
        {"name": "one_point", "points": [[400, 300]], "labels": [1]},
        {"name": "two_points", "points": [[400, 300], [150, 150]], "labels": [1, 1]},
    ],
}

VIDEO_DIR = ROOT / "data" / "videos" / "bedroom"
# Child's face region in bedroom frame 0 (960x540 frames).
VIDEO_FWD_PROMPT = {"frame": 0, "points": [[210, 350]], "labels": [1]}
VIDEO_REV_PROMPT = {"frame": 30, "points": [[210, 350]], "labels": [1]}
HOOKED_FRAMES = (1, 2, 7, 20, 47)


def _clone(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().clone().float() if x.is_floating_point() else x.detach().clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_clone(v) for v in x)
    if isinstance(x, dict):
        return {k: _clone(v) for k, v in x.items()}
    return x


class Recorder:
    """Records (args, kwargs, output) for each forward call of selected modules."""

    def __init__(self, model: torch.nn.Module, names: dict[str, torch.nn.Module]):
        self.calls: dict[str, list[dict[str, Any]]] = {n: [] for n in names}
        self.handles = [
            mod.register_forward_hook(self._hook(name), with_kwargs=True)
            for name, mod in names.items()
        ]

    def _hook(self, name: str):  # type: ignore[no-untyped-def]
        def fn(_mod, args, kwargs, output):  # type: ignore[no-untyped-def]
            self.calls[name].append(
                {"args": _clone(args), "kwargs": _clone(kwargs), "output": _clone(output)}
            )

        return fn

    def reset(self) -> None:
        for v in self.calls.values():
            v.clear()

    def close(self) -> None:
        for h in self.handles:
            h.remove()


def build(model_name: str) -> Any:
    from sam2.build_sam import build_sam2

    cfg, ckpt = CONFIGS[model_name]
    model = build_sam2(cfg, str(ROOT / ckpt), device="cpu", mode="eval")
    model.eval()
    return model


def hook_targets(model: Any) -> dict[str, torch.nn.Module]:
    return {
        "trunk": model.image_encoder.trunk,
        "neck": model.image_encoder.neck,
        "image_encoder": model.image_encoder,
        "prompt_encoder": model.sam_prompt_encoder,
        "mask_decoder": model.sam_mask_decoder,
        "memory_encoder": model.memory_encoder,
        "memory_attention": model.memory_attention,
    }


@torch.no_grad()
def dump_image(model: Any, name: str, path: Path, out_dir: Path) -> dict[str, Any]:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor(model)
    rec = Recorder(model, hook_targets(model))
    image = np.array(Image.open(path).convert("RGB"))
    predictor.set_image(image)
    result: dict[str, Any] = {
        "orig_hw": list(image.shape[:2]),
        "input_image": predictor._transforms(image)[None].float().clone(),
        "trunk": rec.calls["trunk"][0],
        "neck": rec.calls["neck"][0],
        "image_embed": predictor._features["image_embed"].clone(),
        "high_res_feats": [f.clone() for f in predictor._features["high_res_feats"]],
        "prompts": {},
    }
    for prompt in IMAGE_PROMPTS[name]:
        rec.reset()
        pts = np.array(prompt["points"], dtype=np.float32) if "points" in prompt else None
        lbl = np.array(prompt["labels"], dtype=np.int32) if "labels" in prompt else None
        box = np.array(prompt["box"], dtype=np.float32) if "box" in prompt else None
        for multimask in (True, False):
            masks, ious, low_res = predictor.predict(
                point_coords=pts,
                point_labels=lbl,
                box=box,
                multimask_output=multimask,
                return_logits=True,
            )
            key = f"{prompt['name']}/{'multi' if multimask else 'single'}"
            # Full-resolution masks are reproducible from low_res_masks + orig_hw, and the
            # decoder inputs from image_embed/high_res_feats + the prompt encoder outputs,
            # so only the small pieces are kept (the full dump was ~1 GB per image).
            result["prompts"][key] = {
                "masks_hw": list(masks.shape[-2:]),
                "iou_predictions": torch.from_numpy(ious).float(),
                "low_res_masks": torch.from_numpy(low_res).float(),
                "prompt_encoder": rec.calls["prompt_encoder"][-1],
                "mask_decoder_output": rec.calls["mask_decoder"][-1]["output"],
                "mask_decoder_multimask": multimask,
            }
    rec.close()
    torch.save(result, out_dir / f"image_{name}.pt")
    return {
        "file": f"image_{name}.pt",
        "orig_hw": result["orig_hw"],
        "prompts": IMAGE_PROMPTS[name],
    }


@torch.no_grad()
def dump_video(
    model_name: str, prompt: dict[str, Any], reverse: bool, out_dir: Path, tag: str
) -> dict[str, Any]:
    from sam2.build_sam import build_sam2_video_predictor

    cfg, ckpt = CONFIGS[model_name]
    predictor = build_sam2_video_predictor(cfg, str(ROOT / ckpt), device="cpu", mode="eval")
    predictor.eval()
    rec = Recorder(predictor, hook_targets(predictor))
    state = predictor.init_state(video_path=str(VIDEO_DIR), async_loading_frames=False)
    predictor.add_new_points_or_box(
        state,
        frame_idx=prompt["frame"],
        obj_id=1,
        points=np.array(prompt["points"], dtype=np.float32),
        labels=np.array(prompt["labels"], dtype=np.int32),
    )
    # Prompt outputs live in a temp dict until propagation consolidates them (the
    # consolidation also runs the memory encoder on the conditioning frame).
    predictor.propagate_in_video_preflight(state)
    obj_out = state["output_dict_per_obj"][0]
    cond_out = obj_out["cond_frame_outputs"][prompt["frame"]]
    frames: dict[int, dict[str, Any]] = {}
    # One memory-attention / memory-encoder call per tracked frame: index them by frame.
    mem_attn_calls: dict[int, Any] = {}
    mem_enc_calls: dict[int, Any] = {}
    rec.reset()
    for frame_idx, _obj_ids, video_res_masks in predictor.propagate_in_video(
        state, start_frame_idx=prompt["frame"], reverse=reverse
    ):
        out = obj_out["non_cond_frame_outputs"].get(frame_idx) or cond_out
        frames[frame_idx] = {
            "video_res_mask": video_res_masks.clone(),
            "pred_masks": out["pred_masks"].clone(),
            "obj_ptr": out["obj_ptr"].clone(),
            "object_score_logits": out["object_score_logits"].clone(),
            "maskmem_features": out["maskmem_features"].clone(),
            "maskmem_pos_enc": [p.clone() for p in out["maskmem_pos_enc"]],
        }
        if frame_idx in HOOKED_FRAMES and rec.calls["memory_attention"]:
            mem_attn_calls[frame_idx] = rec.calls["memory_attention"][-1]
        if frame_idx in HOOKED_FRAMES and rec.calls["memory_encoder"]:
            mem_enc_calls[frame_idx] = rec.calls["memory_encoder"][-1]
        rec.reset()
    rec.close()
    result = {
        "prompt": prompt,
        "reverse": reverse,
        "num_frames": state["num_frames"],
        "video_hw": [state["video_height"], state["video_width"]],
        "input_frames": state["images"].float().clone(),  # [T, 3, 1024, 1024] normalized
        "cond_frame": {
            "pred_masks": cond_out["pred_masks"].clone(),
            "obj_ptr": cond_out["obj_ptr"].clone(),
            "object_score_logits": cond_out["object_score_logits"].clone(),
            "maskmem_features": cond_out["maskmem_features"].clone(),
            "maskmem_pos_enc": [p.clone() for p in cond_out["maskmem_pos_enc"]],
        },
        "frames": frames,
        "memory_attention_calls": mem_attn_calls,
        "memory_encoder_calls": mem_enc_calls,
    }
    torch.save(result, out_dir / f"video_bedroom_{tag}.pt")
    return {"file": f"video_bedroom_{tag}.pt", "prompt": prompt, "reverse": reverse}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=CONFIGS, default="tiny")
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "reference_dumps" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build(args.model)
    manifest: dict[str, Any] = {
        "model": args.model,
        "config": CONFIGS[args.model][0],
        "checkpoint": CONFIGS[args.model][1],
        "image_size": model.image_size,
        "num_maskmem": model.num_maskmem,
        "max_obj_ptrs_in_encoder": model.max_obj_ptrs_in_encoder,
        "mem_dim": model.mem_dim,
        "hidden_dim": model.hidden_dim,
        "images": {},
        "videos": {},
    }
    for name in IMAGE_PROMPTS:
        print(f"[{args.model}] image {name}")
        manifest["images"][name] = dump_image(
            model, name, ROOT / "data" / "images" / f"{name}.jpg", out_dir
        )
    if not args.skip_video:
        print(f"[{args.model}] video forward")
        manifest["videos"]["fwd"] = dump_video(args.model, VIDEO_FWD_PROMPT, False, out_dir, "fwd")
        print(f"[{args.model}] video reverse")
        manifest["videos"]["rev"] = dump_video(args.model, VIDEO_REV_PROMPT, True, out_dir, "rev")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
