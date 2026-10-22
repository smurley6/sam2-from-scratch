"""End-to-end parity of the image predictor from dumped features and manifest prompts.

Prompts are given in original-image pixel coordinates exactly as in the manifest, so the
coordinate rescaling, box-to-points conversion and padding are all exercised.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from sam2_scratch.image_predictor import ImagePredictor
from sam2_scratch.mask_decoder import MaskDecoder
from sam2_scratch.prompt_encoder import PromptEncoder
from sam2_scratch.weights import load_checkpoint, load_prefix
from tests.conftest import CHECKPOINTS, DUMPS, ROOT, _require, max_abs

IMAGES = ("truck", "cars", "groceries")
PRED_TOL = 1e-3
PREPROCESS_TOL = 1e-4


@pytest.fixture(scope="module", params=["tiny", "small"])
def setup(request: pytest.FixtureRequest) -> tuple[ImagePredictor, dict[str, Any]]:
    model = str(request.param)
    manifest = json.loads(_require(DUMPS / model / "manifest.json").read_text())
    sd = load_checkpoint(_require(CHECKPOINTS[model]))
    pe = PromptEncoder()
    load_prefix(pe, sd, "sam_prompt_encoder")
    md = MaskDecoder()
    load_prefix(md, sd, "sam_mask_decoder")
    return ImagePredictor(pe, md), manifest


@pytest.fixture(scope="module", params=IMAGES)
def case(
    request: pytest.FixtureRequest, setup: tuple[ImagePredictor, dict[str, Any]]
) -> tuple[ImagePredictor, str, dict[str, Any], dict[str, Any]]:
    predictor, manifest = setup
    name = str(request.param)
    info = manifest["images"][name]
    dump = torch.load(
        _require(DUMPS / manifest["model"] / info["file"]), map_location="cpu", weights_only=False
    )
    return predictor, name, info, dump


def _prompt_arrays(
    prompt: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    pts = np.array(prompt["points"], dtype=np.float32) if "points" in prompt else None
    lbl = np.array(prompt["labels"], dtype=np.int32) if "labels" in prompt else None
    box = np.array(prompt["box"], dtype=np.float32) if "box" in prompt else None
    return pts, lbl, box


def test_preprocess_matches_reference(
    case: tuple[ImagePredictor, str, dict[str, Any], dict[str, Any]],
) -> None:
    predictor, name, _, dump = case
    image = np.array(Image.open(_require(ROOT / "data" / "images" / f"{name}.jpg")).convert("RGB"))
    assert list(image.shape[:2]) == dump["orig_hw"]
    assert max_abs(predictor.preprocess(image), dump["input_image"]) < PREPROCESS_TOL


def test_predict_matches_reference(
    case: tuple[ImagePredictor, str, dict[str, Any], dict[str, Any]],
) -> None:
    predictor, _, info, dump = case
    predictor.set_features(dump["image_embed"], dump["high_res_feats"], tuple(dump["orig_hw"]))
    for prompt in info["prompts"]:
        pts, lbl, box = _prompt_arrays(prompt)
        for multimask in (True, False):
            key = f"{prompt['name']}/{'multi' if multimask else 'single'}"
            ref = dump["prompts"][key]
            masks, ious, low_res = predictor.predict(
                pts, lbl, box, multimask_output=multimask, return_logits=True
            )
            n = 3 if multimask else 1
            assert masks.shape == (n, *ref["masks_hw"]), key
            assert max_abs(torch.from_numpy(low_res), ref["low_res_masks"]) < PRED_TOL, key
            assert max_abs(torch.from_numpy(ious), ref["iou_predictions"]) < PRED_TOL, key

            binary, _, _ = predictor.predict(pts, lbl, box, multimask_output=multimask)
            assert binary.dtype == np.bool_
            np.testing.assert_array_equal(binary, masks > 0.0)


def test_set_image_uses_injected_encoder(
    case: tuple[ImagePredictor, str, dict[str, Any], dict[str, Any]],
) -> None:
    predictor, _, info, dump = case
    seen: list[torch.Tensor] = []

    def fake_encoder(x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        seen.append(x)
        return dump["image_embed"], list(dump["high_res_feats"])

    h, w = dump["orig_hw"]
    predictor.set_image(np.zeros((h, w, 3), dtype=np.uint8), fake_encoder)
    assert seen and seen[0].shape == (1, 3, 1024, 1024)
    pts, lbl, box = _prompt_arrays(info["prompts"][0])
    _, ious, _ = predictor.predict(pts, lbl, box, multimask_output=True)
    ref = dump["prompts"][f"{info['prompts'][0]['name']}/multi"]["iou_predictions"]
    assert max_abs(torch.from_numpy(ious), ref) < PRED_TOL


def test_mask_input_round_trip(
    case: tuple[ImagePredictor, str, dict[str, Any], dict[str, Any]],
) -> None:
    """Feeding back the best low-res logits runs the dense-prompt path end to end."""
    predictor, _, info, dump = case
    predictor.set_features(dump["image_embed"], dump["high_res_feats"], tuple(dump["orig_hw"]))
    pts, lbl, box = _prompt_arrays(info["prompts"][0])
    _, ious, low_res = predictor.predict(pts, lbl, box, multimask_output=True)
    best = low_res[int(np.argmax(ious))][None]
    masks, ious2, low_res2 = predictor.predict(
        pts, lbl, box, mask_input=best, multimask_output=False
    )
    assert masks.shape == (1, *dump["orig_hw"])
    assert ious2.shape == (1,) and low_res2.shape == (1, 256, 256)
    assert np.isfinite(low_res2).all() and np.abs(low_res2).max() <= 32.0


def test_predict_requires_features() -> None:
    predictor = ImagePredictor(PromptEncoder(), MaskDecoder())
    with pytest.raises(RuntimeError):
        predictor.predict(np.zeros((1, 2), np.float32), np.ones(1, np.int32))
