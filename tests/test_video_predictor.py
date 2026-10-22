"""End-to-end parity of the video predictor on the bedroom clip, forward and reverse.

Each run loads the frames itself from ``data/videos/bedroom``, adds the dumped click and
propagates through the clip; every propagated frame is compared with the official
package's outputs. One run per (model, direction) is shared by the module's tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from sam2_scratch.configs import CONFIGS
from sam2_scratch.model import COND, NON_COND, SAM2, FrameOutput
from sam2_scratch.video_predictor import VideoPredictor, load_frames, non_overlapping
from tests.conftest import CHECKPOINTS, DUMPS, ROOT, _require, mask_iou, max_abs

VIDEO = ROOT / "data" / "videos" / "bedroom"
CASES = [(m, tag) for m in CONFIGS for tag in ("fwd", "rev")]
FRAME_TOL = 1e-3
IOU_MIN = 0.98
STATE_TOL = 1e-2

_MODELS: dict[str, SAM2] = {}


def _model(name: str) -> SAM2:
    if name not in _MODELS:
        _MODELS[name] = SAM2.from_checkpoint(_require(CHECKPOINTS[name]), CONFIGS[name])
    return _MODELS[name]


@dataclass
class Run:
    name: str
    dump: dict[str, Any]  # the reference dump without the large frame tensor
    frame_error: float  # max abs difference of our loaded frames vs input_frames
    order: list[int]  # frames in the order propagate yielded them
    low_res: dict[int, torch.Tensor]
    video_res: dict[int, torch.Tensor]
    outputs: dict[int, FrameOutput]
    prompt_low_res: torch.Tensor


@pytest.fixture(scope="module", params=CASES, ids=[f"{m}-{t}" for m, t in CASES])
def run(request: pytest.FixtureRequest) -> Iterator[Run]:
    model_name, tag = request.param
    dump = torch.load(
        _require(DUMPS / model_name / f"video_bedroom_{tag}.pt"),
        map_location="cpu",
        weights_only=False,
    )
    for key in ("memory_attention_calls", "memory_encoder_calls"):
        del dump[key]
    predictor = VideoPredictor(_model(model_name))
    state = predictor.init_state(_require(VIDEO))
    frame_error = max_abs(state.images, dump["input_frames"])
    del dump["input_frames"]

    prompt = dump["prompt"]
    _, prompt_low_res, _ = predictor.add_points(
        state, prompt["frame"], 1, prompt["points"], prompt["labels"]
    )
    order: list[int] = []
    low_res: dict[int, torch.Tensor] = {}
    video_res: dict[int, torch.Tensor] = {}
    for frame_idx, per_obj, masks in predictor.propagate(state, reverse=dump["reverse"]):
        order.append(frame_idx)
        low_res[frame_idx] = per_obj[1]
        video_res[frame_idx] = masks
    out_dict = state.objects[1].output
    outputs = {**out_dict[NON_COND], **out_dict[COND]}
    yield Run(model_name, dump, frame_error, order, low_res, video_res, outputs, prompt_low_res)


def test_frame_loading_matches_reference(run: Run) -> None:
    assert run.frame_error < 1e-3, run.frame_error


def test_visits_the_reference_frames_in_order(run: Run) -> None:
    assert run.order == list(run.dump["frames"])
    assert len(run.order) == (31 if run.dump["reverse"] else run.dump["num_frames"])


def test_conditioning_frame_matches_reference(run: Run) -> None:
    ref = run.dump["cond_frame"]
    out = run.outputs[run.dump["prompt"]["frame"]]
    assert max_abs(run.prompt_low_res, ref["pred_masks"]) < FRAME_TOL
    assert max_abs(out.obj_ptr, ref["obj_ptr"]) < STATE_TOL
    assert max_abs(out.object_score_logits, ref["object_score_logits"]) < STATE_TOL
    assert out.maskmem_features is not None and out.maskmem_pos_enc is not None
    assert max_abs(out.maskmem_features, ref["maskmem_features"]) < STATE_TOL
    assert max_abs(out.maskmem_pos_enc[0], ref["maskmem_pos_enc"][0]) < STATE_TOL


def test_propagated_masks_match_reference(run: Run) -> None:
    ious, errs = [], []
    for frame_idx, ref in run.dump["frames"].items():
        ours = run.low_res[frame_idx]
        ious.append(mask_iou(ours, ref["pred_masks"]))
        errs.append(max_abs(ours, ref["pred_masks"]))
        assert mask_iou(run.video_res[frame_idx], ref["video_res_mask"]) >= IOU_MIN, frame_idx
    direction = "rev" if run.dump["reverse"] else "fwd"
    print(
        f"\n[{run.name} {direction}] min IoU {min(ious):.6f}, max abs low-res logit error "
        f"{max(errs):.3e} over {len(ious)} frames"
    )
    assert min(ious) >= IOU_MIN, min(ious)


def test_propagated_memory_state_matches_reference(run: Run) -> None:
    for frame_idx, ref in run.dump["frames"].items():
        out = run.outputs[frame_idx]
        assert out.maskmem_features is not None, frame_idx
        assert max_abs(out.object_score_logits, ref["object_score_logits"]) < STATE_TOL, frame_idx
        assert max_abs(out.obj_ptr, ref["obj_ptr"]) < STATE_TOL, frame_idx
        assert max_abs(out.maskmem_features, ref["maskmem_features"]) < STATE_TOL, frame_idx


# ---------------------------------------------------------------- behaviour


@pytest.fixture(scope="module")
def short_clip() -> list[np.ndarray]:
    paths = sorted(_require(VIDEO).glob("*.jpg"))[:4]
    return [np.array(Image.open(p).convert("RGB")) for p in paths]


def test_array_frames_match_folder_frames(short_clip: list[np.ndarray]) -> None:
    images, hw = load_frames(
        short_clip, 1024, CONFIGS["tiny"].pixel_mean, CONFIGS["tiny"].pixel_std
    )
    folder, hw_folder = load_frames(
        VIDEO, 1024, CONFIGS["tiny"].pixel_mean, CONFIGS["tiny"].pixel_std
    )
    assert hw == hw_folder == (540, 960)
    assert torch.equal(images, folder[:4])


@torch.no_grad()
def test_prompts_accumulate_and_box_goes_first(short_clip: list[np.ndarray]) -> None:
    predictor = VideoPredictor(_model("tiny"))
    state = predictor.init_state(short_clip)
    predictor.add_points(state, 0, 1, [[210, 350]], [1])
    predictor.add_points(state, 0, 1, [[400, 300]], [0], clear_old_points=False)
    coords, labels = state.objects[1].points[0]
    assert labels.tolist() == [[1, 0]]
    assert torch.allclose(coords[0, 0], torch.tensor([210 / 960 * 1024, 350 / 540 * 1024]))
    _, low_res, video_res = predictor.add_points(
        state, 0, 1, None, None, box=[150, 250, 300, 450], clear_old_points=False
    )
    assert 0 in state.objects[1].boxes and state.objects[1].points[0][1].shape[1] == 2
    assert low_res.shape == (1, 1, 256, 256) and video_res.shape == (1, 1, 540, 960)
    predictor.add_points(state, 0, 1, [[210, 350]], [1])  # clearing drops points and box
    assert 0 not in state.objects[1].boxes
    assert state.objects[1].points[0][1].tolist() == [[1]]


@torch.no_grad()
def test_two_objects_mask_prompt_and_correction(short_clip: list[np.ndarray]) -> None:
    predictor = VideoPredictor(_model("tiny"))
    state = predictor.init_state(short_clip)
    _, _, video_res = predictor.add_points(state, 0, 1, [[210, 350]], [1])
    mask = (video_res[0, 0] > 0).numpy()
    # A ground-truth mask is taken as the prediction: +-10 logits, object present.
    _, low_res, _ = predictor.add_mask(state, 2, 7, mask)
    assert float(low_res.max()) <= 10.0 and float(low_res.min()) >= -10.0
    assert float(state.objects[7].pending[2].object_score_logits) == 10.0

    seen = []
    for frame_idx, per_obj, masks in predictor.propagate(state):
        seen.append(frame_idx)
        assert set(per_obj) == {1, 7} and masks.shape == (2, 1, 540, 960)
    assert seen == [0, 1, 2, 3]
    obj7 = state.objects[7].output
    assert set(obj7[COND]) == {2} and set(obj7[NON_COND]) == {0, 1, 3}
    assert obj7[COND][2].maskmem_features is not None

    # A click on an already-tracked frame refines it with memory and makes it a
    # conditioning frame once propagation resumes.
    predictor.add_points(state, 1, 1, [[220, 360]], [1])
    assert 1 in state.objects[1].pending
    list(predictor.propagate(state, start_frame_idx=1, max_frames=1))
    obj1 = state.objects[1].output
    assert set(obj1[COND]) == {0, 1} and 1 not in obj1[NON_COND]
    assert not state.objects[1].pending


def test_reverse_from_first_frame_yields_it() -> None:
    predictor = VideoPredictor(_model("tiny"))
    frames = [np.zeros((64, 96, 3), dtype=np.uint8)] * 2
    state = predictor.init_state(frames)
    predictor.add_points(state, 0, 1, [[40, 30]], [1])
    assert [f for f, _, _ in predictor.propagate(state, reverse=True)] == [0]


def test_non_overlapping_keeps_only_the_best_object() -> None:
    logits = torch.tensor([[[[5.0, -1.0, 2.0]]], [[[3.0, 4.0, 2.5]]]])
    out = non_overlapping(logits)
    assert out[:, 0, 0].tolist() == [[5.0, -10.0, -10.0], [-10.0, 4.0, 2.5]]
    assert torch.equal(non_overlapping(logits[:1]), logits[:1])
