"""Parity of the memory encoder, memory attention and the assembled model.

Every memory-encoder and memory-attention call recorded in the video dumps is replayed
with the dumped inputs. The video dumps are ~0.9 GB, so each one is loaded once per
module and everything but the recorded calls is dropped immediately.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from sam2_scratch.configs import CONFIGS, TINY
from sam2_scratch.memory_attention import RoPEAttention, apply_rope, axial_rope_cis
from sam2_scratch.memory_encoder import MaskDownSampler, MemoryEncoder
from sam2_scratch.model import SAM2, FrameOutput, select_closest_cond_frames
from sam2_scratch.weights import load_checkpoint
from tests.conftest import CHECKPOINTS, DUMPS, ROOT, _require, max_abs

TOL = 1e-4
CASES = [(m, tag) for m in CONFIGS for tag in ("fwd", "rev")]

_MODELS: dict[str, SAM2] = {}


def _model(name: str) -> SAM2:
    if name not in _MODELS:
        _MODELS[name] = SAM2.from_checkpoint(_require(CHECKPOINTS[name]), CONFIGS[name])
    return _MODELS[name]


@pytest.fixture(scope="module", params=CASES, ids=[f"{m}-{t}" for m, t in CASES])
def calls(request: pytest.FixtureRequest) -> Iterator[tuple[SAM2, dict[str, Any]]]:
    model_name, tag = request.param
    path = _require(DUMPS / model_name / f"video_bedroom_{tag}.pt")
    dump = torch.load(path, map_location="cpu", weights_only=False)
    kept = {k: dump[k] for k in ("memory_encoder_calls", "memory_attention_calls")}
    del dump
    yield _model(model_name), kept


@torch.no_grad()
def test_memory_encoder_matches_reference(calls: tuple[SAM2, dict[str, Any]]) -> None:
    model, dump = calls
    assert dump["memory_encoder_calls"], "dump has no recorded memory encoder calls"
    for frame, call in dump["memory_encoder_calls"].items():
        out = model.memory_encoder(*call["args"], **call["kwargs"])
        ref = call["output"]
        assert max_abs(out["vision_features"], ref["vision_features"]) < TOL, frame
        assert max_abs(out["vision_pos_enc"][0], ref["vision_pos_enc"][0]) < TOL, frame


@torch.no_grad()
def test_memory_attention_matches_reference(calls: tuple[SAM2, dict[str, Any]]) -> None:
    model, dump = calls
    assert dump["memory_attention_calls"], "dump has no recorded memory attention calls"
    for frame, call in dump["memory_attention_calls"].items():
        kw = call["kwargs"]
        out = model.memory_attention(
            curr=kw["curr"][0],
            curr_pos=kw["curr_pos"][0],
            memory=kw["memory"],
            memory_pos=kw["memory_pos"],
            num_obj_ptr_tokens=kw["num_obj_ptr_tokens"],
        )
        n_ptr = kw["num_obj_ptr_tokens"]
        # The bank layout is k spatial frames of 4096 tokens plus 4 tokens per pointer.
        assert (kw["memory"].shape[0] - n_ptr) % 4096 == 0 and n_ptr % 4 == 0, frame
        assert max_abs(out, call["output"]) < TOL, frame


@pytest.mark.parametrize("name", list(CONFIGS))
def test_from_checkpoint_consumes_every_key(name: str) -> None:
    sd = load_checkpoint(_require(CHECKPOINTS[name]))
    model = _model(name)
    assert set(model.state_dict()) == set(sd)
    for key, value in model.state_dict().items():
        assert torch.equal(value, sd[key]), key


def test_from_checkpoint_rejects_unconsumed_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import sam2_scratch.model as model_mod

    sd = load_checkpoint(_require(CHECKPOINTS["tiny"]))
    sd["unexpected.weight"] = torch.zeros(1)
    monkeypatch.setattr(model_mod, "load_checkpoint", lambda _path: sd)
    with pytest.raises(KeyError, match="unexpected.weight"):
        SAM2.from_checkpoint("ignored.pt", TINY)


@torch.no_grad()
@pytest.mark.parametrize("name", list(CONFIGS))
def test_full_image_pipeline_truck(name: str) -> None:
    """truck.jpg -> preprocessing -> encoder (+ no-mem embed) -> SAM head."""
    dump = torch.load(
        _require(DUMPS / name / "image_truck.pt"), map_location="cpu", weights_only=False
    )
    predictor = _model(name).image_predictor()
    image = np.array(Image.open(_require(ROOT / "data" / "images" / "truck.jpg")).convert("RGB"))
    predictor.set_image(image)
    _, ious, low_res = predictor.predict(
        np.array([[500, 375]], dtype=np.float32),
        np.array([1], dtype=np.int32),
        multimask_output=True,
        return_logits=True,
    )
    ref = dump["prompts"]["one_point/multi"]
    assert max_abs(torch.from_numpy(low_res), ref["low_res_masks"]) < TOL
    assert max_abs(torch.from_numpy(ious), ref["iou_predictions"]) < TOL


# ---------------------------------------------------------------- unit tests


def test_rope_rotates_pairs_by_axis() -> None:
    cis = axial_rope_cis(8, end_x=3, end_y=2)
    assert cis.shape == (6, 4, 2)
    # Token 4 is (x=1, y=1); pairs 0-1 follow x, pairs 2-3 follow y, freqs 1 and 0.01.
    angles = torch.atan2(cis[..., 1], cis[..., 0])
    assert torch.allclose(angles[4], torch.tensor([1.0, 0.01, 1.0, 0.01]), atol=1e-6)
    assert torch.allclose(angles[2], torch.tensor([2.0, 0.02, 0.0, 0.0]), atol=1e-6)
    x = torch.randn(1, 1, 6, 8)
    y = apply_rope(x, cis)
    # A rotation preserves the norm of every channel pair.
    pair_norm = x.reshape(1, 1, 6, 4, 2).norm(dim=-1)
    assert torch.allclose(y.reshape(1, 1, 6, 4, 2).norm(dim=-1), pair_norm, atol=1e-5)
    assert torch.allclose(y[..., 0, :], x[..., 0, :])  # position (0, 0) is not rotated


def test_rope_attention_key_tiling_and_pointer_exclusion() -> None:
    torch.manual_seed(0)
    attn = RoPEAttention(16, 1, kv_in_dim=8, rope_k_repeat=True, feat_sizes=(2, 2)).eval()
    q = torch.randn(1, 4, 16)
    mem = torch.randn(1, 2 * 4 + 3, 8)  # two frames of the 2x2 grid + 3 pointer tokens
    out = attn(q, mem, mem, num_k_exclude_rope=3)
    assert out.shape == (1, 4, 16)
    with pytest.raises(ValueError):
        attn(q, mem, mem, num_k_exclude_rope=0)  # 11 keys do not tile a 4-token grid
    strict = RoPEAttention(16, 1, kv_in_dim=8, rope_k_repeat=False, feat_sizes=(2, 2))
    with pytest.raises(ValueError):
        strict(q, mem, mem, num_k_exclude_rope=3)


def test_memory_encoder_shapes_and_layout() -> None:
    down = MaskDownSampler()
    assert [i for i, m in enumerate(down.encoder) if isinstance(m, torch.nn.Conv2d)] == [
        0,
        3,
        6,
        9,
        12,
    ]
    enc = MemoryEncoder(TINY.memory_encoder).eval()
    with torch.no_grad():
        out = enc(torch.randn(1, 256, 16, 16), torch.randn(1, 1, 256, 256))
    assert out["vision_features"].shape == (1, 64, 16, 16)
    assert out["vision_pos_enc"][0].shape == (1, 64, 16, 16)


def _fake_out() -> FrameOutput:
    return FrameOutput(torch.zeros(1), None, torch.zeros(1), torch.zeros(1))


def test_select_closest_cond_frames() -> None:
    conds = {t: _fake_out() for t in (0, 10, 20, 30, 40)}
    sel, unsel = select_closest_cond_frames(22, conds, -1)
    assert set(sel) == set(conds) and not unsel
    sel, unsel = select_closest_cond_frames(22, conds, 3)
    # Closest before (20), closest at-or-after (30), then the nearest remaining (10).
    assert set(sel) == {10, 20, 30} and set(unsel) == {0, 40}
