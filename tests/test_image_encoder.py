"""Image encoder parity against the official package (tools/dump_reference.py dumps)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch

from sam2_scratch.configs import CONFIGS, TINY
from sam2_scratch.hiera import Hiera
from sam2_scratch.image_encoder import ImageEncoder
from sam2_scratch.position_encoding import PositionEmbeddingSine
from sam2_scratch.weights import load_checkpoint
from tests.conftest import CHECKPOINTS, DUMPS, _require, max_abs

TOL = 1e-4
IMAGES = ("truck", "cars", "groceries")
CASES = [(m, img) for m in CONFIGS for img in IMAGES]

_ENCODERS: dict[str, ImageEncoder] = {}


def _encoder(model_name: str) -> ImageEncoder:
    if model_name not in _ENCODERS:
        cfg = CONFIGS[model_name]
        sd = load_checkpoint(_require(CHECKPOINTS[model_name]))
        _ENCODERS[model_name] = ImageEncoder.from_checkpoint(sd, cfg.hiera, cfg.neck)
    return _ENCODERS[model_name]


@pytest.fixture(scope="module", params=CASES, ids=[f"{m}-{i}" for m, i in CASES])
def case(request: pytest.FixtureRequest) -> Iterator[dict[str, Any]]:
    """One dump plus our outputs on it, computed once (each dump is ~300 MB)."""
    model_name, image = request.param
    _require(DUMPS / model_name / "manifest.json")
    dump = torch.load(
        _require(DUMPS / model_name / f"image_{image}.pt"), map_location="cpu", weights_only=False
    )
    enc = _encoder(model_name)
    with torch.no_grad():
        trunk = enc.trunk(dump["input_image"])
        # Neck on the reference trunk output isolates neck error from trunk error.
        neck_isolated = enc.neck(dump["trunk"]["output"])
        neck_chained = enc.neck(trunk)
        encoded = enc(dump["input_image"])
    yield {
        "ref_trunk": dump["trunk"]["output"],
        "ref_fpn": dump["neck"]["output"][0],
        "ref_pos": dump["neck"]["output"][1],
        "trunk": trunk,
        "neck_isolated": neck_isolated,
        "neck_chained": neck_chained,
        "encoded": encoded,
    }
    del dump


def _errors(ours: list[torch.Tensor], ref: list[torch.Tensor]) -> list[float]:
    assert len(ours) == len(ref)
    return [max_abs(a, b) for a, b in zip(ours, ref, strict=True)]


def test_trunk_stages(case: dict[str, Any]) -> None:
    errs = _errors(case["trunk"], case["ref_trunk"])
    assert max(errs) < TOL, f"trunk stage max abs errors {errs}"


def test_neck_fpn_and_pos(case: dict[str, Any]) -> None:
    fpn, pos = case["neck_isolated"]
    fpn_errs = _errors(fpn, case["ref_fpn"])
    pos_errs = _errors(pos, case["ref_pos"])
    assert max(fpn_errs) < TOL, f"fpn max abs errors {fpn_errs}"
    assert max(pos_errs) < TOL, f"pos max abs errors {pos_errs}"


def test_trunk_then_neck(case: dict[str, Any]) -> None:
    fpn, pos = case["neck_chained"]
    assert max(_errors(fpn, case["ref_fpn"])) < TOL
    assert max(_errors(pos, case["ref_pos"])) < TOL


def test_encoder_output_drops_scalp_level(case: dict[str, Any]) -> None:
    out = case["encoded"]
    fpn, pos = out["backbone_fpn"], out["vision_pos_enc"]
    assert len(fpn) == len(pos) == 3
    assert max(_errors(fpn, case["ref_fpn"][:3])) < TOL
    assert max(_errors(pos, case["ref_pos"][:3])) < TOL
    assert out["vision_features"].shape == (1, 256, 64, 64)
    assert max_abs(out["vision_features"], case["ref_fpn"][2]) < TOL


def test_random_weights_shapes_at_512() -> None:
    torch.manual_seed(0)
    enc = ImageEncoder(TINY.hiera, TINY.neck).eval()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        stages = enc.trunk(x)
        out = enc(x)
    assert [tuple(s.shape) for s in stages] == [
        (1, 96, 128, 128),
        (1, 192, 64, 64),
        (1, 384, 32, 32),
        (1, 768, 16, 16),
    ]
    assert [tuple(f.shape[-2:]) for f in out["backbone_fpn"]] == [(128, 128), (64, 64), (32, 32)]
    assert [tuple(p.shape) for p in out["vision_pos_enc"]] == [
        (1, 256, 128, 128),
        (1, 256, 64, 64),
        (1, 256, 32, 32),
    ]
    assert out["vision_features"].shape == (1, 256, 32, 32)


def test_parameter_names_match_checkpoint_layout() -> None:
    names = {n for n, _ in Hiera(TINY.hiera).named_parameters()}
    assert {"pos_embed", "pos_embed_window", "patch_embed.proj.weight"} <= names
    # Only the three stage-transition blocks carry a residual projection.
    assert {n for n in names if n.endswith(".proj.weight") and ".attn." not in n} == {
        "patch_embed.proj.weight",
        "blocks.1.proj.weight",
        "blocks.3.proj.weight",
        "blocks.10.proj.weight",
    }


def test_position_encoding_is_cached_and_batch_broadcast() -> None:
    pe = PositionEmbeddingSine(256)
    a = pe(torch.zeros(2, 8, 5, 7))
    b = pe(torch.ones(3, 1, 5, 7))
    assert a.shape == (2, 256, 5, 7)
    assert torch.equal(a[0], b[2])
    assert len(pe._cache) == 1
    assert dict(pe.state_dict()) == {}
