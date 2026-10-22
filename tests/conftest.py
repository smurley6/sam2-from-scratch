"""Shared fixtures: checkpoint paths and reference dumps produced by tools/dump_reference.py.

Tests that need a dump or checkpoint skip cleanly when the files are absent so the suite
still runs on a fresh clone (the dumps are gitignored; see docs/PROGRAM.md T1).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = {
    "tiny": ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt",
    "small": ROOT / "checkpoints" / "sam2.1_hiera_small.pt",
}
DUMPS = ROOT / "reference_dumps"


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"missing {path.relative_to(ROOT)}; run tools/dump_reference.py")
    return path


@pytest.fixture(params=["tiny", "small"])
def model_name(request: pytest.FixtureRequest) -> str:
    """Parametrises a test over both released sizes; skips sizes without dumps."""
    name = str(request.param)
    _require(CHECKPOINTS[name])
    _require(DUMPS / name / "manifest.json")
    return name


@pytest.fixture
def checkpoint_path(model_name: str) -> Path:
    return CHECKPOINTS[model_name]


@pytest.fixture
def manifest(model_name: str) -> dict[str, Any]:
    return json.loads((DUMPS / model_name / "manifest.json").read_text())  # type: ignore[no-any-return]


@pytest.fixture
def load_dump(model_name: str) -> Callable[[str], dict[str, Any]]:
    """``load_dump("image_truck")`` returns the dict saved by the dump tool."""

    def _load(name: str) -> dict[str, Any]:
        path = _require(DUMPS / model_name / f"{name}.pt")
        return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[no-any-return]

    return _load


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape, f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
    return float((a.float() - b.float()).abs().max())


def mask_iou(a: torch.Tensor, b: torch.Tensor, threshold: float = 0.0) -> float:
    """IoU of two logit masks binarised at ``threshold``; 1.0 when both are empty."""
    pa, pb = a > threshold, b > threshold
    inter = (pa & pb).sum().item()
    union = (pa | pb).sum().item()
    return 1.0 if union == 0 else float(inter / union)
