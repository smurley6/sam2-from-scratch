"""ONNX export of the 512 px tiny graphs: contract, op set and fp32 parity with torch.

Exports into a temp dir (about 10 s) and checks one image with the full prompt set from the
manifest; ``tools/ort_parity.py`` runs the complete table over all sizes, precisions and
images.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import CHECKPOINTS, DUMPS, ROOT, _require

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxslim")

from tools import ort_parity as parity  # noqa: E402
from tools.export_onnx import (  # noqa: E402
    ENCODER_INPUTS,
    ENCODER_OUTPUTS,
    HEAD_INPUTS,
    HEAD_OUTPUTS,
    WEBGPU_UNSUPPORTED,
    TorchSam,
    export_image_encoder,
    export_sam_head,
    load_torch,
    op_counts,
)

SIZE = 512
IMAGE = "truck"


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[TorchSam, Path]]:
    _require(CHECKPOINTS["tiny"])
    _require(DUMPS / "tiny" / "manifest.json")
    _require(ROOT / "data" / "images" / f"{IMAGE}.jpg")
    out = tmp_path_factory.mktemp("onnx")
    sam = load_torch("tiny")
    export_image_encoder(sam, SIZE, out / f"image_encoder_{SIZE}.onnx")
    export_sam_head(sam, SIZE, out / f"sam_head_{SIZE}.onnx")
    yield sam, out


def _io(sess: ort.InferenceSession) -> tuple[dict[str, list[object]], dict[str, list[object]]]:
    return (
        {i.name: list(i.shape) for i in sess.get_inputs()},
        {o.name: list(o.shape) for o in sess.get_outputs()},
    )


def test_graph_contract(exported: tuple[TorchSam, Path]) -> None:
    _, out = exported
    enc = ort.InferenceSession(str(out / f"image_encoder_{SIZE}.onnx"))
    ins, outs = _io(enc)
    assert list(ins) == ENCODER_INPUTS and list(outs) == ENCODER_OUTPUTS
    assert ins["image"] == [1, 3, SIZE, SIZE]
    assert outs == {
        "image_embed": [1, 256, SIZE // 16, SIZE // 16],
        "high_res_feat_s0": [1, 32, SIZE // 4, SIZE // 4],
        "high_res_feat_s1": [1, 64, SIZE // 8, SIZE // 8],
    }
    head = ort.InferenceSession(str(out / f"sam_head_{SIZE}.onnx"))
    ins, outs = _io(head)
    assert list(ins) == HEAD_INPUTS and list(outs) == HEAD_OUTPUTS
    assert ins["point_coords"] == [1, "num_points", 2]
    assert ins["point_labels"] == [1, "num_points"]
    assert ins["mask_input"] == [1, 1, SIZE // 4, SIZE // 4]
    assert outs["masks"] == [1, 4, SIZE // 4, SIZE // 4]
    assert outs["sam_tokens"] == [1, 4, 256]
    assert {i.name: i.type for i in head.get_inputs()}["point_labels"] == "tensor(int32)"


def test_op_set_is_webgpu_friendly(exported: tuple[TorchSam, Path]) -> None:
    _, out = exported
    for name in (f"image_encoder_{SIZE}.onnx", f"sam_head_{SIZE}.onnx"):
        ops = op_counts(out / name)
        assert not set(ops) & WEBGPU_UNSUPPORTED, name
    # Every encoder shape is static, so no shape arithmetic survives simplification.
    assert "Shape" not in op_counts(out / f"image_encoder_{SIZE}.onnx")


def test_fp32_parity_one_image(exported: tuple[TorchSam, Path]) -> None:
    sam, out = exported
    rows = parity.evaluate("tiny", SIZE, out, images=[IMAGE], sam=sam, latency_runs=1)
    assert {(r.graph, r.precision) for r in rows} == {
        (f"image_encoder_{SIZE}", "fp32"),
        (f"sam_head_{SIZE}", "fp32"),
    }
    for row in rows:
        assert row.max_abs is not None and row.max_abs < parity.FP32_MAX_ABS, row
        assert row.iou_min >= parity.FP32_IOU, row
        assert row.flips == 0, row


@torch.no_grad()
def test_head_no_prompt_and_many_points(exported: tuple[TorchSam, Path]) -> None:
    """The video loop's prompt-free call (one padding point) plus a mask, and a long P."""
    sam, out = exported
    gen = torch.Generator().manual_seed(1)
    g = SIZE // 16
    feats = (
        torch.randn(1, 256, g, g, generator=gen),
        torch.randn(1, 32, 4 * g, 4 * g, generator=gen),
        torch.randn(1, 64, 2 * g, 2 * g, generator=gen),
    )
    np_feats = {k: t.numpy() for k, t in zip(ENCODER_OUTPUTS, feats, strict=True)}
    head = ort.InferenceSession(str(out / f"sam_head_{SIZE}.onnx"))
    mask = (torch.randn(1, 1, 4 * g, 4 * g, generator=gen) * 8).numpy()
    cases = [
        (np.zeros((1, 1, 2), np.float32), np.array([[-1]], np.int32), mask),
        (
            (torch.rand(1, 6, 2, generator=gen) * SIZE).numpy(),
            np.array([[2, 3, 1, 0, 1, 1]], np.int32),
            None,
        ),
    ]
    for coords, labels, m in cases:
        prompt = parity.Prompt("case", coords, labels)
        want = parity.torch_head(sam, SIZE, feats, prompt, m)
        got = parity.run(head, parity.head_feeds(np_feats, prompt, m, SIZE))
        for key, ref in want.items():
            assert np.abs(got[key] - ref).max() < parity.FP32_MAX_ABS, key
