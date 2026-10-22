"""ONNX export of the 512 px tiny graphs: contract, op set and fp32 parity with torch.

Exports into a temp dir (about 15 s) and checks one image with the full prompt set from the
manifest, the memory graphs, and three frames of the onnxruntime video loop against the
torch video predictor at 512; ``tools/ort_parity.py`` runs the complete table over all
sizes, precisions, images and the full clip.
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
from tools import ort_video_loop as loop  # noqa: E402
from tools.export_onnx import (  # noqa: E402
    ENCODER_INPUTS,
    ENCODER_OUTPUTS,
    HEAD_INPUTS,
    HEAD_OUTPUTS,
    MEMORY_ATTENTION_INPUTS,
    MEMORY_ATTENTION_OUTPUTS,
    MEMORY_ENCODER_INPUTS,
    MEMORY_ENCODER_OUTPUTS,
    WEBGPU_UNSUPPORTED,
    TorchSam,
    export_image_encoder,
    export_memory_attention,
    export_memory_encoder,
    export_object_pointer,
    export_pointer_tpos,
    export_sam_head,
    load_torch,
    op_counts,
    write_constants,
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


@pytest.fixture(scope="module")
def exported_video(exported: tuple[TorchSam, Path]) -> tuple[TorchSam, Path]:
    """Adds the memory graphs, pointer graphs and constants to the image graphs."""
    sam, out = exported
    _require(ROOT / "data" / "videos" / "bedroom")
    export_memory_encoder(sam, SIZE, out / f"memory_encoder_{SIZE}.onnx")
    export_memory_attention(sam, SIZE, out / f"memory_attention_{SIZE}.onnx")
    export_object_pointer(sam, out / "object_pointer.onnx")
    export_pointer_tpos(sam, out / "object_pointer_tpos.onnx")
    write_constants(sam, SIZE, out / f"constants_{SIZE}.json")
    return sam, out


def test_memory_graph_contract_and_ops(exported_video: tuple[TorchSam, Path]) -> None:
    _, out = exported_video
    g = SIZE // 16
    enc = ort.InferenceSession(str(out / f"memory_encoder_{SIZE}.onnx"))
    ins, outs = _io(enc)
    assert list(ins) == MEMORY_ENCODER_INPUTS and list(outs) == MEMORY_ENCODER_OUTPUTS
    assert ins["mask_logits"] == [1, 1, SIZE, SIZE] and ins["binarize"] == [1]
    assert outs["maskmem_features"] == [1, 64, g, g]
    attn = ort.InferenceSession(str(out / f"memory_attention_{SIZE}.onnx"))
    ins, outs = _io(attn)
    assert list(ins) == MEMORY_ATTENTION_INPUTS and list(outs) == MEMORY_ATTENTION_OUTPUTS
    assert ins["curr"] == [1, g * g, 256] and ins["memory"] == [1, "num_memory", 64]
    assert outs == {"fused": [1, g * g, 256], "image_embed": [1, 256, g, g]}
    names = (
        f"memory_encoder_{SIZE}.onnx",
        f"memory_attention_{SIZE}.onnx",
        "object_pointer.onnx",
        "object_pointer_tpos.onnx",
    )
    for name in names:
        assert not set(op_counts(out / name)) & WEBGPU_UNSUPPORTED, name


@torch.no_grad()
def test_memory_encoder_occlusion_and_binarize(exported_video: tuple[TorchSam, Path]) -> None:
    """The branches the bedroom clip never takes: clicked (binarised) masks and frames
    where the object is absent (``no_obj_embed_spatial`` added)."""
    sam, out = exported_video
    gen = torch.Generator().manual_seed(2)
    g = SIZE // 16
    base = parity._encoder_call(
        sam,
        0,
        torch.randn(1, 256, g, g, generator=gen),
        torch.randn(1, 1, SIZE, SIZE, generator=gen) * 6,
        torch.ones(1, 1),
        torch.zeros(1, 64, g, g),
        torch.zeros(1, 64, g, g),
        binarize=False,
    )
    sess = ort.InferenceSession(str(out / f"memory_encoder_{SIZE}.onnx"))
    for call in parity._synthetic_encoder_calls(sam, base):
        got = parity.run(sess, call.feeds)
        for key, want in call.want.items():
            assert np.abs(got[key] - want).max() < parity.FP32_MAX_ABS, key


def test_video_loop_three_frames(exported_video: tuple[TorchSam, Path]) -> None:
    """Click on frame 0, track frames 1-2 purely through onnxruntime; compare with the
    torch predictor at 512 on the same 3-frame clip (which also caps the pointer window
    at num_frames = 3)."""
    _, out = exported_video
    result = loop.evaluate(SIZE, loop.PRECISION_SETS["fp32"], out, max_frames=3, runs=loop.RUNS[:1])
    run = result["runs"]["fwd"]
    assert run["frames"] == 3
    for frame, stats in run["per_frame"].items():
        assert stats["iou"] >= 0.99, (frame, stats)
        assert stats["score_err"] < 1e-2, (frame, stats)
