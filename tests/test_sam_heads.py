"""Parity of the prompt encoder and mask decoder against the reference dumps.

Every recorded prompt-encoder and mask-decoder call is replayed on the reimplemented
modules with the dumped inputs. Dumps are ~300 MB each, so each (model, image) pair is
loaded once per module.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from sam2_scratch.mask_decoder import MaskDecoder
from sam2_scratch.prompt_encoder import PositionEmbeddingRandom, PromptEncoder
from sam2_scratch.weights import load_checkpoint, load_prefix
from tests.conftest import CHECKPOINTS, DUMPS, _require, max_abs

IMAGES = ("truck", "cars", "groceries")
PE_TOL = 1e-5
MD_TOL = 1e-4


def build_heads(model: str) -> tuple[PromptEncoder, MaskDecoder]:
    sd = load_checkpoint(_require(CHECKPOINTS[model]))
    pe = PromptEncoder()
    load_prefix(pe, sd, "sam_prompt_encoder")
    md = MaskDecoder()
    load_prefix(md, sd, "sam_mask_decoder")
    return pe.eval(), md.eval()


@pytest.fixture(scope="module", params=["tiny", "small"])
def heads(request: pytest.FixtureRequest) -> tuple[str, PromptEncoder, MaskDecoder]:
    model = str(request.param)
    _require(DUMPS / model / "manifest.json")
    return (model, *build_heads(model))


@pytest.fixture(scope="module", params=IMAGES)
def case(
    request: pytest.FixtureRequest, heads: tuple[str, PromptEncoder, MaskDecoder]
) -> tuple[PromptEncoder, MaskDecoder, dict[str, Any]]:
    model, pe, md = heads
    path = _require(DUMPS / model / f"image_{request.param}.pt")
    dump = torch.load(path, map_location="cpu", weights_only=False)
    return pe, md, dump


@torch.no_grad()
def test_prompt_encoder_matches_reference(
    case: tuple[PromptEncoder, MaskDecoder, dict[str, Any]],
) -> None:
    pe, _, dump = case
    for key, rec in dump["prompts"].items():
        call = rec["prompt_encoder"]
        sparse, dense = pe(*call["args"], **call["kwargs"])
        ref_sparse, ref_dense = call["output"]
        assert max_abs(sparse, ref_sparse) < PE_TOL, key
        assert max_abs(dense, ref_dense) < PE_TOL, key


@torch.no_grad()
def test_mask_decoder_matches_reference(
    case: tuple[PromptEncoder, MaskDecoder, dict[str, Any]],
) -> None:
    pe, md, dump = case
    image_pe = pe.get_dense_pe()
    for key, rec in dump["prompts"].items():
        sparse, dense = rec["prompt_encoder"]["output"]
        out = md(
            image_embeddings=dump["image_embed"],
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=rec["mask_decoder_multimask"],
            high_res_features=dump["high_res_feats"],
        )
        names = ("masks", "iou", "sam_tokens", "object_score")
        for name, got, ref in zip(names, out, rec["mask_decoder_output"], strict=True):
            assert max_abs(got, ref) < MD_TOL, f"{key} {name}"


def test_dense_pe_grid() -> None:
    pe = PositionEmbeddingRandom(128)
    grid = pe((64, 64))
    assert grid.shape == (256, 64, 64)
    # Pixel-centre convention: cell (0, 0) encodes normalised (0.5/64, 0.5/64).
    coords = torch.tensor([[0.5 / 64, 0.5 / 64]])
    assert torch.allclose(grid[:, 0, 0], pe._pe_encoding(coords)[0], atol=1e-6)


def test_padding_point_has_no_position() -> None:
    pe = PromptEncoder()
    coords = torch.tensor([[[100.0, 200.0]]])
    labels = torch.tensor([[1]], dtype=torch.int32)
    sparse, dense = pe(points=(coords, labels))
    assert sparse.shape == (1, 2, 256)
    assert dense.shape == (1, 256, 64, 64)
    assert torch.equal(sparse[0, 1], pe.not_a_point_embed.weight[0])
    # With a box the decoder sees no padding token.
    sparse, _ = pe(points=(coords, labels), boxes=torch.tensor([[0.0, 0.0, 10.0, 10.0]]))
    assert sparse.shape == (1, 3, 256)


def test_dynamic_multimask_fallback() -> None:
    md = MaskDecoder().eval()
    logits = torch.full((1, 4, 8, 8), -10.0)
    logits[0, 0] = 0.0  # every pixel at threshold: maximally unstable
    logits[0, 2, :4] = 10.0
    iou = torch.tensor([[0.9, 0.1, 0.8, 0.3]])
    masks, scores = md._dynamic_multimask_via_stability(logits, iou)
    assert torch.equal(masks, logits[:, 2:3])
    assert torch.equal(scores, iou[:, 2:3])
    logits[0, 0] = logits[0, 2]  # now stable: the single-mask output is kept
    masks, scores = md._dynamic_multimask_via_stability(logits, iou)
    assert torch.equal(masks, logits[:, 0:1])
    assert torch.equal(scores, iou[:, 0:1])


@pytest.mark.reference
@torch.no_grad()
def test_prompt_encoder_mask_and_box_paths_match_reference() -> None:
    """The dumps never pass ``boxes`` or ``masks``; check those paths directly."""
    ref_mod = pytest.importorskip("sam2.modeling.sam.prompt_encoder")
    sd = load_checkpoint(_require(CHECKPOINTS["tiny"]))
    pe = PromptEncoder()
    load_prefix(pe, sd, "sam_prompt_encoder")
    ref = ref_mod.PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16,
    )
    load_prefix(ref, sd, "sam_prompt_encoder")
    pe.eval()
    ref.eval()
    gen = torch.Generator().manual_seed(0)
    points = (
        torch.rand(2, 3, 2, generator=gen) * 1024,
        torch.tensor([[1, 0, -1], [1, 1, 0]], dtype=torch.int32),
    )
    boxes = torch.tensor([[10.0, 20.0, 300.0, 400.0], [50.0, 60.0, 700.0, 800.0]])
    masks = torch.randn(2, 1, 256, 256, generator=gen) * 10
    for kwargs in (
        {"points": points, "boxes": boxes, "masks": masks},
        {"points": None, "boxes": boxes, "masks": None},
        {"points": points, "boxes": None, "masks": masks},
    ):
        for got, want in zip(pe(**kwargs), ref(**kwargs), strict=True):
            assert max_abs(got, want) < PE_TOL
