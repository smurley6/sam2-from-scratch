"""The assembled SAM 2 model: image encoder, SAM head, memory encoder, memory attention.

This module holds the per-frame building blocks of the video loop (``track_step`` and
the pieces it is made of). It keeps no state across frames: the caller owns the memory
bank (``OutputDict``) and decides which frames are conditioning frames. The video
predictor in ``video_predictor.py`` is that caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sam2_scratch.common import MLP, sine_pe_1d
from sam2_scratch.configs import SAM2Config
from sam2_scratch.image_encoder import ImageEncoder
from sam2_scratch.image_predictor import ImageEncoderForward, ImagePredictor
from sam2_scratch.mask_decoder import MaskDecoder
from sam2_scratch.memory_attention import MemoryAttention
from sam2_scratch.memory_encoder import MemoryEncoder
from sam2_scratch.prompt_encoder import LABEL_PAD, PromptEncoder
from sam2_scratch.weights import load_checkpoint

# Logit written into masks of frames where the object is judged absent: low enough that
# the sigmoid is exactly 0 and every downstream threshold reads "background".
NO_OBJ_SCORE = -1024.0

COND = "cond_frame_outputs"
NON_COND = "non_cond_frame_outputs"


def tokens_to_map(tokens: Tensor, hw: tuple[int, int]) -> Tensor:
    """``[HW, B, C]`` tokens -> ``[B, C, H, W]`` as a view (channels-last strides).

    Every spatial map handed to a conv or the SAM head is produced this way rather than
    taken from the backbone output directly: CPU conv kernels can round differently
    depending on the input strides, and reproducing the reference's views keeps the
    memory features bit-identical to it.
    """
    _, b, c = tokens.shape
    return tokens.permute(1, 2, 0).view(b, c, *hw)


@dataclass
class ImageFeatures:
    """Backbone output for one batch of frames, ready for the SAM head.

    Levels are finest first and stored as token sequences ``[HW, B, C]``: stride 4
    (``C`` = 32) and stride 8 (64), both already projected by the mask decoder's
    ``conv_s0`` / ``conv_s1``, then the stride-16 image embedding (256). ``pos`` holds
    the matching sine encodings, ``sizes`` the ``(H, W)`` of each level.
    """

    tokens: list[Tensor]
    pos: list[Tensor]
    sizes: list[tuple[int, int]]

    @property
    def vision_feat(self) -> Tensor:
        """The stride-16 embedding as a token sequence ``[HW, B, C]``."""
        return self.tokens[-1]

    @property
    def vision_pos(self) -> Tensor:
        return self.pos[-1]

    @property
    def image_embed(self) -> Tensor:
        return tokens_to_map(self.tokens[-1], self.sizes[-1])

    @property
    def high_res_feats(self) -> list[Tensor]:
        return [tokens_to_map(t, s) for t, s in zip(self.tokens[:-1], self.sizes[:-1], strict=True)]


class SamOutputs(NamedTuple):
    low_res_multimasks: Tensor  # [B, M, 256, 256]
    high_res_multimasks: Tensor  # [B, M, 1024, 1024]
    ious: Tensor  # [B, M]
    low_res_masks: Tensor  # [B, 1, 256, 256], the chosen mask
    high_res_masks: Tensor  # [B, 1, 1024, 1024]
    obj_ptr: Tensor  # [B, 256]
    object_score_logits: Tensor  # [B, 1]


@dataclass
class FrameOutput:
    """What the memory bank keeps for one (object, frame).

    ``maskmem_features`` is ``None`` until the memory encoder has run on the frame (the
    predictor defers this for prompted frames until propagation starts).
    """

    pred_masks: Tensor  # low-res logits [B, 1, 256, 256]
    pred_masks_high_res: Tensor | None  # [B, 1, 1024, 1024]
    obj_ptr: Tensor  # [B, 256]
    object_score_logits: Tensor  # [B, 1]
    maskmem_features: Tensor | None = None  # [B, 64, 64, 64]
    maskmem_pos_enc: list[Tensor] | None = None  # [[B, 64, 64, 64]]


# {"cond_frame_outputs": {frame: out}, "non_cond_frame_outputs": {frame: out}}
OutputDict = dict[str, dict[int, FrameOutput]]


class SAM2(nn.Module):
    """SAM 2.1 with every released parameter. ``from_checkpoint`` loads strictly."""

    def __init__(self, cfg: SAM2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.hidden_dim = cfg.hidden_dim
        self.mem_dim = cfg.mem_dim
        self.num_maskmem = cfg.num_maskmem
        emb = cfg.image_embedding_size

        self.image_encoder = ImageEncoder(cfg.hiera, cfg.neck)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=cfg.prompt_embed_dim,
            image_embedding_size=(emb, emb),
            input_image_size=(cfg.image_size, cfg.image_size),
            mask_in_chans=cfg.mask_in_chans,
        )
        self.sam_mask_decoder = MaskDecoder(
            transformer_dim=cfg.prompt_embed_dim,
            num_multimask_outputs=cfg.num_multimask_outputs,
            iou_head_depth=cfg.iou_head_depth,
            iou_head_hidden_dim=cfg.iou_head_hidden_dim,
            transformer_depth=cfg.decoder_depth,
            transformer_heads=cfg.decoder_heads,
            transformer_mlp_dim=cfg.decoder_mlp_dim,
            use_multimask_token_for_obj_ptr=cfg.use_multimask_token_for_obj_ptr,
        )
        self.memory_encoder = MemoryEncoder(cfg.memory_encoder)
        self.memory_attention = MemoryAttention(cfg.memory_attention)

        c, m = cfg.hidden_dim, cfg.mem_dim
        # Added to the image embedding of a frame that has no memory to attend to.
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, c))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, 1, c))  # unused at inference
        # Pointer written for frames where the object is absent (occlusion).
        self.no_obj_ptr = nn.Parameter(torch.zeros(1, c))
        # Added to the spatial memory of frames where the object is absent.
        self.no_obj_embed_spatial = nn.Parameter(torch.zeros(1, m))
        # Temporal encoding of a memory's slot: row 0 is the previous frame, row
        # maskmem_tpos_rows - 1 (the last trained row) is used by conditioning frames.
        # The rows stay at the trained count so that a smaller num_maskmem (an
        # inference-time ablation) still loads the checkpoint and keeps each slot's
        # trained encoding.
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(cfg.maskmem_tpos_rows, 1, 1, m))
        self.obj_ptr_proj = MLP(c, c, c, 3)
        self.obj_ptr_tpos_proj = nn.Linear(c, m)
        # 1024 -> 256 downsampler used to feed a ground-truth mask prompt to the SAM head.
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)

        # The reference video predictor overrides this to True when it builds the model;
        # the predictor sets it accordingly (see video_predictor.py).
        self.binarize_mask_from_pts_for_mem_enc = cfg.binarize_mask_from_pts_for_mem_enc

    @classmethod
    def from_checkpoint(cls, path: str | Path, cfg: SAM2Config) -> SAM2:
        """Build and load every tensor of a released checkpoint, failing on any mismatch."""
        sd = load_checkpoint(path)
        model = cls(cfg)
        ours = set(model.state_dict().keys())
        theirs = set(sd.keys())
        if ours != theirs:
            raise KeyError(
                "checkpoint key mismatch:\n"
                f"  missing from checkpoint: {sorted(ours - theirs)[:20]}\n"
                f"  not consumed by the model: {sorted(theirs - ours)[:20]}"
            )
        model.load_state_dict(sd, strict=True)
        return model.eval()

    @property
    def device(self) -> torch.device:
        return self.no_mem_embed.device

    # ------------------------------------------------------------------ image side

    def forward_image(self, x: Tensor) -> ImageFeatures:
        """Encode ``x [B, 3, S, S]`` (normalised) into SAM-head-ready features."""
        out = self.image_encoder(x)
        fpn = list(cast(list[Tensor], out["backbone_fpn"]))
        pos = list(cast(list[Tensor], out["vision_pos_enc"]))
        # The decoder's skip projections depend only on the image, so they run once per
        # frame here rather than once per prompt inside the decoder.
        fpn[0] = self.sam_mask_decoder.conv_s0(fpn[0])
        fpn[1] = self.sam_mask_decoder.conv_s1(fpn[1])
        sizes = [(int(f.shape[-2]), int(f.shape[-1])) for f in fpn]
        tokens = [f.flatten(2).permute(2, 0, 1) for f in fpn]
        pos_tokens = [p.flatten(2).permute(2, 0, 1) for p in pos]
        return ImageFeatures(tokens, pos_tokens, sizes)

    def encode_for_image_predictor(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Single-image path: the embedding carries ``no_mem_embed``, as in tracking a
        first frame, because SAM 2's head was trained on memory-conditioned inputs."""
        feats = self.forward_image(x)
        embed = tokens_to_map(feats.vision_feat + self.no_mem_embed, feats.sizes[-1])
        return embed, feats.high_res_feats

    def image_predictor(self) -> ModelImagePredictor:
        return ModelImagePredictor(self)

    # ------------------------------------------------------------------ SAM head

    def _use_multimask(self, is_init_cond_frame: bool, num_points: int) -> bool:
        """Ambiguity is only resolvable with multiple outputs for 0 or 1 clicks."""
        cfg = self.cfg
        return (
            cfg.multimask_output_in_sam
            and (is_init_cond_frame or cfg.multimask_output_for_tracking)
            and cfg.multimask_min_pt_num <= num_points <= cfg.multimask_max_pt_num
        )

    def sam_heads(
        self,
        image_embed: Tensor,
        high_res_feats: list[Tensor],
        points: tuple[Tensor, Tensor] | None = None,
        mask_inputs: Tensor | None = None,
        multimask_output: bool = False,
    ) -> SamOutputs:
        """Prompt encoder + mask decoder + object pointer for a batch of frames/objects.

        ``points`` is ``(coords [B, P, 2], labels [B, P])`` in model-input pixels;
        ``mask_inputs`` is mask logits ``[B, 1, h, w]`` (resized to 256x256 if needed).
        """
        b = image_embed.shape[0]
        dev = image_embed.device
        if points is None:
            # The decoder always sees at least one sparse token; a lone padding point
            # carries only the learned "not a point" embedding.
            points = (
                torch.zeros(b, 1, 2, device=dev),
                torch.full((b, 1), LABEL_PAD, dtype=torch.int32, device=dev),
            )
        mask_prompt: Tensor | None = None
        if mask_inputs is not None:
            size = self.sam_prompt_encoder.mask_input_size
            mask_prompt = mask_inputs.float()
            if tuple(mask_prompt.shape[-2:]) != size:
                mask_prompt = F.interpolate(
                    mask_prompt, size=size, mode="bilinear", align_corners=False, antialias=True
                )
        sparse, dense = self.sam_prompt_encoder(points=points, boxes=None, masks=mask_prompt)
        low_res_multimasks, ious, sam_tokens, object_score_logits = self.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask_output,
            high_res_features=high_res_feats,
        )
        is_obj = object_score_logits > 0  # [B, 1]
        low_res_multimasks = torch.where(
            is_obj[:, None, None], low_res_multimasks, NO_OBJ_SCORE
        ).float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        token = sam_tokens[:, 0]
        if multimask_output:
            best = torch.argmax(ious, dim=-1)
            rows = torch.arange(b, device=dev)
            low_res_masks = low_res_multimasks[rows, best].unsqueeze(1)
            high_res_masks = high_res_multimasks[rows, best].unsqueeze(1)
            if sam_tokens.shape[1] > 1:
                token = sam_tokens[rows, best]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks
        obj_ptr = self._gate_obj_ptr(self.obj_ptr_proj(token), is_obj.float())
        return SamOutputs(
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _gate_obj_ptr(self, obj_ptr: Tensor, is_obj: Tensor) -> Tensor:
        """Hard switch to the learned no-object pointer when the object is absent."""
        return is_obj * obj_ptr + (1 - is_obj) * self.no_obj_ptr

    def mask_as_output(
        self, image_embed: Tensor, high_res_feats: list[Tensor], mask_inputs: Tensor
    ) -> SamOutputs:
        """A ground-truth mask prompt ``[B, 1, 1024, 1024]`` is taken as the prediction.

        The binary mask is mapped to confident logits (+10 / -10) instead of being
        re-predicted; the SAM head still runs on the mask (as a dense prompt) only to
        produce an object pointer for the memory bank.
        """
        scale, bias = 20.0, -10.0
        mask = mask_inputs.float()
        high_res = mask * scale + bias
        low_res = F.interpolate(
            high_res,
            size=(high_res.shape[-2] // 4, high_res.shape[-1] // 4),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        ious = mask.new_ones(mask.shape[0], 1)
        sam = self.sam_heads(image_embed, high_res_feats, mask_inputs=self.mask_downsample(mask))
        is_obj = torch.any(mask.flatten(1) > 0.0, dim=1)[:, None].float()
        object_score_logits = scale * is_obj + bias
        obj_ptr = self._gate_obj_ptr(sam.obj_ptr, is_obj)
        return SamOutputs(low_res, high_res, ious, low_res, high_res, obj_ptr, object_score_logits)

    # ------------------------------------------------------------------ memory

    def condition_on_memory(
        self,
        frame_idx: int,
        is_init_cond_frame: bool,
        vision_feat: Tensor,
        vision_pos: Tensor,
        output_dict: OutputDict,
        num_frames: int,
        track_in_reverse: bool = False,
    ) -> Tensor:
        """Image tokens ``[HW, B, C]`` -> memory-conditioned embedding ``[B, C, H, W]``.

        The memory bank is (a) every conditioning frame, (b) up to ``num_maskmem - 1``
        of the most recent tracked frames and (c) object pointers of the conditioning
        frames plus up to ``max_obj_ptrs_in_encoder - 1`` recent tracked frames.
        """
        hw, b, c = vision_feat.shape
        side = int(round(hw**0.5))
        if side * side != hw:
            raise ValueError("the image embedding grid must be square")
        if is_init_cond_frame:
            # Nothing to remember yet: the learned embedding stands in for memory.
            return tokens_to_map(vision_feat + self.no_mem_embed, (side, side))

        cfg = self.cfg
        dev = vision_feat.device
        cond_outputs = output_dict[COND]
        non_cond = output_dict[NON_COND]
        if not cond_outputs:
            raise ValueError("tracking a frame needs at least one conditioning frame")
        selected, unselected = select_closest_cond_frames(
            frame_idx, cond_outputs, cfg.max_cond_frames_in_attn
        )

        # (a) + (b): spatial memories with their temporal slot t_pos (0 = conditioning).
        t_pos_and_prevs: list[tuple[int, FrameOutput | None]] = [
            (0, out) for out in selected.values()
        ]
        stride = cfg.memory_temporal_stride_for_eval
        for t_pos in range(1, self.num_maskmem):
            t_rel = self.num_maskmem - t_pos  # frames back from the current one
            if t_rel == 1:
                prev = frame_idx + 1 if track_in_reverse else frame_idx - 1
            elif track_in_reverse:
                prev = -(-(frame_idx + 2) // stride) * stride + (t_rel - 2) * stride
            else:
                prev = ((frame_idx - 2) // stride) * stride - (t_rel - 2) * stride
            out = non_cond.get(prev)
            if out is None:
                out = unselected.get(prev)
            t_pos_and_prevs.append((t_pos, out))

        memories: list[Tensor] = []
        memory_pos: list[Tensor] = []
        for t_pos, prev_out in t_pos_and_prevs:
            if prev_out is None or prev_out.maskmem_features is None:
                continue
            if prev_out.maskmem_pos_enc is None:
                raise ValueError("memory features without a position encoding")
            feats = prev_out.maskmem_features.to(dev, torch.float32)
            memories.append(feats.flatten(2).permute(2, 0, 1))
            pos = prev_out.maskmem_pos_enc[-1].to(dev).flatten(2).permute(2, 0, 1)
            row = self.cfg.maskmem_tpos_rows - 1 if t_pos == 0 else self.num_maskmem - t_pos - 1
            memory_pos.append(pos + self.maskmem_tpos_enc[row])

        # (c): object pointers, each with a signed temporal distance to this frame.
        max_ptrs = min(num_frames, cfg.max_obj_ptrs_in_encoder)
        sign = -1 if track_in_reverse else 1
        if cfg.only_obj_ptrs_in_the_past_for_eval:
            ptr_cond = {
                t: o
                for t, o in selected.items()
                if (t >= frame_idx if track_in_reverse else t <= frame_idx)
            }
        else:
            ptr_cond = selected
        pos_and_ptrs: list[tuple[int, Tensor]] = [
            ((frame_idx - t) * sign, o.obj_ptr) for t, o in ptr_cond.items()
        ]
        for t_diff in range(1, max_ptrs):
            t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
            if t < 0 or t >= num_frames:
                break
            out = non_cond.get(t, unselected.get(t))
            if out is not None:
                pos_and_ptrs.append((t_diff, out.obj_ptr))

        num_obj_ptr_tokens = 0
        if pos_and_ptrs and cfg.use_obj_ptrs_in_encoder:
            obj_ptrs = torch.stack([p.to(dev) for _, p in pos_and_ptrs], dim=0)  # [N, B, C]
            # Distances are normalised by the window length before the sine encoding.
            t_diffs = torch.tensor([d for d, _ in pos_and_ptrs], device=dev)
            obj_pos = sine_pe_1d(t_diffs / (max_ptrs - 1), dim=c)
            obj_pos = self.obj_ptr_tpos_proj(obj_pos).unsqueeze(1).expand(-1, b, self.mem_dim)
            if self.mem_dim < c:
                # A 256-d pointer becomes 4 memory tokens of 64 channels each.
                split = c // self.mem_dim
                obj_ptrs = obj_ptrs.reshape(-1, b, split, self.mem_dim)
                obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                obj_pos = obj_pos.repeat_interleave(split, dim=0)
            memories.append(obj_ptrs)
            memory_pos.append(obj_pos)
            num_obj_ptr_tokens = obj_ptrs.shape[0]

        conditioned = self.memory_attention(
            curr=vision_feat,
            curr_pos=vision_pos,
            memory=torch.cat(memories, dim=0),
            memory_pos=torch.cat(memory_pos, dim=0),
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        return tokens_to_map(conditioned, (side, side))

    def encode_memory(
        self,
        vision_feat: Tensor,
        high_res_mask_logits: Tensor,
        object_score_logits: Tensor,
        is_mask_from_pts: bool,
    ) -> tuple[Tensor, list[Tensor]]:
        """Memory-bank entry for a frame from its raw image tokens ``[HW, B, C]`` (not the
        memory-conditioned ones) and its predicted ``[B, 1, 1024, 1024]`` mask logits.
        """
        side = int(round(vision_feat.shape[0] ** 0.5))
        pix_feat = tokens_to_map(vision_feat, (side, side))
        if self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts:
            mask = (high_res_mask_logits > 0).float()
        else:
            mask = torch.sigmoid(high_res_mask_logits)
        # Rescale the soft mask to logit-like range; the encoder was trained this way.
        mask = mask * self.cfg.sigmoid_scale_for_mem_enc + self.cfg.sigmoid_bias_for_mem_enc
        out = self.memory_encoder(pix_feat, mask, skip_mask_sigmoid=True)
        features = cast(Tensor, out["vision_features"])
        pos = cast(list[Tensor], out["vision_pos_enc"])
        if self.cfg.no_obj_embed_spatial:
            is_obj = (object_score_logits > 0).float()
            no_obj = self.no_obj_embed_spatial[..., None, None]
            features = features + (1 - is_obj[..., None, None]) * no_obj
        return features, pos

    def track_step(
        self,
        frame_idx: int,
        is_init_cond_frame: bool,
        feats: ImageFeatures,
        points: tuple[Tensor, Tensor] | None,
        mask_inputs: Tensor | None,
        output_dict: OutputDict,
        num_frames: int,
        track_in_reverse: bool = False,
        run_mem_encoder: bool = True,
        prev_sam_mask_logits: Tensor | None = None,
    ) -> FrameOutput:
        """Segment one frame and (optionally) encode it into memory.

        A mask prompt bypasses the SAM head; otherwise the embedding is conditioned on
        memory (or gets ``no_mem_embed`` on an initial conditioning frame) and the SAM
        head runs with the given points, or with no prompt at all while tracking.
        ``prev_sam_mask_logits`` (an earlier prediction for this frame) is fed as a
        dense prompt together with new points so that a click refines the mask instead
        of starting over.
        """
        if mask_inputs is not None and self.cfg.use_mask_input_as_output_without_sam:
            sam = self.mask_as_output(feats.image_embed, feats.high_res_feats, mask_inputs)
        else:
            pix_feat = self.condition_on_memory(
                frame_idx,
                is_init_cond_frame,
                feats.vision_feat,
                feats.vision_pos,
                output_dict,
                num_frames,
                track_in_reverse,
            )
            if prev_sam_mask_logits is not None:
                if points is None or mask_inputs is not None:
                    raise ValueError("prev_sam_mask_logits only refine a point prompt")
                mask_inputs = prev_sam_mask_logits
            num_points = 0 if points is None else int(points[1].shape[1])
            sam = self.sam_heads(
                pix_feat,
                feats.high_res_feats,
                points=points,
                mask_inputs=mask_inputs,
                multimask_output=self._use_multimask(is_init_cond_frame, num_points),
            )
        out = FrameOutput(
            pred_masks=sam.low_res_masks,
            pred_masks_high_res=sam.high_res_masks,
            obj_ptr=sam.obj_ptr,
            object_score_logits=sam.object_score_logits,
        )
        if run_mem_encoder and self.num_maskmem > 0:
            out.maskmem_features, out.maskmem_pos_enc = self.encode_memory(
                feats.vision_feat, sam.high_res_masks, sam.object_score_logits, points is not None
            )
        return out


def select_closest_cond_frames(
    frame_idx: int, cond_outputs: dict[int, FrameOutput], max_cond_frames: int
) -> tuple[dict[int, FrameOutput], dict[int, FrameOutput]]:
    """Split conditioning frames into those attended to and the rest.

    With ``max_cond_frames == -1`` (the released setting) every conditioning frame is
    selected. Otherwise the closest one before and the closest one after ``frame_idx``
    are kept, then the remaining slots go to the temporally nearest frames.
    """
    if max_cond_frames == -1 or len(cond_outputs) <= max_cond_frames:
        return dict(cond_outputs), {}
    if max_cond_frames < 2:
        raise ValueError("max_cond_frames must be -1 or at least 2")
    selected: dict[int, FrameOutput] = {}
    before = max((t for t in cond_outputs if t < frame_idx), default=None)
    if before is not None:
        selected[before] = cond_outputs[before]
    after = min((t for t in cond_outputs if t >= frame_idx), default=None)
    if after is not None:
        selected[after] = cond_outputs[after]
    remaining = max_cond_frames - len(selected)
    rest = sorted((t for t in cond_outputs if t not in selected), key=lambda t: abs(t - frame_idx))
    for t in rest[:remaining]:
        selected[t] = cond_outputs[t]
    unselected = {t: v for t, v in cond_outputs.items() if t not in selected}
    return selected, unselected


class ModelImagePredictor(ImagePredictor):
    """``ImagePredictor`` bound to a full model, so ``set_image`` needs no encoder."""

    def __init__(self, model: SAM2) -> None:
        cfg = model.cfg
        super().__init__(
            model.sam_prompt_encoder,
            model.sam_mask_decoder,
            image_size=cfg.image_size,
            pixel_mean=cfg.pixel_mean,
            pixel_std=cfg.pixel_std,
        )
        self.model = model.eval()

    @torch.no_grad()
    def set_image(
        self,
        image: np.ndarray,
        image_encoder_forward: ImageEncoderForward | None = None,
    ) -> None:
        super().set_image(image, image_encoder_forward or self.model.encode_for_image_predictor)
