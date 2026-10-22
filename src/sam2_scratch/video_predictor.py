"""Promptable video object segmentation: prompt some frames, propagate to the rest.

Usage::

    predictor = VideoPredictor(SAM2.from_checkpoint(path, TINY))
    state = predictor.init_state("data/videos/bedroom")
    predictor.add_points(state, frame_idx=0, obj_id=1, points=[[210, 350]], labels=[1])
    for frame_idx, low_res, video_res in predictor.propagate(state):
        ...

Every object is tracked independently with its own memory bank, one object per model
batch row. Prompted frames become conditioning frames for their object; their memory
is encoded lazily when propagation starts, so a user can click several times on a frame
and only the final mask enters the memory bank.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from sam2_scratch.model import COND, NON_COND, SAM2, FrameOutput, ImageFeatures, OutputDict
from sam2_scratch.prompt_encoder import LABEL_BOX_BR, LABEL_BOX_TL

FRAME_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass
class ObjectState:
    """Prompts and memory bank of one tracked object."""

    points: dict[int, tuple[Tensor, Tensor]] = field(default_factory=dict)  # model pixels
    boxes: dict[int, Tensor] = field(default_factory=dict)  # [1, 2, 2] model pixels
    mask_inputs: dict[int, Tensor] = field(default_factory=dict)  # [1, 1, S, S]
    output: OutputDict = field(default_factory=lambda: {COND: {}, NON_COND: {}})
    # Outputs of frames prompted since the last propagation; memory not yet encoded.
    pending: dict[int, FrameOutput] = field(default_factory=dict)
    # frame -> direction (True = reverse) in which the frame was last tracked
    tracked: dict[int, bool] = field(default_factory=dict)


@dataclass
class VideoState:
    images: Tensor  # [T, 3, S, S] normalised, kept on the CPU
    video_hw: tuple[int, int]
    objects: dict[int, ObjectState] = field(default_factory=dict)
    # Features of the most recently encoded frame: every object reads the same frame
    # in turn, so a one-frame cache removes all repeated backbone passes.
    feature_cache: tuple[int, ImageFeatures] | None = None
    # The memory position encoding is identical for every frame; keep one copy.
    maskmem_pos_enc: list[Tensor] | None = None

    @property
    def num_frames(self) -> int:
        return int(self.images.shape[0])

    @property
    def obj_ids(self) -> list[int]:
        return list(self.objects)


def load_frames(
    frames: str | Path | Sequence[np.ndarray],
    image_size: int,
    pixel_mean: tuple[float, float, float],
    pixel_std: tuple[float, float, float],
) -> tuple[Tensor, tuple[int, int]]:
    """Frames -> ``([T, 3, S, S]`` normalised float32, ``(H, W))``.

    Resizing goes through PIL (bicubic, aspect ratio not preserved) and the arithmetic
    order follows the reference loader (``uint8 / 255`` in float64, then float32
    normalisation), which makes the result bit-identical to it.
    """
    if isinstance(frames, (str, Path)):
        folder = Path(frames)
        paths = [p for p in folder.iterdir() if p.suffix.lower() in FRAME_SUFFIXES]
        if not paths:
            raise FileNotFoundError(f"no frames in {folder}")
        numeric = all(p.stem.isdigit() for p in paths)
        paths.sort(key=lambda p: (int(p.stem), p.name) if numeric else (0, p.name))
        pil = [Image.open(p).convert("RGB") for p in paths]
    else:
        if len(frames) == 0:
            raise ValueError("no frames given")
        pil = []
        for arr in frames:
            if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
                raise ValueError("frames must be HWC uint8 RGB arrays")
            pil.append(Image.fromarray(arr))
    width, height = pil[0].size
    if any(im.size != (width, height) for im in pil):
        raise ValueError("all frames must have the same size")

    images = torch.zeros(len(pil), 3, image_size, image_size, dtype=torch.float32)
    for i, im in enumerate(pil):
        resized = np.array(im.resize((image_size, image_size), Image.Resampling.BICUBIC))
        images[i] = torch.from_numpy(resized / 255.0).permute(2, 0, 1)
    mean = torch.tensor(pixel_mean, dtype=torch.float32)[:, None, None]
    std = torch.tensor(pixel_std, dtype=torch.float32)[:, None, None]
    images -= mean
    images /= std
    return images, (height, width)


def non_overlapping(logits: Tensor) -> Tensor:
    """Resolve overlaps between objects' masks ``[num_obj, 1, H, W]``.

    At every pixel only the object with the highest logit keeps its score; the others
    are clamped to at most -10 so they read as background there. Not applied inside the
    predictor (each object's memory sees its own unconstrained mask, as in the
    reference); it is meant for multi-object outputs such as benchmark submissions.
    """
    if logits.shape[0] <= 1:
        return logits
    winner = torch.argmax(logits, dim=0, keepdim=True)
    rows = torch.arange(logits.shape[0], device=logits.device)[:, None, None, None]
    return torch.where(winner == rows, logits, torch.clamp(logits, max=-10.0))


class VideoPredictor:
    """Stateless driver of ``SAM2.track_step`` over a ``VideoState``."""

    def __init__(self, model: SAM2, binarize_mask_from_pts: bool = True) -> None:
        self.model = model.eval()
        # The reference video predictor builds the model with
        # binarize_mask_from_pts_for_mem_enc=True: a prompted frame's mask enters the
        # memory as a hard 0/1 mask rather than as soft probabilities.
        self.model.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts
        self.image_size = model.image_size

    @property
    def device(self) -> torch.device:
        return self.model.device

    # ------------------------------------------------------------------ setup

    def init_state(self, frames: str | Path | Sequence[np.ndarray]) -> VideoState:
        """Load and normalise every frame (a folder of images or HWC uint8 arrays)."""
        cfg = self.model.cfg
        images, hw = load_frames(frames, self.image_size, cfg.pixel_mean, cfg.pixel_std)
        return VideoState(images=images, video_hw=hw)

    def init_state_from_tensor(self, images: Tensor, video_hw: tuple[int, int]) -> VideoState:
        """Start from already-normalised ``[T, 3, S, S]`` frames (e.g. a cached clip)."""
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(f"expected [T, 3, {self.image_size}, {self.image_size}] frames")
        return VideoState(
            images=images.float().cpu(), video_hw=(int(video_hw[0]), int(video_hw[1]))
        )

    def _object(self, state: VideoState, obj_id: int) -> ObjectState:
        if obj_id not in state.objects:
            state.objects[obj_id] = ObjectState()
        return state.objects[obj_id]

    def _features(self, state: VideoState, frame_idx: int) -> ImageFeatures:
        if state.feature_cache is not None and state.feature_cache[0] == frame_idx:
            return state.feature_cache[1]
        x = state.images[frame_idx].to(self.device).float().unsqueeze(0)
        feats = self.model.forward_image(x)
        state.feature_cache = (frame_idx, feats)
        return feats

    def _video_res(self, state: VideoState, low_res: Tensor) -> Tensor:
        return F.interpolate(low_res, size=state.video_hw, mode="bilinear", align_corners=False)

    def _store_memory(self, state: VideoState, out: FrameOutput) -> None:
        """Compact a frame output for the memory bank.

        Memory features are kept in bfloat16 (halving the bank's footprint), exactly as
        the reference does, so the values attended to later match it bit for bit. The
        high-resolution logits are dropped: only the low-res ones are ever reused.
        """
        if out.maskmem_features is not None:
            out.maskmem_features = out.maskmem_features.to(torch.bfloat16)
        if out.maskmem_pos_enc is not None:
            if state.maskmem_pos_enc is None:
                state.maskmem_pos_enc = [p[:1].clone() for p in out.maskmem_pos_enc]
            b = out.maskmem_pos_enc[0].shape[0]
            out.maskmem_pos_enc = [p.expand(b, -1, -1, -1) for p in state.maskmem_pos_enc]
        out.pred_masks_high_res = None

    # ------------------------------------------------------------------ prompts

    def _prior_output(self, obj: ObjectState, frame_idx: int) -> FrameOutput | None:
        """Most recent prediction for this frame, newest source first."""
        for source in (obj.pending, obj.output[COND], obj.output[NON_COND]):
            if frame_idx in source:
                return source[frame_idx]
        return None

    @torch.no_grad()
    def add_points(
        self,
        state: VideoState,
        frame_idx: int,
        obj_id: int,
        points: np.ndarray | Sequence[Sequence[float]] | None,
        labels: np.ndarray | Sequence[int] | None,
        box: np.ndarray | Sequence[float] | None = None,
        clear_old_points: bool = True,
    ) -> tuple[int, Tensor, Tensor]:
        """Add clicks ``points [P, 2]`` (x, y in original frame pixels, labels 1/0) and/or
        a ``box`` (xyxy) for ``obj_id`` on ``frame_idx``.

        Clicks accumulate on the frame unless ``clear_old_points``; a frame holds at most
        one box, placed before the clicks as the decoder expects. Returns ``(frame_idx,
        low-res logits [1, 1, 256, 256], video-res logits [1, 1, H, W])``.
        """
        if not 0 <= frame_idx < state.num_frames:
            raise IndexError(f"frame {frame_idx} out of range")
        obj = self._object(state, obj_id)
        h, w = state.video_hw
        scale = torch.tensor([w, h], dtype=torch.float32)
        if points is None:
            coords = torch.zeros(1, 0, 2)
            labs = torch.zeros(1, 0, dtype=torch.int32)
        else:
            if labels is None:
                raise ValueError("labels are required with points")
            coords = torch.as_tensor(np.asarray(points), dtype=torch.float32).reshape(1, -1, 2)
            labs = torch.as_tensor(np.asarray(labels), dtype=torch.int32).reshape(1, -1)
            if coords.shape[1] != labs.shape[1]:
                raise ValueError("points and labels differ in length")
        coords = coords / scale * self.image_size

        if clear_old_points:
            obj.points[frame_idx] = (coords, labs)
            obj.boxes.pop(frame_idx, None)
        else:
            old_c, old_l = obj.points.get(frame_idx, (coords[:, :0], labs[:, :0]))
            obj.points[frame_idx] = (torch.cat([old_c, coords], 1), torch.cat([old_l, labs], 1))
        if box is not None:
            box_t = torch.as_tensor(np.asarray(box), dtype=torch.float32).reshape(1, 2, 2)
            obj.boxes[frame_idx] = box_t / scale * self.image_size
        obj.mask_inputs.pop(frame_idx, None)

        p_coords, p_labels = obj.points[frame_idx]
        if frame_idx in obj.boxes:
            box_labels = torch.tensor([[LABEL_BOX_TL, LABEL_BOX_BR]], dtype=torch.int32)
            p_coords = torch.cat([obj.boxes[frame_idx], p_coords], 1)
            p_labels = torch.cat([box_labels, p_labels], 1)
        if p_coords.shape[1] == 0:
            raise ValueError("a point prompt needs at least one point or a box")
        prompt = (p_coords.to(self.device), p_labels.to(self.device))

        prior = self._prior_output(obj, frame_idx)
        prev_logits = None if prior is None else torch.clamp(prior.pred_masks, -32.0, 32.0)
        return self._run_prompt(state, obj, frame_idx, prompt, None, prev_logits)

    @torch.no_grad()
    def add_mask(
        self, state: VideoState, frame_idx: int, obj_id: int, mask: np.ndarray | Tensor
    ) -> tuple[int, Tensor, Tensor]:
        """Prompt ``obj_id`` on ``frame_idx`` with a binary ``mask [H, W]``; any clicks on
        the frame are discarded. Returns as ``add_points``."""
        if not 0 <= frame_idx < state.num_frames:
            raise IndexError(f"frame {frame_idx} out of range")
        obj = self._object(state, obj_id)
        m = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
        if m.ndim != 2:
            raise ValueError("mask must be [H, W]")
        m_in = m[None, None].float().to(self.device)
        s = self.image_size
        if tuple(m_in.shape[-2:]) != (s, s):
            m_in = F.interpolate(
                m_in, size=(s, s), mode="bilinear", align_corners=False, antialias=True
            )
            m_in = (m_in >= 0.5).float()
        obj.mask_inputs[frame_idx] = m_in
        obj.points.pop(frame_idx, None)
        obj.boxes.pop(frame_idx, None)
        return self._run_prompt(state, obj, frame_idx, None, m_in, None)

    def _run_prompt(
        self,
        state: VideoState,
        obj: ObjectState,
        frame_idx: int,
        points: tuple[Tensor, Tensor] | None,
        mask_inputs: Tensor | None,
        prev_logits: Tensor | None,
    ) -> tuple[int, Tensor, Tensor]:
        # A frame that was already tracked is a correction: its prediction is still
        # conditioned on the memory bank, in the direction it was tracked.
        is_init = frame_idx not in obj.tracked
        reverse = False if is_init else obj.tracked[frame_idx]
        out = self.model.track_step(
            frame_idx,
            is_init,
            self._features(state, frame_idx),
            points,
            mask_inputs,
            obj.output,
            state.num_frames,
            track_in_reverse=reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_logits,
        )
        out.pred_masks_high_res = None
        obj.pending[frame_idx] = out
        return frame_idx, out.pred_masks, self._video_res(state, out.pred_masks)

    # ------------------------------------------------------------------ propagation

    def _preflight(self, state: VideoState) -> None:
        """Encode memory for newly prompted frames and promote them to conditioning."""
        if not state.objects:
            raise ValueError("add a prompt before propagating")
        s = self.image_size
        for obj_id, obj in state.objects.items():
            for frame_idx, out in sorted(obj.pending.items()):
                if out.maskmem_features is None:
                    # Upsampling the stored low-res logits (rather than keeping the
                    # high-res ones) is what the reference feeds the memory encoder.
                    high_res = F.interpolate(
                        out.pred_masks.to(self.device),
                        size=(s, s),
                        mode="bilinear",
                        align_corners=False,
                    )
                    feats = self._features(state, frame_idx)
                    out.maskmem_features, out.maskmem_pos_enc = self.model.encode_memory(
                        feats.vision_feat, high_res, out.object_score_logits, True
                    )
                    self._store_memory(state, out)
                obj.output[COND][frame_idx] = out
                obj.output[NON_COND].pop(frame_idx, None)
            obj.pending.clear()
            if not obj.output[COND]:
                raise ValueError(f"object {obj_id} has no prompted frame")

    @torch.no_grad()
    def propagate(
        self,
        state: VideoState,
        start_frame_idx: int | None = None,
        max_frames: int | None = None,
        reverse: bool = False,
    ) -> Iterator[tuple[int, dict[int, Tensor], Tensor]]:
        """Track every object through the video, yielding per frame ``(frame_idx,
        {obj_id: low-res logits [1, 1, 256, 256]}, video-res logits [num_obj, 1, H, W])``.

        Starts at the earliest conditioning frame unless ``start_frame_idx`` is given and
        visits at most ``max_frames`` frames beyond the start. Conditioning frames are
        yielded with their prompted prediction and not re-tracked. Unlike the reference,
        a reverse pass that starts on frame 0 still yields frame 0 instead of nothing.
        """
        self._preflight(state)
        n = state.num_frames
        if start_frame_idx is None:
            start_frame_idx = min(t for o in state.objects.values() for t in o.output[COND])
        if max_frames is None:
            max_frames = n
        if reverse:
            end = max(start_frame_idx - max_frames, 0)
            order = range(start_frame_idx, end - 1, -1)
        else:
            end = min(start_frame_idx + max_frames, n - 1)
            order = range(start_frame_idx, end + 1)

        for frame_idx in order:
            low_res: dict[int, Tensor] = {}
            for obj_id, obj in state.objects.items():
                out = obj.output[COND].get(frame_idx)
                if out is None:
                    out = self.model.track_step(
                        frame_idx,
                        False,
                        self._features(state, frame_idx),
                        None,
                        None,
                        obj.output,
                        n,
                        track_in_reverse=reverse,
                        run_mem_encoder=True,
                    )
                    self._store_memory(state, out)
                    obj.output[NON_COND][frame_idx] = out
                obj.tracked[frame_idx] = reverse
                low_res[obj_id] = out.pred_masks
            video_res = self._video_res(state, torch.cat(list(low_res.values()), dim=0))
            yield frame_idx, low_res, video_res
