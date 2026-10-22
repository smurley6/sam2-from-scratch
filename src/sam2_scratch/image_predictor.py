"""Promptable single-image segmentation on top of the SAM head.

The predictor owns the prompt encoder and mask decoder but not the image encoder: the
image features are either supplied directly (``set_features``) or produced by a callable
injected into ``set_image``. That keeps this module independent of the backbone and lets
the same code run on features from any source (reference dumps, the reimplemented
encoder, an exported graph).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from sam2_scratch.configs import TINY
from sam2_scratch.mask_decoder import MaskDecoder
from sam2_scratch.prompt_encoder import LABEL_BOX_BR, LABEL_BOX_TL, PromptEncoder

ImageEncoderForward = Callable[[Tensor], tuple[Tensor, list[Tensor]]]


class ImagePredictor:
    """Encode an image once, then predict masks for any number of prompts.

    Prompts are given in original-image pixel coordinates; outputs are returned at the
    original resolution as numpy arrays.
    """

    def __init__(
        self,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        image_size: int = 1024,
        mask_threshold: float = 0.0,
        pixel_mean: tuple[float, float, float] = TINY.pixel_mean,
        pixel_std: tuple[float, float, float] = TINY.pixel_std,
    ) -> None:
        self.prompt_encoder = prompt_encoder.eval()
        self.mask_decoder = mask_decoder.eval()
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.pixel_mean = torch.tensor(pixel_mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(pixel_std, dtype=torch.float32).view(1, 3, 1, 1)
        self._image_embed: Tensor | None = None
        self._high_res_feats: list[Tensor] | None = None
        self._orig_hw: tuple[int, int] | None = None

    @property
    def device(self) -> torch.device:
        return self.mask_decoder.iou_token.weight.device

    def preprocess(self, image: np.ndarray) -> Tensor:
        """``HWC uint8 RGB -> [1, 3, S, S]`` float, resized and ImageNet-normalised.

        The aspect ratio is not preserved: the model was trained on square-resized
        inputs. Antialiasing matters when downscaling large photos; without it the input
        drifts from the reference by more than the parity tolerance.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected an HWC RGB image, got shape {image.shape}")
        x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        x = x.to(torch.float32).div(255.0).unsqueeze(0)
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return ((x - self.pixel_mean) / self.pixel_std).to(self.device)

    @torch.no_grad()
    def set_image(self, image: np.ndarray, image_encoder_forward: ImageEncoderForward) -> None:
        """Preprocess ``image`` and compute its features with the injected encoder.

        ``image_encoder_forward`` maps ``[1, 3, S, S]`` to ``(image_embed [1, 256, 64,
        64], [feat_s0 [1, 32, 256, 256], feat_s1 [1, 64, 128, 128]])`` with the
        high-resolution maps already projected by ``conv_s0`` / ``conv_s1``.
        """
        image_embed, high_res_feats = image_encoder_forward(self.preprocess(image))
        self.set_features(image_embed, high_res_feats, (image.shape[0], image.shape[1]))

    def set_features(
        self, image_embed: Tensor, high_res_feats: list[Tensor], orig_hw: tuple[int, int]
    ) -> None:
        """Install precomputed features for an image of size ``orig_hw = (h, w)``."""
        if image_embed.ndim != 4 or image_embed.shape[0] != 1:
            raise ValueError("image_embed must be [1, C, H, W]")
        if len(high_res_feats) != 2:
            raise ValueError("high_res_feats must be [feat_s0, feat_s1]")
        self._image_embed = image_embed.to(self.device)
        self._high_res_feats = [f.to(self.device) for f in high_res_feats]
        self._orig_hw = (int(orig_hw[0]), int(orig_hw[1]))

    def reset(self) -> None:
        self._image_embed = None
        self._high_res_feats = None
        self._orig_hw = None

    def _to_model_coords(self, coords: Tensor, orig_hw: tuple[int, int]) -> Tensor:
        """Original-image pixels to model-input pixels, per axis (no aspect padding)."""
        h, w = orig_hw
        coords = coords.clone()
        coords[..., 0] = coords[..., 0] / w
        coords[..., 1] = coords[..., 1] / h
        return coords * self.image_size

    @torch.no_grad()
    def predict(
        self,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        mask_input: np.ndarray | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict masks for one prompt set on the current image.

        ``point_coords`` is ``[P, 2]`` (x, y) and ``point_labels`` ``[P]`` (1 positive,
        0 negative); ``box`` is ``[4]`` xyxy; ``mask_input`` is ``[1, 256, 256]`` logits
        from a previous call. Returns ``(masks [M, H, W], iou [M], low_res_logits [M,
        256, 256])`` where masks are boolean unless ``return_logits``.
        """
        if self._image_embed is None or self._high_res_feats is None or self._orig_hw is None:
            raise RuntimeError("call set_image or set_features before predict")
        orig_hw = self._orig_hw
        dev = self.device

        points: tuple[Tensor, Tensor] | None = None
        if point_coords is not None:
            if point_labels is None:
                raise ValueError("point_labels are required with point_coords")
            coords = torch.as_tensor(point_coords, dtype=torch.float32, device=dev)
            labels = torch.as_tensor(point_labels, dtype=torch.int32, device=dev)
            if coords.ndim != 2 or coords.shape[-1] != 2 or labels.shape != coords.shape[:1]:
                raise ValueError("point_coords must be [P, 2] and point_labels [P]")
            points = (self._to_model_coords(coords, orig_hw)[None], labels[None])

        if box is not None:
            # A box is two corner points with their own labels, placed before any clicks.
            # It goes through the point path, so the prompt encoder still appends its
            # padding token (the reference predictor does the same).
            box_t = torch.as_tensor(box, dtype=torch.float32, device=dev).reshape(1, 2, 2)
            box_coords = self._to_model_coords(box_t, orig_hw)
            box_labels = torch.tensor([[LABEL_BOX_TL, LABEL_BOX_BR]], dtype=torch.int32)
            box_labels = box_labels.to(dev)
            if points is not None:
                points = (
                    torch.cat([box_coords, points[0]], dim=1),
                    torch.cat([box_labels, points[1]], dim=1),
                )
            else:
                points = (box_coords, box_labels)

        masks_t: Tensor | None = None
        if mask_input is not None:
            masks_t = torch.as_tensor(mask_input, dtype=torch.float32, device=dev)
            if masks_t.ndim == 3:
                masks_t = masks_t[None]

        sparse, dense = self.prompt_encoder(points=points, boxes=None, masks=masks_t)
        low_res, iou, _, _ = self.mask_decoder(
            image_embeddings=self._image_embed,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask_output,
            high_res_features=self._high_res_feats,
        )

        masks = self.postprocess_masks(low_res, orig_hw)
        # Clamp the low-res logits returned for reuse as mask_input, so a feedback loop
        # cannot push extreme values into the mask downscaler.
        low_res = torch.clamp(low_res, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold
        return (
            masks[0].cpu().numpy(),
            iou[0].float().cpu().numpy(),
            low_res[0].float().cpu().numpy(),
        )

    @staticmethod
    def postprocess_masks(low_res: Tensor, orig_hw: tuple[int, int]) -> Tensor:
        """Upsample ``[B, M, 256, 256]`` logits to the original image size."""
        return F.interpolate(low_res, size=orig_hw, mode="bilinear", align_corners=False)
