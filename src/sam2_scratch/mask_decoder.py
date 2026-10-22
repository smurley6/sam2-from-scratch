"""SAM 2 mask decoder: prompt tokens + image embedding to mask logits and scores.

Learned output tokens (object score, IoU, four mask tokens) are prepended to the prompt
tokens and run through the two-way transformer with the image embedding. The image side
is upsampled 4x with the stride-8 and stride-4 backbone features added as skips, and
each mask token is turned by a hypernetwork MLP into a 32-d kernel that is dotted with
every upsampled pixel to give that mask's logits.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sam2_scratch.common import MLP, LayerNorm2d
from sam2_scratch.transformer import TwoWayTransformer


class MaskDecoder(nn.Module):
    """Predicts masks, IoU scores, object-presence logits and the SAM output tokens.

    Mask token 0 is the single-mask output; tokens 1..3 are the multimask candidates
    used for ambiguous prompts. In eval mode a single-mask request whose output is
    unstable under small threshold shifts falls back to the best multimask candidate
    (``dynamic_multimask_via_stability``), which is how SAM 2 handles single-click
    prompts during tracking.
    """

    def __init__(
        self,
        transformer_dim: int = 256,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        transformer_depth: int = 2,
        transformer_heads: int = 8,
        transformer_mlp_dim: int = 2048,
        use_multimask_token_for_obj_ptr: bool = True,
        dynamic_multimask_via_stability: bool = True,
        dynamic_multimask_stability_delta: float = 0.05,
        dynamic_multimask_stability_thresh: float = 0.98,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer(
            depth=transformer_depth,
            embedding_dim=transformer_dim,
            num_heads=transformer_heads,
            mlp_dim=transformer_mlp_dim,
        )
        self.num_mask_tokens = num_multimask_outputs + 1
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
        # Project the stride-4 / stride-8 backbone maps to the upscaler's widths. The
        # image predictor and video loop apply these once per image, not per prompt.
        self.conv_s0 = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1)
        self.conv_s1 = nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1)

        self.output_hypernetworks_mlps = nn.ModuleList(
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)
        )
        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=True,
        )
        self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
        multimask_output: bool,
        high_res_features: list[Tensor],
        repeat_image: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``(masks [B, M, 256, 256], iou [B, M], tokens [B, M, C], obj [B, 1])``.

        ``M`` is 3 with ``multimask_output`` and 1 otherwise. ``high_res_features`` are
        ``[feat_s0 [B, 32, 256, 256], feat_s1 [B, 64, 128, 128]]``, already projected by
        ``conv_s0`` / ``conv_s1``. ``repeat_image`` broadcasts one image over a batch of
        prompts.
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            high_res_features,
            repeat_image,
        )

        if multimask_output:
            masks = masks[:, 1:]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1]
            iou_pred = iou_pred[:, 0:1]

        # The object pointer is taken from the multimask tokens only when multimask
        # outputs were requested; a single-mask call always returns token 0.
        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
        high_res_features: list[Tensor],
        repeat_image: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """All four mask outputs before single/multi selection."""
        # Token order matters: hs[:, 0] is the object score token, hs[:, 1] the IoU token.
        output_tokens = torch.cat(
            [self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        b = sparse_prompt_embeddings.shape[0]
        output_tokens = output_tokens.unsqueeze(0).expand(b, -1, -1)
        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)

        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, b, dim=0)
        else:
            if image_embeddings.shape[0] != b:
                raise ValueError("image batch must match prompt batch unless repeat_image")
            src = image_embeddings
        src = src + dense_prompt_embeddings
        if image_pe.shape[0] != 1:
            raise ValueError("image_pe must have batch size 1")
        pos_src = torch.repeat_interleave(image_pe, b, dim=0)
        _, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 1, :]
        mask_tokens_out = hs[:, 2 : 2 + self.num_mask_tokens, :]

        src = src.transpose(1, 2).reshape(b, c, h, w)
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        upscaled = act1(ln1(dc1(src) + feat_s1))
        upscaled = act2(dc2(upscaled) + feat_s0)

        hyper_in = torch.stack(
            [mlp(mask_tokens_out[:, i, :]) for i, mlp in enumerate(self.output_hypernetworks_mlps)],
            dim=1,
        )
        _, uc, uh, uw = upscaled.shape
        masks = (hyper_in @ upscaled.reshape(b, uc, uh * uw)).reshape(b, -1, uh, uw)

        iou_pred = self.iou_prediction_head(iou_token_out)
        object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits: Tensor) -> Tensor:
        """IoU between the masks thresholded at +delta and -delta (1.0 if both empty).

        A mask whose area barely changes when the threshold moves has confident logits.
        """
        mask_logits = mask_logits.flatten(-2)
        delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -delta, dim=-1).float()
        return torch.where(area_u > 0, area_i / area_u, 1.0)

    def _dynamic_multimask_via_stability(
        self, all_mask_logits: Tensor, all_iou_scores: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Keep the single-mask output if stable, else the highest-IoU multimask output."""
        multimask_logits = all_mask_logits[:, 1:]
        multimask_iou = all_iou_scores[:, 1:]
        best = torch.argmax(multimask_iou, dim=-1)
        batch = torch.arange(multimask_iou.shape[0], device=all_iou_scores.device)
        best_logits = multimask_logits[batch, best].unsqueeze(1)
        best_iou = multimask_iou[batch, best].unsqueeze(1)

        single_logits = all_mask_logits[:, 0:1]
        single_iou = all_iou_scores[:, 0:1]
        is_stable = self._get_stability_scores(single_logits) >= (
            self.dynamic_multimask_stability_thresh
        )
        mask_out = torch.where(is_stable[..., None, None], single_logits, best_logits)
        iou_out = torch.where(is_stable, single_iou, best_iou)
        return mask_out, iou_out
