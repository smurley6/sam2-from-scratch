"""Pinned SAM 2.1 hyperparameters, transcribed from Meta's released YAML configs.

Only the inference-relevant values are kept. Anything not listed here is a fixed
architectural constant written directly into the module that uses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HieraConfig:
    embed_dim: int = 96
    num_heads: int = 1
    stages: tuple[int, ...] = (1, 2, 7, 2)
    global_att_blocks: tuple[int, ...] = (5, 7, 9)
    window_pos_embed_bkg_spatial_size: tuple[int, int] = (7, 7)
    # Hiera defaults that the YAML does not override.
    patch_kernel: tuple[int, int] = (7, 7)
    patch_stride: tuple[int, int] = (4, 4)
    patch_padding: tuple[int, int] = (3, 3)
    q_pool: int = 3
    q_stride: tuple[int, int] = (2, 2)
    dim_mul: float = 2.0
    head_mul: float = 2.0
    window_spec: tuple[int, ...] = (8, 4, 14, 7)
    mlp_ratio: float = 4.0

    @property
    def channel_list(self) -> tuple[int, ...]:
        """Output channels per stage, highest resolution first (96, 192, 384, 768)."""
        return tuple(int(self.embed_dim * self.dim_mul**i) for i in range(len(self.stages)))


@dataclass(frozen=True)
class NeckConfig:
    d_model: int = 256
    backbone_channel_list: tuple[int, ...] = (768, 384, 192, 96)
    fpn_top_down_levels: tuple[int, ...] = (2, 3)
    pos_num_feats: int = 256
    pos_temperature: int = 10000
    scalp: int = 1  # drop the lowest-resolution level after the neck


@dataclass(frozen=True)
class MemoryAttentionConfig:
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 1
    dim_feedforward: int = 2048
    kv_in_dim: int = 64
    rope_theta: float = 10000.0
    feat_sizes: tuple[int, int] = (64, 64)
    pos_enc_at_input: bool = True
    pos_enc_at_cross_attn_keys: bool = True
    pos_enc_at_cross_attn_queries: bool = False
    rope_k_repeat_in_cross_attn: bool = True


@dataclass(frozen=True)
class MemoryEncoderConfig:
    out_dim: int = 64
    in_dim: int = 256
    pos_num_feats: int = 64
    pos_temperature: int = 10000
    mask_down_kernel: int = 3
    mask_down_stride: int = 2
    mask_down_padding: int = 1
    mask_down_total_stride: int = 16
    fuser_layers: int = 2
    fuser_kernel: int = 7
    fuser_padding: int = 3


@dataclass(frozen=True)
class SAM2Config:
    name: str
    hiera: HieraConfig
    neck: NeckConfig = field(default_factory=NeckConfig)
    memory_attention: MemoryAttentionConfig = field(default_factory=MemoryAttentionConfig)
    memory_encoder: MemoryEncoderConfig = field(default_factory=MemoryEncoderConfig)
    image_size: int = 1024
    backbone_stride: int = 16
    hidden_dim: int = 256
    mem_dim: int = 64
    num_maskmem: int = 7
    # Rows of the trained temporal-slot embedding; fixed by the checkpoint. num_maskmem may
    # be lowered for ablations without changing this.
    maskmem_tpos_rows: int = 7
    max_cond_frames_in_attn: int = -1
    max_obj_ptrs_in_encoder: int = 16
    memory_temporal_stride_for_eval: int = 1
    sigmoid_scale_for_mem_enc: float = 20.0
    sigmoid_bias_for_mem_enc: float = -10.0
    binarize_mask_from_pts_for_mem_enc: bool = False
    use_mask_input_as_output_without_sam: bool = True
    directly_add_no_mem_embed: bool = True
    no_obj_embed_spatial: bool = True
    use_high_res_features_in_sam: bool = True
    multimask_output_in_sam: bool = True
    multimask_min_pt_num: int = 0
    multimask_max_pt_num: int = 1
    multimask_output_for_tracking: bool = True
    use_multimask_token_for_obj_ptr: bool = True
    iou_prediction_use_sigmoid: bool = True
    use_obj_ptrs_in_encoder: bool = True
    add_tpos_enc_to_obj_ptrs: bool = True
    proj_tpos_enc_in_obj_ptrs: bool = True
    use_signed_tpos_enc_to_obj_ptrs: bool = True
    only_obj_ptrs_in_the_past_for_eval: bool = True
    pred_obj_scores: bool = True
    pred_obj_scores_mlp: bool = True
    fixed_no_obj_ptr: bool = True
    soft_no_obj_ptr: bool = False
    use_mlp_for_obj_ptr_proj: bool = True
    # Mask decoder / prompt encoder constants shared by every SAM 2 size.
    prompt_embed_dim: int = 256
    mask_in_chans: int = 16
    decoder_depth: int = 2
    decoder_heads: int = 8
    decoder_mlp_dim: int = 2048
    num_multimask_outputs: int = 3
    iou_head_depth: int = 3
    iou_head_hidden_dim: int = 256
    # ImageNet normalisation used by the reference transforms.
    pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @property
    def image_embedding_size(self) -> int:
        return self.image_size // self.backbone_stride


TINY = SAM2Config(name="tiny", hiera=HieraConfig())
SMALL = SAM2Config(
    name="small",
    hiera=HieraConfig(stages=(1, 2, 11, 2), global_att_blocks=(7, 10, 13)),
)

CONFIGS: dict[str, SAM2Config] = {"tiny": TINY, "small": SMALL}
