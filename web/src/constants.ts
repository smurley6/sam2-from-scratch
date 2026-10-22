/**
 * Loop constants exported next to the graphs (`constants_{S}.json`).
 *
 * Tensor entries are float32 values written as float64 in the JSON, so `Float32Array.from`
 * restores them bit for bit. `pixel_mean` / `pixel_std` are plain float64 values and stay
 * JavaScript numbers: normalisation runs in float64, as in the Python reference.
 */
export interface Constants {
  readonly model: string;
  readonly image_size: number;
  readonly embed_grid: number;
  readonly mask_size: number;
  readonly hidden_dim: number;
  readonly mem_dim: number;
  readonly tokens_per_obj_ptr: number;
  readonly num_maskmem: number;
  readonly max_obj_ptrs_in_encoder: number;
  readonly max_cond_frames_in_attn: number;
  readonly memory_temporal_stride_for_eval: number;
  readonly only_obj_ptrs_in_the_past_for_eval: boolean;
  readonly pixel_mean: readonly [number, number, number];
  readonly pixel_std: readonly [number, number, number];
  readonly multimask_min_pt_num: number;
  readonly multimask_max_pt_num: number;
  readonly stability_delta: number;
  readonly stability_thresh: number;
  readonly no_obj_score: number;
  readonly mask_input_clamp: number;
  readonly sigmoid_scale_for_mem_enc: number;
  readonly sigmoid_bias_for_mem_enc: number;
  /** `[hidden_dim]`; `vision_feat = image_embed - no_mem_embed` per channel. */
  readonly no_mem_embed: Float32Array;
  /** `num_maskmem` rows of `[mem_dim]`: temporal slot encodings of spatial memories. */
  readonly maskmem_tpos_enc: readonly Float32Array[];
  /** `[hidden_dim]`; baked into `object_pointer`, kept for auditing. */
  readonly no_obj_ptr: Float32Array;
  /** `[mem_dim]`; baked into `memory_encoder`, kept for auditing. */
  readonly no_obj_embed_spatial: Float32Array;
  /** `[16]`, row-major 4x4: `SAM2.mask_downsample`, the stride-4 conv a mask prompt goes through. */
  readonly mask_downsample_weight: Float32Array;
  readonly mask_downsample_bias: number;
}

type Json = Record<string, unknown>;

function num(raw: Json, key: string): number {
  const v = raw[key];
  if (typeof v !== "number") throw new Error(`constants: "${key}" must be a number`);
  return v;
}

function vec(raw: Json, key: string, length?: number): Float32Array {
  const v = raw[key];
  if (!Array.isArray(v) || !v.every((x) => typeof x === "number")) {
    throw new Error(`constants: "${key}" must be an array of numbers`);
  }
  if (length !== undefined && v.length !== length) {
    throw new Error(`constants: "${key}" has length ${v.length}, expected ${length}`);
  }
  return Float32Array.from(v as number[]);
}

function triple(raw: Json, key: string): readonly [number, number, number] {
  const v = raw[key];
  if (!Array.isArray(v) || v.length !== 3 || !v.every((x) => typeof x === "number")) {
    throw new Error(`constants: "${key}" must be three numbers`);
  }
  return [v[0] as number, v[1] as number, v[2] as number];
}

/** One level of nesting removed (`[[...], [...]]` -> `[...]`); anything else is left to `vec` to reject. */
function flatten(v: unknown): unknown {
  return Array.isArray(v) && v.every((row) => Array.isArray(row)) ? v.flat() : v;
}

/** Parse `constants_{S}.json` (an object from `JSON.parse` or the raw text). */
export function parseConstants(json: string | Json): Constants {
  const raw: Json = typeof json === "string" ? (JSON.parse(json) as Json) : json;
  const hidden = num(raw, "hidden_dim");
  const memDim = num(raw, "mem_dim");
  const nMem = num(raw, "num_maskmem");
  const tpos = raw["maskmem_tpos_enc"];
  if (!Array.isArray(tpos) || tpos.length !== nMem) {
    throw new Error(`constants: "maskmem_tpos_enc" must have ${nMem} rows`);
  }
  const tposRows = tpos.map((row, i) => vec({ [`maskmem_tpos_enc[${i}]`]: row }, `maskmem_tpos_enc[${i}]`, memDim));
  const onlyPast = raw["only_obj_ptrs_in_the_past_for_eval"];
  if (typeof onlyPast !== "boolean") {
    throw new Error('constants: "only_obj_ptrs_in_the_past_for_eval" must be a boolean');
  }
  return {
    model: String(raw["model"] ?? ""),
    image_size: num(raw, "image_size"),
    embed_grid: num(raw, "embed_grid"),
    mask_size: num(raw, "mask_size"),
    hidden_dim: hidden,
    mem_dim: memDim,
    tokens_per_obj_ptr: num(raw, "tokens_per_obj_ptr"),
    num_maskmem: nMem,
    max_obj_ptrs_in_encoder: num(raw, "max_obj_ptrs_in_encoder"),
    max_cond_frames_in_attn: num(raw, "max_cond_frames_in_attn"),
    memory_temporal_stride_for_eval: num(raw, "memory_temporal_stride_for_eval"),
    only_obj_ptrs_in_the_past_for_eval: onlyPast,
    pixel_mean: triple(raw, "pixel_mean"),
    pixel_std: triple(raw, "pixel_std"),
    multimask_min_pt_num: num(raw, "multimask_min_pt_num"),
    multimask_max_pt_num: num(raw, "multimask_max_pt_num"),
    stability_delta: num(raw, "stability_delta"),
    stability_thresh: num(raw, "stability_thresh"),
    no_obj_score: num(raw, "no_obj_score"),
    mask_input_clamp: num(raw, "mask_input_clamp"),
    sigmoid_scale_for_mem_enc: num(raw, "sigmoid_scale_for_mem_enc"),
    sigmoid_bias_for_mem_enc: num(raw, "sigmoid_bias_for_mem_enc"),
    no_mem_embed: vec(raw, "no_mem_embed", hidden),
    maskmem_tpos_enc: tposRows,
    no_obj_ptr: vec(raw, "no_obj_ptr", hidden),
    no_obj_embed_spatial: vec(raw, "no_obj_embed_spatial", memDim),
    mask_downsample_weight: vec({ mask_downsample_weight: flatten(raw["mask_downsample_weight"]) }, "mask_downsample_weight", 16),
    mask_downsample_bias: vec(raw, "mask_downsample_bias", 1)[0] as number,
  };
}
