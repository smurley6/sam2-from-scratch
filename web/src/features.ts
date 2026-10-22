import type { Constants } from "./constants";
import { f32, runGraph, type ImageSessionSet, type Timings } from "./sessions";

/** Image encoder outputs of one frame (flat CHW, batch 1). */
export interface ImageFeatures {
  /** `[256, g, g]` stride-16 level with `no_mem_embed` added: the head's input. */
  readonly imageEmbed: Float32Array;
  /** `[32, 4g, 4g]` stride-4 skip features. */
  readonly highResS0: Float32Array;
  /** `[64, 2g, 2g]` stride-8 skip features. */
  readonly highResS1: Float32Array;
  /** Embedding grid side `g = S / 16`. */
  readonly grid: number;
}

/** Plus the raw tokens the memory graphs take. */
export interface FrameFeatures extends ImageFeatures {
  /** `[256, g, g]` raw level: `image_embed - no_mem_embed[c]`. */
  readonly visionFeat: Float32Array;
  /** `[g*g, 256]` raw tokens, row-major over the grid (transpose of `visionFeat`). */
  readonly curr: Float32Array;
}

/** Run the image encoder on a normalised `[3, S, S]` input. */
export async function encodeImage(
  sessions: Pick<ImageSessionSet, "imageEncoder">,
  input: Float32Array,
  size: number,
  timings?: Timings,
): Promise<ImageFeatures> {
  if (input.length !== 3 * size * size) {
    throw new Error(`encodeImage: expected ${3 * size * size} values, got ${input.length}`);
  }
  const out = await runGraph(
    sessions.imageEncoder,
    { image: f32(input, [1, 3, size, size]) },
    timings,
    "image_encoder",
  );
  return {
    imageEmbed: out["image_embed"] as Float32Array,
    highResS0: out["high_res_feat_s0"] as Float32Array,
    highResS1: out["high_res_feat_s1"] as Float32Array,
    grid: size / 16,
  };
}

/** Add the raw stride-16 map and its token transpose (recipe step 3). */
export function withMemoryTokens(
  feats: ImageFeatures,
  consts: Pick<Constants, "no_mem_embed">,
): FrameFeatures {
  const hw = feats.grid * feats.grid;
  const channels = consts.no_mem_embed.length;
  const visionFeat = new Float32Array(channels * hw);
  const curr = new Float32Array(channels * hw);
  for (let c = 0; c < channels; c++) {
    const e = consts.no_mem_embed[c] as number;
    const base = c * hw;
    for (let t = 0; t < hw; t++) {
      const v = (feats.imageEmbed[base + t] as number) - e;
      visionFeat[base + t] = v;
      curr[t * channels + c] = v;
    }
  }
  return { ...feats, visionFeat, curr };
}
