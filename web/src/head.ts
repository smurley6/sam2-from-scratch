import type { Constants } from "./constants";
import type { ImageFeatures } from "./features";
import { f32, i32, runGraph, type ImageSessionSet, type Timings } from "./sessions";

export const LABEL_PAD = -1;
const NUM_TOKENS = 4;

/** Raw `sam_head` outputs (batch dimension dropped). */
export interface HeadOutput {
  /** `[4, m, m]` logits of all four mask tokens. */
  readonly masks: Float32Array;
  /** `[4]` predicted IoU (already sigmoid). */
  readonly ious: Float32Array;
  /** `[4, 256]` mask-token outputs (object pointer inputs). */
  readonly tokens: Float32Array;
  /** Object score logit; the object is present when > 0. */
  readonly score: number;
  /** Mask side `m = S / 4`. */
  readonly maskSize: number;
}

export interface Candidate {
  /** `[m, m]` logits (a view into the head output). */
  readonly mask: Float32Array;
  readonly iou: number;
}

export interface Selection {
  /** `[m, m]` chosen logits; `no_obj_score` everywhere when the object is absent. */
  readonly lowRes: Float32Array;
  /** Predicted IoU of the chosen mask. */
  readonly iou: number;
  readonly score: number;
  /** `[256]` mask token for the object pointer (selection rule 3). */
  readonly token: Float32Array;
  /** Token index of the chosen mask (0 single, 1..3 multimask). */
  readonly index: number;
  /** Whether the multimask rule applied (0 or 1 prompt points). */
  readonly multimask: boolean;
  /** The three multimask candidates (tokens 1..3) for the image UI. */
  readonly candidates: readonly [Candidate, Candidate, Candidate];
}

/**
 * Run `sam_head`. `pointCoordsS` are interleaved (x, y) in S-pixel space, `labels` one per
 * point (-1 pad, 0 negative, 1 positive, 2 / 3 box corners). The graph appends its own
 * padding point. `maskInput` is a previous low-res output, already clamped, or null.
 */
export async function runHead(
  sessions: Pick<ImageSessionSet, "samHead">,
  feats: ImageFeatures,
  pointCoordsS: ArrayLike<number>,
  labels: ArrayLike<number>,
  maskInput: Float32Array | null,
  timings?: Timings,
): Promise<HeadOutput> {
  const g = feats.grid;
  const m = 4 * g;
  const p = labels.length;
  if (pointCoordsS.length !== 2 * p || p === 0) {
    throw new Error(`runHead: need 2 coordinates per label and at least one point`);
  }
  if (maskInput !== null && maskInput.length !== m * m) {
    throw new Error(`runHead: mask input must have ${m * m} values`);
  }
  const out = await runGraph(
    sessions.samHead,
    {
      image_embed: f32(feats.imageEmbed, [1, 256, g, g]),
      high_res_feat_s0: f32(feats.highResS0, [1, 32, 4 * g, 4 * g]),
      high_res_feat_s1: f32(feats.highResS1, [1, 64, 2 * g, 2 * g]),
      point_coords: f32(Float32Array.from(pointCoordsS), [1, p, 2]),
      point_labels: i32(Int32Array.from(labels), [1, p]),
      mask_input: f32(maskInput ?? new Float32Array(m * m), [1, 1, m, m]),
      has_mask_input: f32(Float32Array.of(maskInput === null ? 0 : 1), [1]),
    },
    timings,
    "sam_head",
  );
  return {
    masks: out["masks"] as Float32Array,
    ious: out["iou_predictions"] as Float32Array,
    tokens: out["sam_tokens"] as Float32Array,
    score: (out["object_score_logits"] as Float32Array)[0] as number,
    maskSize: m,
  };
}

/** `count(logit > delta) / count(logit > -delta)` in float32; 1 when the denominator is 0. */
export function stabilityScore(logits: Float32Array, delta: number): number {
  let inter = 0;
  let union = 0;
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i] as number;
    if (v > delta) inter++;
    if (v > -delta) union++;
  }
  return union === 0 ? 1 : Math.fround(inter / union);
}

/**
 * Output selection of the recipe (step 6). Multimask when `numPoints` (a box counts 2) is
 * within `[multimask_min_pt_num, multimask_max_pt_num]`: the first argmax of
 * `ious[1..3]` and its token. Otherwise token 0's mask if stable, else the best
 * multimask candidate (mask and IoU); the token is always `tokens[0]`. When
 * `score <= 0` the chosen mask becomes `no_obj_score` everywhere.
 */
export function selectOutput(
  masks: Float32Array,
  ious: Float32Array,
  tokens: Float32Array,
  score: number,
  numPoints: number,
  consts: Pick<
    Constants,
    | "multimask_min_pt_num"
    | "multimask_max_pt_num"
    | "stability_delta"
    | "stability_thresh"
    | "no_obj_score"
  >,
): Selection {
  const area = masks.length / NUM_TOKENS;
  const dim = tokens.length / NUM_TOKENS;
  const mask = (k: number): Float32Array => masks.subarray(k * area, (k + 1) * area);
  const token = (k: number): Float32Array => tokens.subarray(k * dim, (k + 1) * dim);
  let best = 1;
  for (let k = 2; k < NUM_TOKENS; k++) {
    if ((ious[k] as number) > (ious[best] as number)) best = k;
  }
  const multimask =
    consts.multimask_min_pt_num <= numPoints && numPoints <= consts.multimask_max_pt_num;
  let index: number;
  let tokenIndex: number;
  if (multimask) {
    index = best;
    tokenIndex = best;
  } else {
    const stable = stabilityScore(mask(0), consts.stability_delta) >= consts.stability_thresh;
    index = stable ? 0 : best;
    tokenIndex = 0;
  }
  const lowRes = score > 0 ? mask(index).slice() : new Float32Array(area).fill(consts.no_obj_score);
  return {
    lowRes,
    iou: ious[index] as number,
    score,
    token: token(tokenIndex),
    index,
    multimask,
    candidates: [
      { mask: mask(1), iou: ious[1] as number },
      { mask: mask(2), iou: ious[2] as number },
      { mask: mask(3), iou: ious[3] as number },
    ],
  };
}

/** Clamp a previous low-res output for use as `mask_input` (to +-`mask_input_clamp`). */
export function clampMaskInput(lowRes: Float32Array, clamp: number): Float32Array {
  const out = new Float32Array(lowRes.length);
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.max(-clamp, Math.min(clamp, lowRes[i] as number));
  }
  return out;
}
