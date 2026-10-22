import type { Constants } from "./constants";
import { encodeImage, withMemoryTokens, type FrameFeatures } from "./features";
import { clampMaskInput, LABEL_PAD, runHead, selectOutput } from "./head";
import type { Box, Point } from "./imagePredictor";
import { maskDownsample, resizeAntialias, roundBf16, upsampleBilinear } from "./preprocess";
import { f32, i64, runGraph, type SessionSet, type Timings } from "./sessions";

/** Normalised `[3, S, S]` frames, decoded on demand. */
export interface FrameSource {
  readonly count: number;
  get(i: number): Promise<Float32Array>;
}

/** What the memory bank keeps for one frame. */
interface FrameOutput {
  /** `[m, m]` chosen low-res logits (`no_obj_score` everywhere if absent). */
  readonly predMasks: Float32Array;
  /** `[256]` object pointer. */
  readonly objPtr: Float32Array;
  readonly score: number;
  /** `[64, g, g]` bfloat16-rounded memory features, once encoded. */
  maskmemFeatures: Float32Array | null;
}

export interface TrackedFrame {
  readonly frame: number;
  /** `[m, m]` low-res logits; threshold at 0 after upsampling for display. */
  readonly lowRes: Float32Array;
  /** Object score logit; the object is occluded / absent when <= 0. */
  readonly score: number;
  readonly isConditioning: boolean;
  /** Wall-clock ms this frame's tracking took, excluding the image encoder (0 if stored). */
  readonly ms: number;
}

export interface FrameEncoderOptions {
  /** Keep every frame's encoder outputs (about 6 MB per frame at 512). Default false. */
  readonly cacheAll?: boolean;
  /** Where `image_encoder` timings go (default: the encoder's own `timings`). */
  readonly timings?: Timings;
}

export interface VideoTrackerOptions extends FrameEncoderOptions {
  /** Round stored memory features to bfloat16 as the reference does. Default true. */
  readonly bf16Memory?: boolean;
  /**
   * A per-frame encoder cache shared with other trackers of the same clip (one encoder
   * pass per frame for every object; see `MultiObjectTracker`). Default: a private one.
   */
  readonly encoder?: FrameEncoder;
}

export interface VideoSize {
  readonly height: number;
  readonly width: number;
}

/** Python `//` for integers. */
const floorDiv = (a: number, b: number): number => Math.floor(a / b);

const LABEL_BOX_TL = 2;
const LABEL_BOX_BR = 3;
/** `SAM2.mask_as_output`: a mask prompt becomes +-10 logits and a +-10 object score. */
const MASK_SCALE = 20;
const MASK_BIAS = -10;

/**
 * The image encoder over a clip, run once per frame: a one-frame cache (the frame being
 * clicked or tracked), plus every frame when `cacheAll` is set. The encoder output does
 * not depend on the object, so several trackers can share one of these.
 */
export class FrameEncoder {
  readonly timings: Timings;
  readonly numFrames: number;
  readonly size: number;
  #last: { frame: number; feats: FrameFeatures } | null = null;
  readonly #cache: Map<number, FrameFeatures> | null;

  constructor(
    private readonly sessions: Pick<SessionSet, "imageEncoder">,
    private readonly consts: Constants,
    readonly frames: FrameSource,
    opts: FrameEncoderOptions = {},
  ) {
    this.numFrames = frames.count;
    this.size = consts.image_size;
    this.timings = opts.timings ?? {};
    this.#cache = opts.cacheAll === true ? new Map() : null;
  }

  async features(frame: number): Promise<FrameFeatures> {
    const cached = this.#cache?.get(frame);
    if (cached !== undefined) return cached;
    if (this.#last !== null && this.#last.frame === frame) return this.#last.feats;
    if (!(frame >= 0 && frame < this.numFrames)) throw new Error(`frame ${frame} out of range`);
    const input = await this.frames.get(frame);
    const enc = await encodeImage(this.sessions, input, this.size, this.timings);
    const feats = withMemoryTokens(enc, this.consts);
    this.#last = { frame, feats };
    this.#cache?.set(frame, feats);
    return feats;
  }

  /** Drop the cached features. */
  clear(): void {
    this.#last = null;
    this.#cache?.clear();
  }
}

/**
 * One object tracked through one video with the exported graphs: a port of
 * `OrtVideoTracker` (tools/ort_video_loop.py), which mirrors the reference
 * `VideoPredictor.add_points` / `propagate`. Method names follow the Python steps.
 * Calls must not overlap: await `addClick` and finish (or break out of) `propagate`
 * before starting the next one.
 */
export class VideoTracker {
  readonly timings: Timings = {};
  /** Wall-clock ms of each tracked frame, excluding the image encoder. */
  readonly frameTimes: number[] = [];

  readonly numFrames: number;
  readonly size: number;
  readonly grid: number;

  readonly #cond = new Map<number, FrameOutput>();
  readonly #nonCond = new Map<number, FrameOutput>();
  readonly #pending = new Map<number, FrameOutput>();
  /** frame -> reverse flag of its last tracking. */
  readonly #tracked = new Map<number, boolean>();
  #maskmemPos: Float32Array | null = null; // [HW, 64] tokens, cached once
  readonly #encoder: FrameEncoder;
  readonly #bf16: boolean;
  #busy = false;

  constructor(
    private readonly sessions: SessionSet,
    private readonly consts: Constants,
    frames: FrameSource,
    private readonly videoSize: VideoSize,
    opts: VideoTrackerOptions = {},
  ) {
    this.numFrames = frames.count;
    this.size = consts.image_size;
    this.grid = consts.embed_grid;
    if (opts.encoder !== undefined && opts.encoder.numFrames !== frames.count) {
      throw new Error("VideoTracker: the shared encoder belongs to a clip of another length");
    }
    this.#encoder =
      opts.encoder ?? new FrameEncoder(sessions, consts, frames, { cacheAll: opts.cacheAll, timings: this.timings });
    this.#bf16 = opts.bf16Memory ?? true;
  }

  // ------------------------------------------------------------------ step 1: encode

  /** Encode once per frame (one-frame cache, plus the full cache when enabled). */
  features(frame: number): Promise<FrameFeatures> {
    return this.#encoder.features(frame);
  }

  // ------------------------------------------------------------------ step 2: SAM head

  /** Head graph + output selection + object pointer. */
  async samHead(
    imageEmbed: Float32Array,
    feats: FrameFeatures,
    coords: ArrayLike<number>,
    labels: ArrayLike<number>,
    maskInput: Float32Array | null,
  ): Promise<FrameOutput> {
    const head = await runHead(
      this.sessions,
      { ...feats, imageEmbed },
      coords,
      labels,
      maskInput,
      this.timings,
    );
    let numPoints = 0;
    for (let i = 0; i < labels.length; i++) if (labels[i] !== LABEL_PAD) numPoints++;
    const sel = selectOutput(head.masks, head.ious, head.tokens, head.score, numPoints, this.consts);
    const ptr = await runGraph(
      this.sessions.objectPointer,
      {
        mask_token: f32(sel.token.slice(), [1, sel.token.length]),
        object_score_logits: f32(Float32Array.of(head.score), [1, 1]),
      },
      this.timings,
      "object_pointer",
    );
    return {
      predMasks: sel.lowRes,
      objPtr: ptr["obj_ptr"] as Float32Array,
      score: head.score,
      maskmemFeatures: null,
    };
  }

  // ------------------------------------------------------------------ step 3: memory

  /** Memory encoder on the chosen mask; stores bfloat16 features in `out`. */
  async encodeMemory(frame: number, out: FrameOutput, fromClicks: boolean): Promise<void> {
    const feats = await this.features(frame);
    const S = this.size;
    const g = this.grid;
    const m = this.consts.mask_size;
    const highRes = upsampleBilinear(out.predMasks, m, S);
    const mem = await runGraph(
      this.sessions.memoryEncoder,
      {
        vision_feat: f32(feats.visionFeat, [1, this.consts.hidden_dim, g, g]),
        mask_logits: f32(highRes, [1, 1, S, S]),
        object_score_logits: f32(Float32Array.of(out.score), [1, 1]),
        binarize: f32(Float32Array.of(fromClicks ? 1 : 0), [1]),
      },
      this.timings,
      "memory_encoder",
    );
    const features = mem["maskmem_features"] as Float32Array;
    out.maskmemFeatures = this.#bf16 ? roundBf16(features) : features;
    if (this.#maskmemPos === null) {
      this.#maskmemPos = transpose(mem["maskmem_pos_enc"] as Float32Array, this.consts.mem_dim);
    }
  }

  async ptrPos(tDiffs: readonly number[], maxPtrs: number): Promise<Float32Array> {
    const out = await runGraph(
      this.sessions.objectPointerTpos,
      {
        t_diff: f32(Float32Array.from(tDiffs), [tDiffs.length]),
        t_diff_max: f32(Float32Array.of(maxPtrs - 1), [1]),
      },
      this.timings,
      "object_pointer_tpos",
    );
    return out["ptr_pos"] as Float32Array;
  }

  /** The `memory` / `memory_pos` / `num_obj_ptr_tokens` the memory attention takes. */
  async memoryBank(
    frame: number,
    reverse: boolean,
  ): Promise<{ memory: Float32Array; memoryPos: Float32Array; numTokens: number; numPtrTokens: number }> {
    const c = this.consts;
    const nMem = c.num_maskmem;
    const stride = c.memory_temporal_stride_for_eval;
    const dim = c.mem_dim;
    const [selected, unselected] = selectClosestCondFrames(
      frame,
      this.#cond,
      c.max_cond_frames_in_attn,
    );
    // (a) every selected conditioning frame (t_pos 0), (b) the num_maskmem - 1 most recent
    // frames in tracking direction (t_pos 1 = oldest ... nMem - 1 = previous frame).
    const slots: { tPos: number; out: FrameOutput | undefined }[] = [];
    for (const out of selected.values()) slots.push({ tPos: 0, out });
    for (let tPos = 1; tPos < nMem; tPos++) {
      const tRel = nMem - tPos;
      let prev: number;
      if (tRel === 1) prev = reverse ? frame + 1 : frame - 1;
      else if (reverse) prev = -floorDiv(-(frame + 2), stride) * stride + (tRel - 2) * stride;
      else prev = floorDiv(frame - 2, stride) * stride - (tRel - 2) * stride;
      slots.push({ tPos, out: this.#nonCond.get(prev) ?? unselected.get(prev) });
    }
    const pos = this.#maskmemPos;
    if (pos === null) throw new Error("memoryBank: no memory encoded yet");
    const hw = pos.length / dim;
    const spatial: { feats: Float32Array; tpos: Float32Array }[] = [];
    for (const { tPos, out } of slots) {
      if (out === undefined || out.maskmemFeatures === null) continue;
      spatial.push({
        feats: out.maskmemFeatures,
        tpos: c.maskmem_tpos_enc[nMem - tPos - 1] as Float32Array,
      });
    }

    // (c) object pointers: past conditioning frames, then up to maxPtrs - 1 recent tracked
    // frames, each with its signed distance to this frame.
    const maxPtrs = Math.min(this.numFrames, c.max_obj_ptrs_in_encoder);
    const sign = reverse ? -1 : 1;
    const ptrs: { dist: number; ptr: Float32Array }[] = [];
    for (const [t, o] of selected) {
      if (c.only_obj_ptrs_in_the_past_for_eval && !(reverse ? t >= frame : t <= frame)) continue;
      ptrs.push({ dist: (frame - t) * sign, ptr: o.objPtr });
    }
    for (let tDiff = 1; tDiff < maxPtrs; tDiff++) {
      const t = reverse ? frame + tDiff : frame - tDiff;
      if (t < 0 || t >= this.numFrames) break;
      const out = this.#nonCond.get(t) ?? unselected.get(t);
      if (out !== undefined) ptrs.push({ dist: tDiff, ptr: out.objPtr });
    }
    const split = c.tokens_per_obj_ptr;
    const numPtrTokens = split * ptrs.length;
    const numTokens = spatial.length * hw + numPtrTokens;
    const memory = new Float32Array(numTokens * dim);
    const memoryPos = new Float32Array(numTokens * dim);
    let row = 0;
    for (const { feats, tpos } of spatial) {
      // tokens = transpose of [64, HW]; pos = maskmem_pos + maskmem_tpos_enc[row].
      for (let t = 0; t < hw; t++) {
        const o = (row + t) * dim;
        for (let ch = 0; ch < dim; ch++) {
          memory[o + ch] = feats[ch * hw + t] as number;
          memoryPos[o + ch] = (pos[t * dim + ch] as number) + (tpos[ch] as number);
        }
      }
      row += hw;
    }
    if (ptrs.length > 0) {
      const ptrPos = await this.ptrPos(
        ptrs.map((p) => p.dist),
        maxPtrs,
      );
      // Pointer k becomes tokens split*k .. split*k + split-1 (its channels in chunks of
      // mem_dim), all with pointer k's temporal encoding.
      ptrs.forEach(({ ptr }, k) => {
        for (let i = 0; i < split; i++) {
          const o = (row + i) * dim;
          memory.set(ptr.subarray(i * dim, (i + 1) * dim), o);
          memoryPos.set(ptrPos.subarray(k * dim, (k + 1) * dim), o);
        }
        row += split;
      });
    }
    return { memory, memoryPos, numTokens, numPtrTokens };
  }

  /** The SAM head's `image_embed` for this frame: raw encoder output, or memory-conditioned. */
  async condition(frame: number, isInit: boolean, reverse: boolean): Promise<Float32Array> {
    const feats = await this.features(frame);
    if (isInit) return feats.imageEmbed;
    const bank = await this.memoryBank(frame, reverse);
    const dim = this.consts.mem_dim;
    const hw = this.grid * this.grid;
    const out = await runGraph(
      this.sessions.memoryAttention,
      {
        curr: f32(feats.curr, [1, hw, this.consts.hidden_dim]),
        memory: f32(bank.memory, [1, bank.numTokens, dim]),
        memory_pos: f32(bank.memoryPos, [1, bank.numTokens, dim]),
        num_obj_ptr_tokens: i64(BigInt64Array.of(BigInt(bank.numPtrTokens)), [1]),
      },
      this.timings,
      "memory_attention",
    );
    return out["image_embed"] as Float32Array;
  }

  // ------------------------------------------------------------------ prompts / propagate

  /**
   * Clicks and/or a box on `frame` (replacing earlier prompts there); points and box in
   * video pixels, labels 1 positive / 0 negative. The box becomes two leading points
   * labelled 2 and 3 (so a box alone is a single-mask call). A frame that was tracked
   * before is a correction, conditioned on memory in the direction it was last tracked.
   * Returns the frame's new prediction.
   */
  async addClick(
    frame: number,
    points: readonly Point[],
    labels: readonly number[],
    box?: Box,
  ): Promise<{ lowRes: Float32Array; score: number }> {
    if (points.length !== labels.length) throw new Error("addClick: one label per point");
    if (points.length === 0 && box === undefined) {
      throw new Error("addClick: need at least one point or a box");
    }
    this.#enter();
    try {
      const f = Math.fround;
      const S = f(this.size);
      const { width: w, height: h } = this.videoSize;
      const xy: number[] = [];
      const labs: number[] = [];
      if (box !== undefined) {
        xy.push(box.x0, box.y0, box.x1, box.y1);
        labs.push(LABEL_BOX_TL, LABEL_BOX_BR);
      }
      for (let i = 0; i < points.length; i++) {
        const p = points[i] as Point;
        xy.push(p.x, p.y);
        labs.push(labels[i] as number);
      }
      const coords = new Float32Array(xy.length);
      for (let i = 0; i < xy.length; i += 2) {
        coords[i] = f(f(xy[i] as number) / f(w)) * S;
        coords[i + 1] = f(f(xy[i + 1] as number) / f(h)) * S;
      }
      const prior = this.#pending.get(frame) ?? this.#cond.get(frame) ?? this.#nonCond.get(frame);
      const maskInput =
        prior === undefined ? null : clampMaskInput(prior.predMasks, this.consts.mask_input_clamp);
      const isInit = !this.#tracked.has(frame);
      const reverse = isInit ? false : (this.#tracked.get(frame) as boolean);
      const embed = await this.condition(frame, isInit, reverse);
      const out = await this.samHead(embed, await this.features(frame), coords, labs, maskInput);
      this.#pending.set(frame, out);
      return { lowRes: out.predMasks, score: out.score };
    } finally {
      this.#busy = false;
    }
  }

  /**
   * A mask prompt on `frame` (`VideoPredictor.add_mask` / `SAM2.mask_as_output`): the
   * binary mask is taken as the frame's prediction instead of being re-predicted, and
   * replaces any clicks there. `mask` holds one value per pixel, row-major, foreground
   * where `> 0` (so low-res logits, a video-resolution mask or 0/1 bytes all work); `size`
   * is its height and width, or omitted for a square mask. Any size: it is resized to
   * S x S (antialiased bilinear, threshold 0.5). The low-res logits are `mask * 20 - 10`
   * antialias-downsampled by 4, the score is 10 (any pixel set) or -10, and the SAM head
   * runs only for the object pointer, with the mask (through the learned stride-4 conv) as
   * its dense prompt and no points; an empty mask gets the learned no-object pointer. The
   * memory is not consulted, even on a frame that was already tracked.
   */
  async addMask(
    frame: number,
    mask: ArrayLike<number>,
    size?: VideoSize,
  ): Promise<{ lowRes: Float32Array; score: number }> {
    const side = Math.round(Math.sqrt(mask.length));
    const h = size?.height ?? side;
    const w = size?.width ?? side;
    if (mask.length !== h * w || h <= 0 || w <= 0) {
      throw new Error(`addMask: ${mask.length} values do not form a ${h} x ${w} mask`);
    }
    this.#enter();
    try {
      const S = this.size;
      const c = this.consts;
      const binary = new Float32Array(h * w);
      for (let i = 0; i < binary.length; i++) binary[i] = (mask[i] as number) > 0 ? 1 : 0;
      let m = binary;
      if (h !== S || w !== S) {
        const soft = resizeAntialias(binary, { height: h, width: w }, S);
        m = new Float32Array(S * S);
        for (let i = 0; i < m.length; i++) m[i] = (soft[i] as number) >= 0.5 ? 1 : 0;
      }
      let isObj = false;
      const highRes = new Float32Array(S * S);
      for (let i = 0; i < m.length; i++) {
        if ((m[i] as number) > 0) isObj = true;
        highRes[i] = (m[i] as number) * MASK_SCALE + MASK_BIAS;
      }
      const lowRes = resizeAntialias(highRes, S, c.mask_size);
      const feats = await this.features(frame);
      const head = await runHead(
        this.sessions,
        feats,
        [0, 0],
        [LABEL_PAD],
        maskDownsample(m, S, c.mask_downsample_weight, c.mask_downsample_bias),
        this.timings,
      );
      let objPtr: Float32Array;
      if (isObj) {
        const dim = head.tokens.length / 4;
        const ptr = await runGraph(
          this.sessions.objectPointer,
          {
            mask_token: f32(head.tokens.slice(0, dim), [1, dim]),
            object_score_logits: f32(Float32Array.of(head.score), [1, 1]),
          },
          this.timings,
          "object_pointer",
        );
        objPtr = ptr["obj_ptr"] as Float32Array;
      } else {
        objPtr = c.no_obj_ptr.slice();
      }
      const score = isObj ? MASK_SCALE + MASK_BIAS : MASK_BIAS;
      this.#pending.set(frame, { predMasks: lowRes, objPtr, score, maskmemFeatures: null });
      return { lowRes, score };
    } finally {
      this.#busy = false;
    }
  }

  /** Encode prompted frames into memory and make them conditioning frames. */
  async preflight(): Promise<void> {
    const frames = [...this.#pending.keys()].sort((a, b) => a - b);
    for (const frame of frames) {
      const out = this.#pending.get(frame) as FrameOutput;
      if (out.maskmemFeatures === null) await this.encodeMemory(frame, out, true);
      this.#cond.set(frame, out);
      this.#nonCond.delete(frame);
    }
    this.#pending.clear();
    if (this.#cond.size === 0) throw new Error("add a prompt before propagating");
  }

  /**
   * Track from `start` (default: the first conditioning frame) to the clip end in one
   * direction. Conditioning frames are yielded with their stored prediction.
   */
  async *propagate(start?: number, reverse = false): AsyncGenerator<TrackedFrame, void, void> {
    this.#enter();
    try {
      await this.preflight();
      const first = start ?? Math.min(...this.#cond.keys());
      const step = reverse ? -1 : 1;
      for (let frame = first; reverse ? frame >= 0 : frame < this.numFrames; frame += step) {
        let out = this.#cond.get(frame);
        const isConditioning = out !== undefined;
        let took = 0;
        if (out === undefined) {
          const feats = await this.features(frame);
          const t0 = performance.now();
          const embed = await this.condition(frame, false, reverse);
          out = await this.samHead(embed, feats, [0, 0], [LABEL_PAD], null);
          await this.encodeMemory(frame, out, false);
          this.#nonCond.set(frame, out);
          took = performance.now() - t0;
          this.frameTimes.push(took);
        }
        this.#tracked.set(frame, reverse);
        yield { frame, lowRes: out.predMasks, score: out.score, isConditioning, ms: took };
      }
    } finally {
      this.#busy = false;
    }
  }

  /** The latest prediction stored for a frame (clicked, conditioning or tracked), if any. */
  prediction(frame: number): TrackedFrame | null {
    const pending = this.#pending.get(frame);
    const cond = this.#cond.get(frame);
    const out = pending ?? cond ?? this.#nonCond.get(frame);
    if (out === undefined) return null;
    return {
      frame,
      lowRes: out.predMasks,
      score: out.score,
      isConditioning: pending !== undefined || cond !== undefined,
      ms: 0,
    };
  }

  /** Prompted frames (conditioning frames plus prompts not yet propagated), ascending. */
  get promptedFrames(): number[] {
    return [...new Set([...this.#cond.keys(), ...this.#pending.keys()])].sort((a, b) => a - b);
  }

  get hasPrompts(): boolean {
    return this.#cond.size > 0 || this.#pending.size > 0;
  }

  get busy(): boolean {
    return this.#busy;
  }

  /** Forget every prompt and memory (keeps the encoder cache). */
  reset(): void {
    if (this.#busy) throw new Error("VideoTracker: busy");
    this.#cond.clear();
    this.#nonCond.clear();
    this.#pending.clear();
    this.#tracked.clear();
  }

  #enter(): void {
    if (this.#busy) throw new Error("VideoTracker: a click or propagation is already running");
    this.#busy = true;
  }
}

/** `[C, HW]` -> `[HW, C]`. */
function transpose(x: Float32Array, channels: number): Float32Array {
  const hw = x.length / channels;
  const out = new Float32Array(x.length);
  for (let c = 0; c < channels; c++) {
    for (let t = 0; t < hw; t++) out[t * channels + c] = x[c * hw + t] as number;
  }
  return out;
}

/**
 * `select_closest_cond_frames`: every conditioning frame for the released
 * `max_cond_frames_in_attn = -1`; otherwise the closest before / after plus the nearest
 * nearest others (in that order, as the Python dict; the slice keeps Python's semantics
 * for a negative count).
 */
export function selectClosestCondFrames<T>(
  frame: number,
  cond: ReadonlyMap<number, T>,
  maxCondFrames: number,
): [Map<number, T>, Map<number, T>] {
  if (maxCondFrames === -1 || cond.size <= maxCondFrames) return [new Map(cond), new Map()];
  const keys = [...cond.keys()];
  const chosen: number[] = [];
  const before = keys.filter((t) => t < frame);
  if (before.length > 0) chosen.push(Math.max(...before));
  const after = keys.filter((t) => t >= frame);
  if (after.length > 0) chosen.push(Math.min(...after));
  const rest = keys
    .filter((t) => !chosen.includes(t))
    .sort((a, b) => Math.abs(a - frame) - Math.abs(b - frame));
  chosen.push(...rest.slice(0, maxCondFrames - chosen.length));
  const selected = new Map<number, T>();
  for (const t of chosen) selected.set(t, cond.get(t) as T);
  const unselected = new Map<number, T>();
  for (const [t, v] of cond) if (!selected.has(t)) unselected.set(t, v);
  return [selected, unselected];
}
