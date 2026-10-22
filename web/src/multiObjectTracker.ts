import type { Constants } from "./constants";
import type { Point } from "./imagePredictor";
import type { SessionSet, Timings } from "./sessions";
import {
  FrameEncoder,
  VideoTracker,
  type FrameSource,
  type TrackedFrame,
  type VideoSize,
  type VideoTrackerOptions,
} from "./videoTracker";

/** One object's output on one frame of a multi-object propagation. */
export interface ObjectFrame {
  /** `[m, m]` low-res logits of this object alone (its memory sees these, unconstrained). */
  readonly lowRes: Float32Array;
  readonly score: number;
  readonly isConditioning: boolean;
  /** Wall-clock ms of this object's step on the frame, excluding the shared encoder. */
  readonly ms: number;
}

export interface MultiTrackedFrame {
  readonly frame: number;
  /** Every object with prompts, in the order the objects were added. */
  readonly perObject: Map<number, ObjectFrame>;
  /** Wall-clock ms of the image encoder on this frame (0 when its features were cached). */
  readonly encoderMs: number;
}

/** `non_overlapping` clamps the losing objects' logits to at most this value. */
export const NON_OVERLAP_CLAMP = -10;

/**
 * Several objects tracked through one video, each with its own prompts and memory bank,
 * sharing one image-encoder pass per frame. Objects are independent exactly as in the
 * reference (and in `src/sam2_scratch/video_predictor.py`): each one's memory sees its own
 * unconstrained mask; overlaps are resolved for display only, with `nonOverlapping`.
 * Calls must not overlap: await `addClick` and finish (or break out of) `propagate`
 * before starting the next one.
 */
export class MultiObjectTracker {
  /** Shared `image_encoder` timings; each object's other graphs are timed on its tracker. */
  readonly timings: Timings = {};
  readonly encoder: FrameEncoder;
  readonly numFrames: number;

  readonly #objects = new Map<number, VideoTracker>();
  readonly #opts: VideoTrackerOptions;
  #nextId = 1;
  #busy = false;

  constructor(
    private readonly sessions: SessionSet,
    private readonly consts: Constants,
    private readonly frames: FrameSource,
    private readonly videoSize: VideoSize,
    opts: Omit<VideoTrackerOptions, "encoder" | "timings"> = {},
  ) {
    this.numFrames = frames.count;
    this.encoder = new FrameEncoder(sessions, consts, frames, { cacheAll: opts.cacheAll, timings: this.timings });
    this.#opts = { bf16Memory: opts.bf16Memory, encoder: this.encoder };
  }

  // ------------------------------------------------------------------ objects

  /** Create an object (an empty memory bank) and return its id (1, 2, ...). */
  addObject(): number {
    const id = this.#nextId++;
    this.#objects.set(id, new VideoTracker(this.sessions, this.consts, this.frames, this.videoSize, this.#opts));
    return id;
  }

  /** Drop an object and everything it remembers. */
  removeObject(id: number): void {
    this.#check();
    if (!this.#objects.delete(id)) throw new Error(`MultiObjectTracker: no object ${id}`);
  }

  /** Forget an object's clicks and memory; it stays in the list. */
  clearObject(id: number): void {
    this.#check();
    this.#object(id).reset();
  }

  /** Object ids in creation order. */
  get objectIds(): number[] {
    return [...this.#objects.keys()];
  }

  /** The object's own tracker: its timings, frame times and stored predictions. */
  object(id: number): VideoTracker {
    return this.#object(id);
  }

  hasPrompts(id: number): boolean {
    return this.#object(id).hasPrompts;
  }

  // ------------------------------------------------------------------ prompts / propagate

  /** Clicks for one object on `frame`: see `VideoTracker.addClick`. */
  async addClick(
    objId: number,
    frame: number,
    points: readonly Point[],
    labels: readonly number[],
  ): Promise<{ lowRes: Float32Array; score: number }> {
    const tracker = this.#object(objId);
    this.#enter();
    try {
      return await tracker.addClick(frame, points, labels);
    } finally {
      this.#busy = false;
    }
  }

  /**
   * Track every prompted object from `start` (default: the first prompted frame over all of
   * them) to the clip end in one direction, frame by frame, so each frame is encoded once
   * and every object steps on it before the next frame is touched. Objects without prompts
   * are skipped.
   */
  async *propagate(start?: number, reverse = false): AsyncGenerator<MultiTrackedFrame, void, void> {
    this.#enter();
    const gens: [number, AsyncGenerator<TrackedFrame, void, void>][] = [];
    try {
      const prompted = [...this.#objects].filter(([, t]) => t.hasPrompts);
      if (prompted.length === 0) throw new Error("add a click before propagating");
      const first = start ?? Math.min(...prompted.map(([, t]) => t.promptedFrames[0] as number));
      for (const [id, t] of prompted) gens.push([id, t.propagate(first, reverse)]);
      for (;;) {
        const before = this.timings["image_encoder"]?.length ?? 0;
        const perObject = new Map<number, ObjectFrame>();
        let frame = -1;
        for (const [id, gen] of gens) {
          const r = await gen.next();
          if (r.done === true) break;
          const { lowRes, score, isConditioning, ms } = r.value;
          frame = r.value.frame;
          perObject.set(id, { lowRes, score, isConditioning, ms });
        }
        // Every object walks the same frames, so they all finish on the same step.
        if (perObject.size < gens.length) break;
        const enc = this.timings["image_encoder"] ?? [];
        let encoderMs = 0;
        for (let i = before; i < enc.length; i++) encoderMs += enc[i] as number;
        yield { frame, perObject, encoderMs };
      }
    } finally {
      // Leaving early (break, throw) must release every object's generator too.
      for (const [, gen] of gens) await gen.return(undefined);
      this.#busy = false;
    }
  }

  /** The latest prediction stored for each object on a frame (objects without one are absent). */
  prediction(frame: number): Map<number, TrackedFrame> {
    const out = new Map<number, TrackedFrame>();
    for (const [id, t] of this.#objects) {
      const p = t.prediction(frame);
      if (p !== null) out.set(id, p);
    }
    return out;
  }

  /** Forget every object's clicks and memory; the objects stay (the encoder cache too). */
  reset(): void {
    this.#check();
    for (const t of this.#objects.values()) t.reset();
  }

  #object(id: number): VideoTracker {
    const t = this.#objects.get(id);
    if (t === undefined) throw new Error(`MultiObjectTracker: no object ${id}`);
    return t;
  }

  #check(): void {
    if (this.#busy) throw new Error("MultiObjectTracker: busy");
  }

  #enter(): void {
    this.#check();
    this.#busy = true;
  }
}

/**
 * `non_overlapping` (src/sam2_scratch/video_predictor.py): at every pixel only the object
 * with the highest logit keeps its value; the others are clamped to at most -10, so they
 * read as background there. Ties go to the earliest object in the map, as `torch.argmax`
 * returns the first maximum. With one object (or none) the logits are returned unchanged.
 */
export function nonOverlapping(logitsByObject: ReadonlyMap<number, Float32Array>): Map<number, Float32Array> {
  const ids = [...logitsByObject.keys()];
  const out = new Map<number, Float32Array>();
  if (ids.length <= 1) {
    for (const id of ids) out.set(id, logitsByObject.get(id) as Float32Array);
    return out;
  }
  const maps = ids.map((id) => logitsByObject.get(id) as Float32Array);
  const n = (maps[0] as Float32Array).length;
  for (const m of maps) if (m.length !== n) throw new Error("nonOverlapping: objects differ in mask size");
  const result = maps.map((m) => new Float32Array(m.length));
  for (let i = 0; i < n; i++) {
    let winner = 0;
    let best = (maps[0] as Float32Array)[i] as number;
    for (let k = 1; k < maps.length; k++) {
      const v = (maps[k] as Float32Array)[i] as number;
      if (v > best) {
        best = v;
        winner = k;
      }
    }
    for (let k = 0; k < maps.length; k++) {
      const v = (maps[k] as Float32Array)[i] as number;
      (result[k] as Float32Array)[i] = k === winner ? v : Math.min(v, NON_OVERLAP_CLAMP);
    }
  }
  ids.forEach((id, k) => out.set(id, result[k] as Float32Array));
  return out;
}
