import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { Constants } from "../src/constants";
import { normalize } from "../src/preprocess";
import { releaseSessions, type SessionSet } from "../src/sessions";
import { VideoTracker, type FrameSource } from "../src/videoTracker";
import {
  bytes,
  haveExports,
  loadConstants,
  loadSessions,
  maskIou,
  maxAbsDiff,
  median,
  readF32,
  readJson,
  REFERENCE,
  SIZE,
} from "./helpers";

interface Meta {
  size: number;
  video_hw: [number, number];
  num_frames: number;
  click: { frame: number; points: [number, number][]; labels: number[] };
  /** Box in video pixels (x0, y0, x1, y1) of the box run. */
  box?: { frame: number; box: [number, number, number, number] };
  /** The mask run's prompt: a `[h, w]` uint8 file. */
  mask_prompt?: { frame: number; shape: [number, number]; file: string };
  frames_u8_shape: [number, number, number, number];
  masks_shape: [number, number, number];
  object_score_logits: number[];
  object_score_logits_box?: number[];
  object_score_logits_mask?: number[];
}

const FULL = process.env["SAM2_FULL"] === "1";

describe.skipIf(!haveExports)("video parity (blackswan, forward from a click on frame 0)", () => {
  const dir = join(REFERENCE, "clip_blackswan");
  let sessions: SessionSet;
  let consts: Constants;
  let meta: Meta;
  let frames: Uint8Array;
  let ref: Float32Array;

  beforeAll(async () => {
    [sessions, consts, meta, frames, ref] = await Promise.all([
      loadSessions(),
      loadConstants(),
      readJson<Meta>(join(dir, "meta.json")),
      bytes(join(dir, "frames_u8.bin")),
      readF32(join(dir, "masks.bin")),
    ]);
  });

  afterAll(async () => {
    await releaseSessions(sessions);
  });

  it(`matches the Python loop per frame (${FULL ? "all" : "first 12"} frames)`, async () => {
    const [count, S] = meta.frames_u8_shape;
    expect(S).toBe(SIZE);
    const frameBytes = S * S * 3;
    // Every frame of the clip is visible to the tracker (the pointer temporal scale depends
    // on the clip length); only the number of propagated frames is limited.
    const source: FrameSource = {
      count,
      get: async (i) => normalize(frames.subarray(i * frameBytes, (i + 1) * frameBytes), S, consts),
    };
    const [h, w] = meta.video_hw;
    const tracker = new VideoTracker(sessions, consts, source, { height: h, width: w });
    const { frame, points, labels } = meta.click;
    await tracker.addClick(
      frame,
      points.map(([x, y]) => ({ x, y })),
      labels,
    );

    const limit = FULL ? count : 12;
    const m = meta.masks_shape[1];
    const area = m * m;
    const ious: number[] = [];
    let maxScoreErr = 0;
    let maxLogitErr = 0;
    let seen = 0;
    for await (const out of tracker.propagate()) {
      expect(out.frame).toBe(seen);
      expect(out.isConditioning).toBe(out.frame === frame);
      const want = ref.subarray(out.frame * area, (out.frame + 1) * area);
      const iou = maskIou(out.lowRes, want);
      const scoreErr = Math.abs(out.score - (meta.object_score_logits[out.frame] as number));
      ious.push(iou);
      maxScoreErr = Math.max(maxScoreErr, scoreErr);
      maxLogitErr = Math.max(maxLogitErr, maxAbsDiff(out.lowRes, want));
      expect(iou, `frame ${out.frame} IoU`).toBeGreaterThanOrEqual(0.99);
      expect(scoreErr, `frame ${out.frame} object score`).toBeLessThan(0.1);
      if (++seen >= limit) break;
    }
    expect(seen).toBe(limit);

    const mean = ious.reduce((a, b) => a + b, 0) / ious.length;
    const worst = ious.indexOf(Math.min(...ious));
    const t = tracker.timings;
    const ms = (k: string): string => (t[k] ? median(t[k]).toFixed(1) : "-");
    console.log(
      `[video parity] ${seen} frames: IoU min ${Math.min(...ious).toFixed(5)} (frame ${worst}) ` +
        `mean ${mean.toFixed(5)}, max|score err| ${maxScoreErr.toExponential(2)}, ` +
        `max|logit err| ${maxLogitErr.toExponential(2)}\n` +
        `  median ms: image_encoder ${ms("image_encoder")}, memory_attention ${ms("memory_attention")}, ` +
        `sam_head ${ms("sam_head")}, memory_encoder ${ms("memory_encoder")}, ` +
        `object_pointer ${ms("object_pointer")}, object_pointer_tpos ${ms("object_pointer_tpos")}; ` +
        `tracked frame excl. encoder ${median(tracker.frameTimes).toFixed(0)}, ` +
        `incl. encoder ~${(median(tracker.frameTimes) + median(t["image_encoder"] ?? [0])).toFixed(0)}`,
    );

    // The finished generator releases the tracker for the next call; a stored prediction
    // is still available for display.
    expect(tracker.prediction(1)?.lowRes.length).toBe(area);
  });

  /**
   * Propagate forward from the prompt on frame 0 and compare every frame with `want`
   * (`[T, m, m]` low-res logits) and `scores`; returns the per-frame IoUs.
   */
  async function checkForward(tracker: VideoTracker, want: Float32Array, scores: number[], label: string): Promise<number[]> {
    const limit = FULL ? meta.frames_u8_shape[0] : 12;
    const area = meta.masks_shape[1] * meta.masks_shape[2];
    const ious: number[] = [];
    let seen = 0;
    for await (const out of tracker.propagate()) {
      expect(out.frame).toBe(seen);
      const ref = want.subarray(out.frame * area, (out.frame + 1) * area);
      const iou = maskIou(out.lowRes, ref);
      ious.push(iou);
      expect(iou, `${label}: frame ${out.frame} IoU`).toBeGreaterThanOrEqual(0.99);
      expect(Math.abs(out.score - (scores[out.frame] as number)), `${label}: frame ${out.frame} score`).toBeLessThan(0.1);
      if (++seen >= limit) break;
    }
    expect(seen).toBe(limit);
    console.log(`[video parity, ${label}] ${seen} frames: IoU min ${Math.min(...ious).toFixed(5)}`);
    return ious;
  }

  const source = (): FrameSource => {
    const [count, S] = meta.frames_u8_shape;
    const frameBytes = S * S * 3;
    return { count, get: async (i) => normalize(frames.subarray(i * frameBytes, (i + 1) * frameBytes), S, consts) };
  };

  it("tracks from a box on frame 0 like the Python loop (two leading points, labels 2 and 3)", async () => {
    if (meta.box === undefined) return; // dumps from before the box run
    const [h, w] = meta.video_hw;
    const tracker = new VideoTracker(sessions, consts, source(), { height: h, width: w });
    const [x0, y0, x1, y1] = meta.box.box;
    const first = await tracker.addClick(meta.box.frame, [], [], { x0, y0, x1, y1 });
    expect(first.score).toBeGreaterThan(0);
    await expect(tracker.addClick(0, [], [])).rejects.toThrow(/point or a box/);
    const want = await readF32(join(dir, "masks_box.bin"));
    await checkForward(tracker, want, meta.object_score_logits_box as number[], "box");
  });

  it("tracks from a mask prompt on frame 0 like the Python loop (mask taken as the prediction)", async () => {
    if (meta.mask_prompt === undefined) return;
    const [h, w] = meta.video_hw;
    const [mh, mw] = meta.mask_prompt.shape;
    const prompt = await bytes(join(dir, meta.mask_prompt.file));
    expect(prompt.length).toBe(mh * mw);
    const tracker = new VideoTracker(sessions, consts, source(), { height: h, width: w });
    const first = await tracker.addMask(meta.mask_prompt.frame, prompt, { height: mh, width: mw });
    // +-10 logits and a +10 score, bit for bit on the prompted frame itself.
    expect(first.score).toBe(10);
    const want = await readF32(join(dir, "masks_mask.bin"));
    const area = meta.masks_shape[1] * meta.masks_shape[2];
    expect(maxAbsDiff(first.lowRes, want.subarray(0, area))).toBeLessThan(1e-3);
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of first.lowRes) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    expect(lo).toBeGreaterThanOrEqual(-10);
    expect(hi).toBeLessThanOrEqual(10);
    await checkForward(tracker, want, meta.object_score_logits_mask as number[], "mask");
    // A square low-res mask needs no size; an empty mask scores the object absent.
    const empty = await tracker.addMask(1, new Uint8Array(area));
    expect(empty.score).toBe(-10);
    await expect(tracker.addMask(1, new Uint8Array(7))).rejects.toThrow(/mask/);
  });

  it("handles a correction on a tracked frame and reverse propagation", async () => {
    const [count, S] = meta.frames_u8_shape;
    const frameBytes = S * S * 3;
    const source: FrameSource = {
      count,
      get: async (i) => normalize(frames.subarray(i * frameBytes, (i + 1) * frameBytes), S, consts),
    };
    const [h, w] = meta.video_hw;
    const tracker = new VideoTracker(sessions, consts, source, { height: h, width: w }, { cacheAll: true });
    const click = meta.click.points.map(([x, y]) => ({ x, y }));
    await tracker.addClick(0, click, [1]);
    const forward: number[] = [];
    for await (const out of tracker.propagate()) {
      forward.push(out.frame);
      if (forward.length === 4) break;
    }
    expect(forward).toEqual([0, 1, 2, 3]);
    // Frame 3 was tracked forward: the click is a correction conditioned on memory.
    const area = meta.masks_shape[1] * meta.masks_shape[2];
    const fix = await tracker.addClick(3, click, [1]);
    expect(fix.score).toBeGreaterThan(0);
    const back: { frame: number; cond: boolean; iou: number }[] = [];
    for await (const out of tracker.propagate(3, true)) {
      const want = ref.subarray(out.frame * area, (out.frame + 1) * area);
      back.push({ frame: out.frame, cond: out.isConditioning, iou: maskIou(out.lowRes, want) });
    }
    expect(back.map((b) => [b.frame, b.cond])).toEqual([
      [3, true],
      [2, false],
      [1, false],
      [0, true],
    ]);
    // No reference exists for this prompt sequence; the swan must still be the object.
    for (const b of back) expect(b.iou, `frame ${b.frame}`).toBeGreaterThan(0.9);
    tracker.reset();
    expect(tracker.prediction(0)).toBeNull();
  });
});
