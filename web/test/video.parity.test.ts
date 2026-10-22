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
  frames_u8_shape: [number, number, number, number];
  masks_shape: [number, number, number];
  object_score_logits: number[];
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
