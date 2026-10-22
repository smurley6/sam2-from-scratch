import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { Constants } from "../src/constants";
import { MultiObjectTracker, nonOverlapping } from "../src/multiObjectTracker";
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

interface Click {
  frame: number;
  points: [number, number][];
  labels: number[];
}
interface Meta {
  size: number;
  video_hw: [number, number];
  num_frames: number;
  objects: Click[];
  frames_u8_shape: [number, number, number, number];
  /** `[objects, frames, m, m]`. */
  masks_shape: [number, number, number, number];
  object_score_logits: number[][];
  pairwise_iou: number[];
}

const FULL = process.env["SAM2_FULL"] === "1";

describe.skipIf(!haveExports)("multi-object parity (bmx-trees, rider and bike from frame 0)", () => {
  const dir = join(REFERENCE, "clip_bmx-trees");
  let sessions: SessionSet;
  let consts: Constants;
  let meta: Meta;
  let frames: Uint8Array;
  let ref: Float32Array;
  let refNonOverlap: Float32Array;
  let source: FrameSource;
  let area: number;

  /** Reference low-res logits of object `k` on `frame`. */
  const want = (buf: Float32Array, k: number, frame: number): Float32Array => {
    const T = meta.masks_shape[1];
    return buf.subarray((k * T + frame) * area, (k * T + frame + 1) * area);
  };

  beforeAll(async () => {
    [sessions, consts, meta, frames, ref, refNonOverlap] = await Promise.all([
      loadSessions(),
      loadConstants(),
      readJson<Meta>(join(dir, "meta.json")),
      bytes(join(dir, "frames_u8.bin")),
      readF32(join(dir, "masks.bin")),
      readF32(join(dir, "nonoverlap.bin")),
    ]);
    const [count, S] = meta.frames_u8_shape;
    expect(S).toBe(SIZE);
    const frameBytes = S * S * 3;
    source = {
      count,
      get: async (i) => normalize(frames.subarray(i * frameBytes, (i + 1) * frameBytes), S, consts),
    };
    area = meta.masks_shape[2] * meta.masks_shape[3];
  });

  afterAll(async () => {
    await releaseSessions(sessions);
  });

  it("nonOverlapping reproduces the torch non_overlapping on the reference runs exactly", () => {
    const [K, T] = meta.masks_shape;
    expect(K).toBe(2);
    for (let frame = 0; frame < T; frame++) {
      const byObject = new Map<number, Float32Array>();
      for (let k = 0; k < K; k++) byObject.set(k, want(ref, k, frame));
      const out = nonOverlapping(byObject);
      for (let k = 0; k < K; k++) {
        expect(maxAbsDiff(out.get(k) as Float32Array, want(refNonOverlap, k, frame)), `frame ${frame} object ${k}`).toBe(0);
      }
    }
    // The two prompts pick different things, so the test says something.
    expect(Math.max(...meta.pairwise_iou)).toBeLessThan(0.2);
  });

  it(`tracks both objects with one encoder pass per frame (${FULL ? "all" : "first 12"} frames)`, async () => {
    const [count] = meta.frames_u8_shape;
    const [h, w] = meta.video_hw;
    const tracker = new MultiObjectTracker(sessions, consts, source, { height: h, width: w });
    const ids = meta.objects.map(() => tracker.addObject());
    expect(ids).toEqual([1, 2]);
    for (const [k, click] of meta.objects.entries()) {
      const res = await tracker.addClick(
        ids[k] as number,
        click.frame,
        click.points.map(([x, y]) => ({ x, y })),
        click.labels,
      );
      expect(maskIou(res.lowRes, want(ref, k, click.frame)), `object ${k} click`).toBeGreaterThanOrEqual(0.99);
    }

    const limit = FULL ? count : 12;
    const ious: number[][] = meta.objects.map(() => []);
    const combinedIous: number[] = [];
    let maxScoreErr = 0;
    let seen = 0;
    let objectMs = 0;
    for await (const out of tracker.propagate()) {
      expect(out.frame).toBe(seen);
      expect([...out.perObject.keys()]).toEqual(ids);
      const byObject = new Map<number, Float32Array>();
      for (const [k, id] of ids.entries()) {
        const o = out.perObject.get(id);
        expect(o).toBeDefined();
        if (o === undefined) continue;
        expect(o.isConditioning).toBe(out.frame === (meta.objects[k] as Click).frame);
        const iou = maskIou(o.lowRes, want(ref, k, out.frame));
        (ious[k] as number[]).push(iou);
        expect(iou, `frame ${out.frame} object ${k} IoU`).toBeGreaterThanOrEqual(0.99);
        const scoreErr = Math.abs(o.score - ((meta.object_score_logits[k] as number[])[out.frame] as number));
        maxScoreErr = Math.max(maxScoreErr, scoreErr);
        expect(scoreErr, `frame ${out.frame} object ${k} score`).toBeLessThan(0.1);
        byObject.set(k, o.lowRes);
        objectMs += o.ms;
      }
      const combined = nonOverlapping(byObject);
      for (const k of byObject.keys()) {
        const iou = maskIou(combined.get(k) as Float32Array, want(refNonOverlap, k, out.frame));
        combinedIous.push(iou);
        expect(iou, `frame ${out.frame} object ${k} non-overlapping IoU`).toBeGreaterThanOrEqual(0.99);
      }
      if (++seen >= limit) break;
    }
    expect(seen).toBe(limit);

    // One encoder call per frame, on the shared encoder; none on the per-object trackers.
    expect(tracker.timings["image_encoder"]?.length).toBe(seen);
    for (const id of ids) expect(tracker.object(id).timings["image_encoder"]).toBeUndefined();
    expect(tracker.object(ids[0] as number).frameTimes.length).toBe(seen - 1);

    const enc = median(tracker.timings["image_encoder"] ?? [0]);
    const perObject = ids.map((id) => median(tracker.object(id).frameTimes));
    const minIou = ious.map((v) => Math.min(...v));
    console.log(
      `[multi-object parity] ${seen} frames x ${ids.length} objects: IoU min ${minIou.map((v) => v.toFixed(5)).join(" / ")}, ` +
        `non-overlapping IoU min ${Math.min(...combinedIous).toFixed(5)}, max|score err| ${maxScoreErr.toExponential(2)}\n` +
        `  median ms: encoder ${enc.toFixed(0)} once per frame, per object ${perObject.map((v) => v.toFixed(0)).join(" / ")}; ` +
        `tracked frame ~${(enc + perObject.reduce((a, b) => a + b, 0)).toFixed(0)} (object steps ${(objectMs / Math.max(1, seen - 1)).toFixed(0)} per frame)`,
    );

    // Breaking out released every object: the tracker takes the next call.
    expect(tracker.prediction(1).size).toBe(ids.length);
    tracker.clearObject(ids[1] as number);
    expect(tracker.prediction(1).size).toBe(1);
    expect(tracker.hasPrompts(ids[1] as number)).toBe(false);
    tracker.removeObject(ids[1] as number);
    expect(tracker.objectIds).toEqual([ids[0]]);
    expect(() => tracker.removeObject(ids[1] as number)).toThrow(/no object/);
  });

  it("an object's masks are bit-identical to tracking it alone, and unprompted objects are skipped", async () => {
    const [h, w] = meta.video_hw;
    const click = meta.objects[0] as Click;
    const points = click.points.map(([x, y]) => ({ x, y }));
    const alone = new VideoTracker(sessions, consts, source, { height: h, width: w });
    await alone.addClick(click.frame, points, click.labels);
    const single: Float32Array[] = [];
    for await (const out of alone.propagate()) {
      single.push(out.lowRes);
      if (single.length === 4) break;
    }

    const multi = new MultiObjectTracker(sessions, consts, source, { height: h, width: w });
    const rider = multi.addObject();
    multi.addObject(); // never prompted
    await multi.addClick(rider, click.frame, points, click.labels);
    let seen = 0;
    for await (const out of multi.propagate()) {
      expect([...out.perObject.keys()]).toEqual([rider]);
      expect(maxAbsDiff((out.perObject.get(rider) as { lowRes: Float32Array }).lowRes, single[seen] as Float32Array)).toBe(0);
      // Frame 0 was encoded by the click; every later frame is encoded inside propagate.
      if (seen > 0) expect(out.encoderMs).toBeGreaterThan(0);
      else expect(out.encoderMs).toBe(0);
      if (++seen === single.length) break;
    }
    expect(seen).toBe(4);

    const empty = new MultiObjectTracker(sessions, consts, source, { height: h, width: w });
    empty.addObject();
    await expect(empty.propagate().next()).rejects.toThrow(/add a click/);
    // The failed call released the tracker.
    expect(() => empty.reset()).not.toThrow();
  });
});
