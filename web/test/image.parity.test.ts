import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { Constants } from "../src/constants";
import { ImagePredictor } from "../src/imagePredictor";
import { normalize } from "../src/preprocess";
import { releaseSessions, type SessionSet } from "../src/sessions";
import {
  bytes,
  haveExports,
  loadConstants,
  loadSessions,
  maskIou,
  maxAbsDiff,
  readF32,
  readJson,
  REFERENCE,
  SIZE,
} from "./helpers";

interface Prompt {
  name: string;
  point_coords_512: [number, number][];
  point_labels: number[];
  iou_predictions: number[];
  object_score_logits: number;
}

interface Prompts {
  size: number;
  masks_shape: [number, number, number, number];
  prompts: Prompt[];
}

describe.skipIf(!haveExports)("image parity (truck, int8 encoder + fp32 head)", () => {
  const dir = join(REFERENCE, "image_truck");
  let sessions: SessionSet;
  let consts: Constants;
  let predictor: ImagePredictor;
  let meta: Prompts;
  let ref: Float32Array;

  beforeAll(async () => {
    [sessions, consts, meta, ref] = await Promise.all([
      loadSessions(),
      loadConstants(),
      readJson<Prompts>(join(dir, "prompts.json")),
      readF32(join(dir, "masks.bin")),
    ]);
    predictor = new ImagePredictor(sessions, consts);
    const u8 = await bytes(join(dir, "image_u8.bin"));
    await predictor.setImage(normalize(u8, SIZE, consts));
  });

  afterAll(async () => {
    await releaseSessions(sessions);
  });

  it("reproduces every prompt's four masks and IoU predictions", async () => {
    const [n, k, m] = meta.masks_shape;
    expect(n).toBe(meta.prompts.length);
    const area = m * m;
    const ious: number[] = [];
    let maxErr = 0;
    let maxIouPredErr = 0;
    let maxScoreErr = 0;
    for (const [p, prompt] of meta.prompts.entries()) {
      const labels = prompt.point_labels;
      const pts = prompt.point_coords_512.map(([x, y]) => ({ x, y }));
      // Exercise the box path: labels 2 / 3 are the box corners, placed first.
      const isBox = labels[0] === 2 && labels[1] === 3;
      const [c0, c1] = pts;
      const out = isBox
        ? await predictor.predict(pts.slice(2), labels.slice(2), {
            x0: c0!.x,
            y0: c0!.y,
            x1: c1!.x,
            y1: c1!.y,
          })
        : await predictor.predict(pts, labels);
      for (let t = 0; t < k; t++) {
        const off = (p * k + t) * area;
        const want = ref.subarray(off, off + area);
        const got = out.raw.masks.subarray(t * area, (t + 1) * area);
        const err = maxAbsDiff(got, want);
        maxErr = Math.max(maxErr, err);
        expect(err, `${prompt.name} token ${t} max abs error`).toBeLessThan(1e-2);
        if (want.some((v) => v > 0) || got.some((v) => v > 0)) {
          const iou = maskIou(got, want);
          ious.push(iou);
          expect(iou, `${prompt.name} token ${t} IoU`).toBeGreaterThanOrEqual(0.999);
        }
        const iouErr = Math.abs((out.raw.ious[t] as number) - (prompt.iou_predictions[t] as number));
        maxIouPredErr = Math.max(maxIouPredErr, iouErr);
        expect(iouErr, `${prompt.name} iou_predictions[${t}]`).toBeLessThan(1e-3);
      }
      maxScoreErr = Math.max(maxScoreErr, Math.abs(out.score - prompt.object_score_logits));
      // The selection agrees with the reference IoUs: multimask picks their argmax.
      if (labels.length <= 1) {
        const r = prompt.iou_predictions;
        const best = 1 + r.slice(1).indexOf(Math.max(...r.slice(1)));
        expect(out.index).toBe(best);
      }
    }
    const mean = ious.reduce((a, b) => a + b, 0) / ious.length;
    console.log(
      `[image parity] ${meta.prompts.length} prompts x ${k} masks: IoU min ${Math.min(...ious).toFixed(5)} ` +
        `mean ${mean.toFixed(5)} (${ious.length} non-empty), max|logit err| ${maxErr.toExponential(2)}, ` +
        `max|iou pred err| ${maxIouPredErr.toExponential(2)}, max|score err| ${maxScoreErr.toExponential(2)}; ` +
        `encoder ${predictor.timings["image_encoder"]?.[0]?.toFixed(0)} ms, ` +
        `head ${predictor.timings["sam_head"]?.map((v) => v.toFixed(0)).join("/")} ms`,
    );
  });

  it("feeds a previous output back as a mask prompt", async () => {
    const first = await predictor.predict([{ x: 142.4, y: 160.26 }], [1]);
    const again = await predictor.predict([{ x: 142.4, y: 160.26 }], [1], undefined, first.lowRes);
    expect(again.lowRes.length).toBe(consts.mask_size ** 2);
    expect(maskIou(again.lowRes, first.lowRes)).toBeGreaterThan(0.5);
  });
});
