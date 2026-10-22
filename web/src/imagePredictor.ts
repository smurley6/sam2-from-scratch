import type { Constants } from "./constants";
import { encodeImage, type ImageFeatures } from "./features";
import { clampMaskInput, LABEL_PAD, runHead, selectOutput, type HeadOutput, type Selection } from "./head";
import type { ImageSessionSet, Timings } from "./sessions";

export interface Point {
  readonly x: number;
  readonly y: number;
}

export interface Box {
  readonly x0: number;
  readonly y0: number;
  readonly x1: number;
  readonly y1: number;
}

export interface ImagePrediction extends Selection {
  /** All four raw mask tokens and IoUs, as the graph returned them. */
  readonly raw: HeadOutput;
}

/**
 * Promptable segmentation of one image (`ImagePredictor` of the Python package).
 * Coordinates are in S-pixel space: scale video / image pixels by `S / W`, `S / H` first.
 */
export class ImagePredictor {
  readonly timings: Timings = {};
  #feats: ImageFeatures | null = null;

  constructor(
    private readonly sessions: ImageSessionSet,
    private readonly consts: Constants,
  ) {}

  get hasImage(): boolean {
    return this.#feats !== null;
  }

  /** Encode a normalised `[3, S, S]` image (see `normalize`). */
  async setImage(input: Float32Array): Promise<void> {
    this.#feats = null;
    this.#feats = await encodeImage(this.sessions, input, this.consts.image_size, this.timings);
  }

  /**
   * Decode one prompt. A box becomes two leading points labelled 2 and 3; with no
   * points and no box a single padding point is passed (mask-only or empty prompt).
   * `maskInput` is a previous `lowRes` output; it is clamped to +-`mask_input_clamp`.
   */
  async predict(
    points: readonly Point[],
    labels: readonly number[],
    box?: Box,
    maskInput?: Float32Array,
  ): Promise<ImagePrediction> {
    const feats = this.#feats;
    if (feats === null) throw new Error("ImagePredictor: call setImage first");
    if (points.length !== labels.length) {
      throw new Error("ImagePredictor: one label per point");
    }
    const coords: number[] = [];
    const labs: number[] = [];
    if (box !== undefined) {
      coords.push(box.x0, box.y0, box.x1, box.y1);
      labs.push(2, 3);
    }
    for (let i = 0; i < points.length; i++) {
      const p = points[i] as Point;
      coords.push(p.x, p.y);
      labs.push(labels[i] as number);
    }
    const numPoints = labs.length;
    if (numPoints === 0) {
      coords.push(0, 0);
      labs.push(LABEL_PAD);
    }
    const mask =
      maskInput === undefined ? null : clampMaskInput(maskInput, this.consts.mask_input_clamp);
    const raw = await runHead(this.sessions, feats, coords, labs, mask, this.timings);
    const sel = selectOutput(raw.masks, raw.ious, raw.tokens, raw.score, numPoints, this.consts);
    return { ...sel, raw };
  }
}
