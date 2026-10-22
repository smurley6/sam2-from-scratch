import { describe, expect, it } from "vitest";
import { parseConstants } from "../src/constants";
import { clampMaskInput, selectOutput, stabilityScore } from "../src/head";
import { normalize, roundBf16, upsampleBilinear } from "../src/preprocess";
import { nonOverlapping } from "../src/multiObjectTracker";
import { selectClosestCondFrames } from "../src/videoTracker";

const bits = (x: Float32Array): number[] => Array.from(new Uint32Array(x.buffer, x.byteOffset, x.length));
const fromBits = (u: readonly number[]): Float32Array => new Float32Array(Uint32Array.from(u).buffer);

const CONSTS = {
  multimask_min_pt_num: 0,
  multimask_max_pt_num: 1,
  stability_delta: 0.05,
  stability_thresh: 0.98,
  no_obj_score: -1024,
};

/** 4 tokens of 2x2 masks, token k filled with `fill[k]`, tokens of dim 2 holding k. */
function head(fill: readonly (number | readonly number[])[]): { masks: Float32Array; tokens: Float32Array } {
  const masks = new Float32Array(16);
  fill.forEach((v, k) => {
    for (let i = 0; i < 4; i++) masks[k * 4 + i] = typeof v === "number" ? v : (v[i] as number);
  });
  const tokens = Float32Array.from([0, 0, 1, 1, 2, 2, 3, 3]);
  return { masks, tokens };
}

describe("selectOutput", () => {
  it("multimask: first argmax of ious[1..3], its mask and token", () => {
    const { masks, tokens } = head([5, 1, 2, 3]);
    const ious = Float32Array.from([0.99, 0.5, 0.8, 0.8]);
    for (const n of [0, 1]) {
      const s = selectOutput(masks, ious, tokens, 3.5, n, CONSTS);
      expect(s.multimask).toBe(true);
      expect(s.index).toBe(2);
      expect(Array.from(s.lowRes)).toEqual([2, 2, 2, 2]);
      expect(Array.from(s.token)).toEqual([2, 2]);
      expect(s.iou).toBeCloseTo(0.8, 6);
      expect(s.candidates.map((c) => c.mask[0])).toEqual([1, 2, 3]);
    }
  });

  it("single mask: token 0 when stable (and when the denominator is 0)", () => {
    const ious = Float32Array.from([0.7, 0.9, 0.1, 0.2]);
    for (const m0 of [1, -1]) {
      const { masks, tokens } = head([m0, 1, 2, 3]);
      const s = selectOutput(masks, ious, tokens, 2, 2, CONSTS);
      expect(s.multimask).toBe(false);
      expect(s.index).toBe(0);
      expect(Array.from(s.lowRes)).toEqual([m0, m0, m0, m0]);
      expect(Array.from(s.token)).toEqual([0, 0]);
      expect(s.iou).toBeCloseTo(0.7, 6);
    }
  });

  it("single mask: unstable token 0 falls back to the best candidate, token stays 0", () => {
    const { masks, tokens } = head([[1, 1, 0, 0.03], 1, 2, 3]);
    expect(stabilityScore(masks.subarray(0, 4), 0.05)).toBe(0.5);
    const ious = Float32Array.from([0.95, 0.2, 0.3, 0.9]);
    const s = selectOutput(masks, ious, tokens, 2, 3, CONSTS);
    expect(s.index).toBe(3);
    expect(Array.from(s.lowRes)).toEqual([3, 3, 3, 3]);
    expect(Array.from(s.token)).toEqual([0, 0]);
    expect(s.iou).toBeCloseTo(0.9, 6);
  });

  it("absent object: the chosen mask becomes no_obj_score (score <= 0)", () => {
    const { masks, tokens } = head([5, 1, 2, 3]);
    const ious = Float32Array.from([0.9, 0.5, 0.8, 0.1]);
    for (const score of [0, -4]) {
      const s = selectOutput(masks, ious, tokens, score, 1, CONSTS);
      expect(Array.from(s.lowRes)).toEqual([-1024, -1024, -1024, -1024]);
      expect(Array.from(s.token)).toEqual([2, 2]);
      expect(masks[8]).toBe(2); // head output untouched
    }
  });

  it("stability ratio is computed in float32", () => {
    const m = Float32Array.from([1, 1, 0]);
    expect(stabilityScore(m, 0.05)).toBe(Math.fround(2 / 3));
  });
});

describe("roundBf16", () => {
  it("rounds to nearest, ties to even, on the bit pattern", () => {
    const input = [
      0x3f800000, 0x3f808000, 0x3f818000, 0x3f807fff, 0x3f808001, 0x40490fdb, 0x7f7fffff,
      0xbf808001, 0x00000001, 0x80000000,
    ];
    // Expected values from the numpy reference (round_bf16 in tools/ort_video_loop.py).
    const expected = [
      0x3f800000, 0x3f800000, 0x3f820000, 0x3f800000, 0x3f810000, 0x40490000, 0x7f800000,
      0xbf810000, 0x00000000, 0x80000000,
    ];
    const x = fromBits(input);
    expect(bits(roundBf16(x))).toEqual(expected);
    expect(bits(x)).toEqual(input); // input not modified
  });
});

describe("upsampleBilinear", () => {
  const src = Float32Array.from([0, 1, 2, 3]);

  it("2x2 -> 4x4 with half-pixel centres and edge clamp", () => {
    expect(Array.from(upsampleBilinear(src, 2, 4))).toEqual([
      0, 0.25, 0.75, 1, 0.5, 0.75, 1.25, 1.5, 1.5, 1.75, 2.25, 2.5, 2, 2.25, 2.75, 3,
    ]);
  });

  it("non-square destination", () => {
    expect(Array.from(upsampleBilinear(src, 2, { height: 2, width: 3 }))).toEqual([
      0, 0.5, 1, 2, 2.5, 3,
    ]);
  });

  it("is bit-identical to the numpy reference on a random 5x5 -> 7x13 case", () => {
    const x = fromBits([
      1040236396, 3188147839, 1059320509, 1037489631, 3205046689, 1052320509, 1067903353,
      1064465382, 3207866366, 3215063381, 3206516458, 1026114926, 3222588750, 3193965292,
      3214899715, 3208345056, 3205190799, 3198284312, 1053999421, 1065709844, 3187908247,
      1068427334, 3207219763, 1051982113, 1063733714,
    ]);
    const expected = [
      1040236396, 1037623397, 1004307040, 3183300633, 1027415434, 1051706516, 1059320509,
      1054765196, 1047131964, 1003494176, 3195390036, 3203991746, 3205046689, 1048928661,
      1050033418, 1055557205, 1059022799, 1060619760, 1061440098, 1062260436, 1052420205,
      3182506810, 3202729988, 3207431707, 3211274293, 3212042808, 1034174932, 1041636324,
      1056244069, 1062173291, 1060796180, 1052612114, 1011334912, 3193366866, 3202164554,
      3207334807, 3211817338, 3214568367, 3215016619, 3206516458, 3205658755, 3198292218,
      3178858366, 3204528056, 3216317243, 3222588750, 3217156484, 3207885013, 3200314690,
      3209009244, 3214236940, 3214899715, 3207822600, 3207404232, 3205312387, 3201992829,
      3204293802, 3207683069, 3210995110, 3203065382, 3168800760, 1048766916, 1050795258,
      1052823599, 1053229268, 3203559894, 3201628845, 3187887546, 1042504161, 1037335172,
      3191431720, 3203301580, 3188560426, 1044503875, 1056217010, 1060443496, 1064296182,
      1065066718, 3187908247, 3160260896, 1057995812, 1066497957, 1063635550, 1039007932,
      3207219763, 3196870134, 1039098346, 1054831454, 1059459706, 1063021381, 1063733714,
    ];
    expect(bits(upsampleBilinear(x, 5, { height: 7, width: 13 }))).toEqual(expected);
  });
});

describe("normalize", () => {
  // A 3x3 RGB image; expected float32 bit patterns from the Python reference
  // ((u8 / 255.0 - mean) / std in float64, then astype(float32), CHW).
  const u8 = Uint8Array.from([
    0, 1, 2, 127, 128, 129, 254, 255, 77, 13, 200, 99, 255, 0, 128, 1, 254, 3, 50, 60, 70, 80,
    90, 100, 110, 120, 130,
  ]);
  const expected = [
    0xc0078bbd, 0x3d6939b7, 0x400ed58b, 0xbff2989b, 0x400fee1d, 0xc006732b, 0xbfa17e48,
    0xbf3f77ec, 0xbe6fcd25, 0xc0012a4f, 0x3e521b40, 0x401b6db7, 0x3fbb9b9c, 0xc0024925,
    0x401a4ee1, 0xbf7c3c3c, 0xbeeb9022, 0x3d8560cf, 0xbfe281cc, 0x3ee349b0, 0xbeecbf3c,
    0xbda1b2c4, 0x3eda5d35, 0xbfe046ad, 0xbf159b4c, 0xbd7c01b2, 0x3eec362b,
  ];
  const consts = {
    pixel_mean: [0.485, 0.456, 0.406] as const,
    pixel_std: [0.229, 0.224, 0.225] as const,
  };

  it("is bit-identical to the Python path (RGB)", () => {
    expect(bits(normalize(u8, 3, consts))).toEqual(expected);
  });

  it("reads RGBA with stride 4", () => {
    const rgba = new Uint8ClampedArray(36);
    for (let i = 0; i < 9; i++) {
      rgba.set(u8.subarray(3 * i, 3 * i + 3), 4 * i);
      rgba[4 * i + 3] = 255;
    }
    expect(bits(normalize(rgba, 3, consts, 4))).toEqual(expected);
  });
});

describe("misc", () => {
  it("clampMaskInput clamps to +-32", () => {
    expect(Array.from(clampMaskInput(Float32Array.from([-1024, -3, 0, 40]), 32))).toEqual([
      -32, -3, 0, 32,
    ]);
  });

  it("selectClosestCondFrames keeps every frame for -1 and the nearest otherwise", () => {
    const cond = new Map([
      [10, "a"],
      [2, "b"],
      [30, "c"],
      [20, "d"],
    ]);
    const [all, none] = selectClosestCondFrames(12, cond, -1);
    expect([...all.keys()]).toEqual([10, 2, 30, 20]);
    expect(none.size).toBe(0);
    const [sel, unsel] = selectClosestCondFrames(12, cond, 3);
    expect([...sel.keys()]).toEqual([10, 20, 2]);
    expect([...unsel.keys()]).toEqual([30]);
  });

  it("parseConstants keeps tensors bit-exact and the pixel stats as float64", () => {
    const c = parseConstants({
      model: "m", image_size: 512, embed_grid: 32, mask_size: 128, hidden_dim: 2, mem_dim: 1,
      tokens_per_obj_ptr: 4, num_maskmem: 2, max_obj_ptrs_in_encoder: 16,
      max_cond_frames_in_attn: -1, memory_temporal_stride_for_eval: 1,
      only_obj_ptrs_in_the_past_for_eval: true, pixel_mean: [0.485, 0.456, 0.406],
      pixel_std: [0.229, 0.224, 0.225], multimask_min_pt_num: 0, multimask_max_pt_num: 1,
      stability_delta: 0.05, stability_thresh: 0.98, no_obj_score: -1024, mask_input_clamp: 32,
      sigmoid_scale_for_mem_enc: 20, sigmoid_bias_for_mem_enc: -10,
      no_mem_embed: [0.01892760396003723, -0.5], maskmem_tpos_enc: [[0.280518501996994], [1]],
      no_obj_ptr: [1, 2], no_obj_embed_spatial: [3],
    });
    expect(c.no_mem_embed[0]).toBe(0.01892760396003723);
    expect(c.maskmem_tpos_enc[0]?.[0]).toBe(0.280518501996994);
    expect(c.pixel_mean[0]).toBe(0.485);
    expect(() => parseConstants({ ...JSON.parse("{}") })).toThrow();
  });
});

describe("nonOverlapping", () => {
  it("keeps only the best object per pixel and clamps the rest to -10 (the torch test case)", () => {
    const a = Float32Array.from([5, -1, 2]);
    const b = Float32Array.from([3, 4, 2.5]);
    const out = nonOverlapping(new Map([[7, a], [9, b]]));
    expect([...out.keys()]).toEqual([7, 9]);
    expect(Array.from(out.get(7) as Float32Array)).toEqual([5, -10, -10]);
    expect(Array.from(out.get(9) as Float32Array)).toEqual([-10, 4, 2.5]);
    // Inputs are untouched; a losing logit already below -10 stays where it is.
    expect(Array.from(a)).toEqual([5, -1, 2]);
    const low = nonOverlapping(new Map([[1, Float32Array.of(-20)], [2, Float32Array.of(1)]]));
    expect(Array.from(low.get(1) as Float32Array)).toEqual([-20]);
  });

  it("ties go to the earlier object, as torch.argmax picks the first maximum", () => {
    const out = nonOverlapping(new Map([[2, Float32Array.of(1)], [1, Float32Array.of(1)]]));
    expect(Array.from(out.get(2) as Float32Array)).toEqual([1]);
    expect(Array.from(out.get(1) as Float32Array)).toEqual([-10]);
  });

  it("one object or none is returned as is", () => {
    const a = Float32Array.from([-30, 2]);
    const one = nonOverlapping(new Map([[3, a]]));
    expect(one.get(3)).toBe(a);
    expect(nonOverlapping(new Map()).size).toBe(0);
  });
});
