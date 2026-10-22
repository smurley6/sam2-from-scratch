import { describe, expect, it } from "vitest";
import { maskDownsample, resizeAntialias, upsampleBilinear } from "../src/preprocess";

/* Reference values from tools/ort_video_loop.py (`resize_antialias`, `mask_downsample`)
   on seeded numpy inputs; the two sides accumulate in a different order, so the match is
   to float32 round-off rather than bit for bit. */

// prettier-ignore
const X = Float32Array.from([
  20.409191, -25.55665, 4.1809883, -5.6776958, -4.526493, -2.1559718, -20.199862, -2.3193238, -8.652131, 33.229996,
  2.2578661, -3.526308, -2.8128743, -6.680464, -10.551505, -3.9080098, 4.819454, -2.3855362, 9.577587, -1.9980214,
  0.24259566, 15.458208, 5.451055, -5.0522876, -1.8283898, 5.4052515, 19.35088, -2.6962032, -2.4355867, 10.023136,
  -8.864599, -2.9172025, 8.82539, 5.8035, 0.91516703, 6.701044, -28.281622, 10.213069, -9.596447, -16.6862,
  2.7644575, 7.005449, -4.4476743, -10.764059, 0.26124832, -0.5274731, 14.055982, 7.4740796, 1.9381565, 11.116332,
  -2.0552304, -9.258996, 5.840583, 5.825384, -2.1482892, -7.828086, 2.291539, -24.938944, 6.9012475, 4.9136825,
]);
// prettier-ignore
const DOWN = [
  -1.0568793, -4.4054365, -3.1744015, 4.0146503, 2.1969695, 0.43334436, 1.1322913, -0.9490933, -0.50812316,
  -0.6519346, -0.85676956, 1.1816633,
];
// prettier-ignore
const UP = [
  2.0409191e1, -9.6453972, -1.2975341e1, 2.2850873, -5.2985153, -4.8364325, -3.3412323, -7.0139422, -1.9512148e1,
  -5.7578888, -5.9728661, 5.8455286, 3.3229996e1, 6.1474361, -3.2643733, -5.3139429, -2.3048425, -6.2674541,
  -8.5079727, -6.396502, -2.7273722, -6.1219394e-1, -2.0195169, 2.2685945, 5.629549, 5.5508394, 9.6233517e-1,
  6.0072083, 6.0640965, 9.3553019e-1, -5.3209548, -5.1295547, -1.4323508, 5.3319325, 1.3516994e1, 6.3519859e-1,
  -2.3664236e-2, 3.1961901, 5.7298656, -4.3110018, 2.6076741, 6.6376152, 5.837719, 6.357069e-1, -2.3255275e-1,
  2.7982683, 3.2212391, -4.1490707, 2.1769319, -1.8806727, -5.086772, -3.3315315, -4.7113647, -1.221156,
  2.0897741, 3.2776067, 4.7995284e-2, 4.675566e-1, 2.400528, -5.3300595e-1, -1.2299667e1, 4.9279528,
  7.4727917e-1, -5.9199457, -6.7567234, 1.731667, 2.9010992, 1.0819089, -3.1980729, -7.0181732, -2.1273382,
  -1.1734856, 1.5768962, 1.1111699e1, 2.6450849, 1.9553033, 5.3505087, 9.7871923, -2.0552304, -6.7653847,
  -2.8707123, 5.8376598, 5.8259687, -1.5308857e-3, -4.9881878, -5.1035714, 1.2442126, -1.9702311e1, -6.569603,
  6.213244, 4.9136825,
];

const close = (got: Float32Array, want: readonly number[], tol: number) => {
  expect(got.length).toBe(want.length);
  for (let i = 0; i < want.length; i++) expect(Math.abs((got[i] as number) - (want[i] as number)), `index ${i}`).toBeLessThan(tol);
};

describe("resizeAntialias", () => {
  it("downsamples 6x10 -> 3x4 like the numpy reference (triangle filter, renormalised borders)", () => {
    close(resizeAntialias(X, { height: 6, width: 10 }, { height: 3, width: 4 }), DOWN, 2e-5);
  });

  it("enlarges 6x10 -> 7x13 like the numpy reference, and equals plain bilinear interpolation", () => {
    const got = resizeAntialias(X, { height: 6, width: 10 }, { height: 7, width: 13 });
    close(got, UP, 2e-5);
    // A square case against upsampleBilinear itself (bit for bit: both are two taps per axis).
    const sq = Float32Array.from([0, 1, 2, 3]);
    expect(Array.from(resizeAntialias(sq, 2, 4))).toEqual(Array.from(upsampleBilinear(sq, 2, 4)));
  });

  it("a 4x downsample of a constant map is the constant (to round-off at the renormalised borders)", () => {
    const c = new Float32Array(64 * 64).fill(-10);
    const out = resizeAntialias(c, 64, 16);
    expect(out.length).toBe(256);
    for (const v of out) expect(Math.abs(v + 10)).toBeLessThan(1e-5);
  });

  it("rejects a short source", () => {
    expect(() => resizeAntialias(new Float32Array(3), 2, 1)).toThrow();
  });
});

describe("maskDownsample", () => {
  it("is the 4x4 stride-4 conv of the numpy reference", () => {
    // prettier-ignore
    const m = Float32Array.from([
      1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1,
      1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ]);
    // prettier-ignore
    const w = Float32Array.from([
      0.0432959, 0.64097106, 2.0478861, -0.19744544, 0.76750255, 0.15541781, 1.7599263, 0.7421579,
      1.3685504, -1.0776752, -0.19224072, -0.81377244, 1.5049474, 0.6576399, -0.30514443, -0.4524679,
    ]);
    close(maskDownsample(m, 8, w, 0.25), [6.7648063, 0.660187, 4.034928, 3.3686314], 2e-6);
  });

  it("rejects the wrong shapes", () => {
    expect(() => maskDownsample(new Float32Array(16), 4, new Float32Array(15), 0)).toThrow();
    expect(() => maskDownsample(new Float32Array(9), 3, new Float32Array(16), 0)).toThrow();
  });
});
