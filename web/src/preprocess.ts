import type { Constants } from "./constants";

const f = Math.fround;

/**
 * Interleaved RGB(A) pixels of an S x S image -> normalised CHW `Float32Array [3, S, S]`.
 *
 * `(x / 255 - mean) / std` is computed in float64 (JavaScript numbers) and stored as
 * float32, which is bit-identical to the Python reference (`uint8 / 255.0` in float64,
 * then `astype(float32)`). `stride` is 3 for packed RGB, 4 for canvas RGBA.
 */
export function normalize(
  rgb: Uint8Array | Uint8ClampedArray,
  S: number,
  consts: Pick<Constants, "pixel_mean" | "pixel_std">,
  stride: 3 | 4 = 3,
): Float32Array {
  const n = S * S;
  if (rgb.length < n * stride) {
    throw new Error(`normalize: expected ${n * stride} bytes, got ${rgb.length}`);
  }
  const out = new Float32Array(3 * n);
  for (let c = 0; c < 3; c++) {
    const mean = consts.pixel_mean[c] as number;
    const std = consts.pixel_std[c] as number;
    const base = c * n;
    for (let i = 0; i < n; i++) {
      out[base + i] = ((rgb[i * stride + c] as number) / 255 - mean) / std;
    }
  }
  return out;
}

interface Taps {
  readonly i0: Int32Array;
  readonly i1: Int32Array;
  readonly w1: Float32Array;
  readonly w0: Float32Array;
}

/**
 * Source taps of bilinear resizing with half-pixel centres (`align_corners=False`):
 * source coordinate `(i + 0.5) * src / dst - 0.5`, clamped at 0, right tap clamped at
 * `src - 1`. Every step is rounded to float32 as in the numpy reference.
 */
function linearTaps(nIn: number, nOut: number): Taps {
  const scale = f(nIn / nOut);
  const i0 = new Int32Array(nOut);
  const i1 = new Int32Array(nOut);
  const w0 = new Float32Array(nOut);
  const w1 = new Float32Array(nOut);
  for (let i = 0; i < nOut; i++) {
    const src = Math.max(f(f(f(i + 0.5) * scale) - 0.5), 0);
    const lo = Math.floor(src);
    const w = f(src - lo);
    i0[i] = lo;
    i1[i] = Math.min(lo + 1, nIn - 1);
    w1[i] = w;
    w0[i] = f(1 - w);
  }
  return { i0, i1, w0, w1 };
}

export type Size = number | { readonly height: number; readonly width: number };

/**
 * Bilinear resize of a square `[srcSize, srcSize]` map to `dstSize` (a side length, or
 * height and width), as `F.interpolate(..., mode="bilinear", align_corners=False)`:
 * rows first, then columns, in float32. Used for the memory encoder's high-res mask input
 * (low-res logits to S x S) and for display.
 */
export function upsampleBilinear(src: Float32Array, srcSize: number, dstSize: Size): Float32Array {
  const h = typeof dstSize === "number" ? dstSize : dstSize.height;
  const w = typeof dstSize === "number" ? dstSize : dstSize.width;
  if (src.length < srcSize * srcSize) {
    throw new Error(`upsampleBilinear: expected ${srcSize * srcSize} values, got ${src.length}`);
  }
  const ty = linearTaps(srcSize, h);
  const tx = linearTaps(srcSize, w);
  const rows = new Float32Array(h * srcSize);
  for (let y = 0; y < h; y++) {
    const a = (ty.i0[y] as number) * srcSize;
    const b = (ty.i1[y] as number) * srcSize;
    const wa = ty.w0[y] as number;
    const wb = ty.w1[y] as number;
    const o = y * srcSize;
    for (let x = 0; x < srcSize; x++) {
      rows[o + x] = f((src[a + x] as number) * wa) + f((src[b + x] as number) * wb);
    }
  }
  const out = new Float32Array(h * w);
  for (let y = 0; y < h; y++) {
    const r = y * srcSize;
    const o = y * w;
    for (let x = 0; x < w; x++) {
      out[o + x] =
        f((rows[r + (tx.i0[x] as number)] as number) * (tx.w0[x] as number)) +
        f((rows[r + (tx.i1[x] as number)] as number) * (tx.w1[x] as number));
    }
  }
  return out;
}

/**
 * Round float32 values to the nearest bfloat16 (ties to even), returned as float32:
 * `u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000` on the bit pattern. The reference
 * stores memory features in bfloat16; this reproduces that storage.
 */
export function roundBf16(x: Float32Array): Float32Array {
  const out = new Float32Array(x);
  const u = new Uint32Array(out.buffer, out.byteOffset, out.length);
  for (let i = 0; i < u.length; i++) {
    const v = u[i] as number;
    // `&` works on int32; the Uint32Array store maps it back mod 2^32 (numpy's wrap).
    u[i] = (v + 0x7fff + ((v >>> 16) & 1)) & 0xffff0000;
  }
  return out;
}

interface AntialiasTaps {
  /** First source index of each destination index. */
  readonly start: Int32Array;
  /** Number of taps of each destination index. */
  readonly count: Int32Array;
  /** `[nOut, maxTaps]` normalised weights (float32, as the numpy reference stores them). */
  readonly weights: Float32Array;
  readonly maxTaps: number;
}

/**
 * One axis of `F.interpolate(mode="bilinear", align_corners=False, antialias=True)`: a
 * triangle filter of half-width `max(nIn / nOut, 1)` around each destination centre,
 * normalised over the taps inside the image (PyTorch's
 * `_compute_indices_min_size_weights_aa`; `resize_antialias` in tools/ort_video_loop.py).
 * Enlarging degenerates to plain bilinear interpolation.
 */
function antialiasTaps(nIn: number, nOut: number): AntialiasTaps {
  const scale = nIn / nOut;
  const support = Math.max(scale, 1);
  const inv = scale >= 1 ? 1 / scale : 1;
  const maxTaps = Math.ceil(2 * support) + 1;
  const start = new Int32Array(nOut);
  const count = new Int32Array(nOut);
  const weights = new Float32Array(nOut * maxTaps);
  for (let i = 0; i < nOut; i++) {
    const center = scale * (i + 0.5);
    const lo = Math.max(Math.trunc(center - support + 0.5), 0); // trunc: the C++ int cast
    const hi = Math.min(Math.trunc(center + support + 0.5), nIn);
    let total = 0;
    const row: number[] = [];
    for (let j = lo; j < hi; j++) {
      const w = Math.max(0, 1 - Math.abs((j - center + 0.5) * inv));
      row.push(w);
      total += w;
    }
    start[i] = lo;
    count[i] = hi - lo;
    for (let k = 0; k < row.length; k++) weights[i * maxTaps + k] = (row[k] as number) / total;
  }
  return { start, count, weights, maxTaps };
}

/**
 * `F.interpolate(x, size, mode="bilinear", align_corners=False, antialias=True)` of a
 * `[srcSize]` map to `dstSize` (side lengths, or heights and widths): rows first, then
 * columns, accumulated in float32. Used for mask prompts: any mask to `S x S` (threshold
 * 0.5) and the `+-10` logits down to the low-res side `S / 4`.
 */
export function resizeAntialias(src: Float32Array, srcSize: Size, dstSize: Size): Float32Array {
  const sh = typeof srcSize === "number" ? srcSize : srcSize.height;
  const sw = typeof srcSize === "number" ? srcSize : srcSize.width;
  const h = typeof dstSize === "number" ? dstSize : dstSize.height;
  const w = typeof dstSize === "number" ? dstSize : dstSize.width;
  if (src.length < sh * sw) {
    throw new Error(`resizeAntialias: expected ${sh * sw} values, got ${src.length}`);
  }
  const ty = antialiasTaps(sh, h);
  const tx = antialiasTaps(sw, w);
  const rows = new Float32Array(h * sw);
  for (let y = 0; y < h; y++) {
    const lo = ty.start[y] as number;
    const n = ty.count[y] as number;
    const o = y * sw;
    for (let k = 0; k < n; k++) {
      const wk = ty.weights[y * ty.maxTaps + k] as number;
      const r = (lo + k) * sw;
      for (let x = 0; x < sw; x++) rows[o + x] = f((rows[o + x] as number) + f((src[r + x] as number) * wk));
    }
  }
  const out = new Float32Array(h * w);
  for (let y = 0; y < h; y++) {
    const r = y * sw;
    const o = y * w;
    for (let x = 0; x < w; x++) {
      const lo = tx.start[x] as number;
      const n = tx.count[x] as number;
      let acc = 0;
      for (let k = 0; k < n; k++) {
        acc = f(acc + f((rows[r + lo + k] as number) * (tx.weights[x * tx.maxTaps + k] as number)));
      }
      out[o + x] = acc;
    }
  }
  return out;
}

/**
 * `SAM2.mask_downsample`: the checkpoint's 4x4 stride-4 conv (`mask_downsample_weight`,
 * row-major `[4, 4]`, and `mask_downsample_bias`) over a `[S, S]` 0/1 mask, giving the
 * head's `[S / 4, S / 4]` `mask_input` for a mask prompt.
 */
export function maskDownsample(mask: Float32Array, S: number, weight: Float32Array, bias: number): Float32Array {
  if (mask.length < S * S || S % 4 !== 0) throw new Error(`maskDownsample: need an [S, S] mask with S a multiple of 4`);
  if (weight.length !== 16) throw new Error("maskDownsample: weight must have 16 values");
  const s = S / 4;
  const out = new Float32Array(s * s);
  for (let i = 0; i < s; i++) {
    for (let j = 0; j < s; j++) {
      let acc = 0;
      for (let a = 0; a < 4; a++) {
        const r = (4 * i + a) * S + 4 * j;
        for (let b = 0; b < 4; b++) acc = f(acc + f((mask[r + b] as number) * (weight[4 * a + b] as number)));
      }
      out[i * s + j] = f(acc + bias);
    }
  }
  return out;
}
