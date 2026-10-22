/**
 * Browser glue: the only module that touches the DOM. Nothing here runs at import time.
 */
import { upsampleBilinear } from "./preprocess";
import { setOrt, type LoadBytes, type OrtModule, type Provider } from "./sessions";

/** Draw any image / video / canvas source onto an S x S canvas (aspect not kept). */
export function resizeToSquare(source: CanvasImageSource, S: number): Uint8ClampedArray {
  let ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null;
  if (typeof OffscreenCanvas !== "undefined") {
    ctx = new OffscreenCanvas(S, S).getContext("2d", { willReadFrequently: true });
  } else {
    const canvas = document.createElement("canvas");
    canvas.width = S;
    canvas.height = S;
    ctx = canvas.getContext("2d", { willReadFrequently: true });
  }
  if (ctx === null) throw new Error("resizeToSquare: no 2D canvas context");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, S, S);
  return ctx.getImageData(0, 0, S, S).data;
}

interface GpuNavigator {
  readonly gpu?: { requestAdapter(): Promise<unknown> };
}

/** WebGPU when an adapter is available, else WebAssembly. */
export async function pickProvider(): Promise<Provider> {
  const gpu = (globalThis.navigator as GpuNavigator | undefined)?.gpu;
  if (gpu === undefined) return "wasm";
  try {
    return (await gpu.requestAdapter()) ? "webgpu" : "wasm";
  } catch {
    return "wasm";
  }
}

interface OrtEnv {
  readonly env: { readonly wasm: { wasmPaths?: unknown; numThreads?: number } };
}

/**
 * Point ORT Web at the jsDelivr copy of its wasm binaries (the site's convention), use one
 * thread (no cross-origin isolation needed), and register the module with the runtime.
 */
export function configureOrt(ort: OrtModule & OrtEnv, version: string): void {
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${version}/dist/`;
  ort.env.wasm.numThreads = 1;
  setOrt(ort);
}

/**
 * A `LoadBytes` that fetches `${baseUrl}/${file}` with download progress. Progress is
 * measured against `sizes[file]` when given (exact, and works when the server sends no
 * content-length or a compressed one), else against the content-length header.
 */
export function fetchLoader(baseUrl: string, sizes?: Readonly<Record<string, number>>): LoadBytes {
  const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  return async (file, onProgress) => {
    const res = await fetch(base + file);
    if (!res.ok || res.body === null) throw new Error(`Could not download ${file} (${res.status})`);
    const total = sizes?.[file] ?? (Number(res.headers.get("content-length")) || 0);
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      if (total > 0) onProgress?.(Math.min(got / total, 1));
    }
    const buf = new Uint8Array(got);
    let off = 0;
    for (const c of chunks) {
      buf.set(c, off);
      off += c.length;
    }
    return buf;
  };
}

/**
 * Low-res logits `[m, m]` -> RGBA overlay at display size: bilinear upsample, then
 * `color` where the logit is > 0 and transparent elsewhere.
 */
export function maskToImageData(
  lowRes: Float32Array,
  m: number,
  width: number,
  height: number,
  color: readonly [number, number, number, number],
): ImageData {
  const logits = upsampleBilinear(lowRes, m, { height, width });
  const rgba = new Uint8ClampedArray(width * height * 4);
  const [r, g, b, a] = color;
  for (let i = 0; i < logits.length; i++) {
    if ((logits[i] as number) > 0) {
      const o = i * 4;
      rgba[o] = r;
      rgba[o + 1] = g;
      rgba[o + 2] = b;
      rgba[o + 3] = a;
    }
  }
  return new ImageData(rgba, width, height);
}
