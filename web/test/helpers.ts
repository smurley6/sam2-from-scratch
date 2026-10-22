import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-web";
import { parseConstants, type Constants } from "../src/constants";
import { createSessions, setOrt, type SessionSet } from "../src/sessions";

export const EXPORTS = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "exports");
export const REFERENCE = join(EXPORTS, "browser_reference");
export const SIZE = 512;

/** Graphs, constants and reference dumps are build outputs; skip the parity suites without them. */
export const haveExports =
  existsSync(join(EXPORTS, `constants_${SIZE}.json`)) &&
  existsSync(join(EXPORTS, `image_encoder_${SIZE}.int8.onnx`)) &&
  existsSync(REFERENCE);

/** Threads for the wasm backend in Node (0 = ORT's default). The browser uses 1. */
const threads = Number(process.env["SAM2_THREADS"] ?? 0);

export async function bytes(path: string): Promise<Uint8Array> {
  const b = await readFile(path);
  return new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
}

export async function loadConstants(): Promise<Constants> {
  return parseConstants(await readFile(join(EXPORTS, `constants_${SIZE}.json`), "utf8"));
}

export async function loadSessions(): Promise<SessionSet> {
  ort.env.wasm.numThreads = threads;
  ort.env.logLevel = "error";
  setOrt(ort);
  return createSessions((file) => bytes(join(EXPORTS, file)), { provider: "wasm", size: SIZE });
}

export async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(await readFile(path, "utf8")) as T;
}

export async function readF32(path: string): Promise<Float32Array> {
  const b = await readFile(path);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

/** IoU of two logit maps at threshold 0 (1 when both are empty). */
export function maskIou(a: Float32Array, b: Float32Array): number {
  let inter = 0;
  let union = 0;
  for (let i = 0; i < a.length; i++) {
    const pa = (a[i] as number) > 0;
    const pb = (b[i] as number) > 0;
    if (pa && pb) inter++;
    if (pa || pb) union++;
  }
  return union === 0 ? 1 : inter / union;
}

export function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs((a[i] as number) - (b[i] as number)));
  return m;
}

export const median = (v: readonly number[]): number => {
  const s = [...v].sort((x, y) => x - y);
  const k = Math.floor(s.length / 2);
  return s.length % 2 ? (s[k] as number) : ((s[k - 1] as number) + (s[k] as number)) / 2;
};
