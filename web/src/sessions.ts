/**
 * ONNX Runtime plumbing, independent of which ORT build is used.
 *
 * The runtime only needs `InferenceSession.create`, `Tensor` and `session.run`, so it is
 * typed against that minimal surface. The browser passes `onnxruntime-web` (after setting
 * `env.wasm.wasmPaths`), tests pass whatever build runs in Node. Call `setOrt` once.
 */

export type TensorType = "float32" | "int32" | "int64";
export type TensorData = Float32Array | Int32Array | BigInt64Array;

export interface OrtTensor {
  readonly dims: readonly number[];
  readonly type: string;
  readonly data: unknown;
}

export interface OrtSession {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
  release?(): Promise<void>;
}

export interface OrtModule {
  readonly InferenceSession: {
    create(model: Uint8Array, options?: Record<string, unknown>): Promise<OrtSession>;
  };
  readonly Tensor: new (type: TensorType, data: TensorData, dims: readonly number[]) => OrtTensor;
}

let ortModule: OrtModule | null = null;

/** Register the ORT module (`import * as ort from "onnxruntime-web"`). */
export function setOrt(ort: OrtModule): void {
  ortModule = ort;
}

export function getOrt(): OrtModule {
  if (ortModule === null) throw new Error("sam2: call setOrt(ort) before creating sessions");
  return ortModule;
}

export function f32(data: Float32Array, dims: readonly number[]): OrtTensor {
  return new (getOrt().Tensor)("float32", data, dims);
}

export function i32(data: Int32Array, dims: readonly number[]): OrtTensor {
  return new (getOrt().Tensor)("int32", data, dims);
}

export function i64(data: BigInt64Array, dims: readonly number[]): OrtTensor {
  return new (getOrt().Tensor)("int64", data, dims);
}

/** The float32 payload of an output tensor. */
export function data32(t: OrtTensor | undefined, name: string): Float32Array {
  if (t === undefined) throw new Error(`sam2: missing output "${name}"`);
  if (!(t.data instanceof Float32Array)) throw new Error(`sam2: output "${name}" is not float32`);
  return t.data;
}

/** Per-graph wall-clock times in ms, appended on every run. */
export type Timings = Record<string, number[]>;

/** Run a session and return the named float32 outputs. */
export async function runGraph(
  session: OrtSession,
  feeds: Record<string, OrtTensor>,
  timings?: Timings,
  label?: string,
): Promise<Record<string, Float32Array>> {
  const t0 = performance.now();
  const out = await session.run(feeds);
  if (timings !== undefined && label !== undefined) {
    (timings[label] ??= []).push(performance.now() - t0);
  }
  const result: Record<string, Float32Array> = {};
  for (const name of session.outputNames) result[name] = data32(out[name], name);
  return result;
}

export interface ImageSessionSet {
  readonly imageEncoder: OrtSession;
  readonly samHead: OrtSession;
}

export interface SessionSet extends ImageSessionSet {
  readonly memoryEncoder: OrtSession;
  readonly memoryAttention: OrtSession;
  readonly objectPointer: OrtSession;
  readonly objectPointerTpos: OrtSession;
}

export type GraphRole = keyof SessionSet;
export type Provider = "webgpu" | "wasm";

/** Shipped file per graph at size S (int8 encoder, fp32 head and memory graphs). */
export function graphFiles(size = 512): Record<GraphRole, string> {
  return {
    imageEncoder: `image_encoder_${size}.int8.onnx`,
    samHead: `sam_head_${size}.onnx`,
    memoryEncoder: `memory_encoder_${size}.onnx`,
    memoryAttention: `memory_attention_${size}.onnx`,
    objectPointer: "object_pointer.onnx",
    objectPointerTpos: "object_pointer_tpos.onnx",
  };
}

/**
 * Fetches one graph. The optional callback reports that file's download fraction; a
 * loader that cannot report progress may ignore it.
 */
export type LoadBytes = (file: string, onProgress?: (frac: number) => void) => Promise<Uint8Array>;

export interface SessionOptions {
  readonly provider: Provider;
  /** Called with each file name and its fraction (download, then 1 once the session exists). */
  onProgress?(file: string, frac: number): void;
  /** Input size of the size-dependent graphs (default 512). */
  readonly size?: number;
  /** Override individual file names. */
  readonly files?: Partial<Record<GraphRole, string>>;
}

function sessionOptions(provider: Provider): Record<string, unknown> {
  return {
    executionProviders: provider === "webgpu" ? ["webgpu", "wasm"] : ["wasm"],
    // Fold the int8 weights' DequantizeLinear once at load instead of on every run.
    extra: { session: { disable_quant_qdq: "1" } },
  };
}

async function createRoles<R extends GraphRole>(
  roles: readonly R[],
  loadBytes: LoadBytes,
  opts: SessionOptions,
): Promise<Record<R, OrtSession>> {
  const ort = getOrt();
  const files = { ...graphFiles(opts.size), ...opts.files };
  const out: Partial<Record<R, OrtSession>> = {};
  // Sequential on purpose: one model's bytes are released before the next download.
  for (const role of roles) {
    const file = files[role];
    opts.onProgress?.(file, 0);
    const bytes = await loadBytes(file, (frac) => opts.onProgress?.(file, Math.min(frac, 0.999)));
    out[role] = await ort.InferenceSession.create(bytes, sessionOptions(opts.provider));
    opts.onProgress?.(file, 1);
  }
  return out as Record<R, OrtSession>;
}

/** The two graphs of image mode (encoder + SAM head). */
export function createImageSessions(
  loadBytes: LoadBytes,
  opts: SessionOptions,
): Promise<ImageSessionSet> {
  return createRoles(["imageEncoder", "samHead"] as const, loadBytes, opts);
}

/** All six graphs (image and video mode). */
export function createSessions(loadBytes: LoadBytes, opts: SessionOptions): Promise<SessionSet> {
  return createRoles(
    [
      "imageEncoder",
      "samHead",
      "memoryEncoder",
      "memoryAttention",
      "objectPointer",
      "objectPointerTpos",
    ] as const,
    loadBytes,
    opts,
  );
}

/** Release every session of a set (frees wasm / GPU memory). */
export async function releaseSessions(set: Partial<SessionSet>): Promise<void> {
  for (const s of Object.values(set)) await s?.release?.();
}
