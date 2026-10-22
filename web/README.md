# sam2-web

Framework-free TypeScript runtime for the SAM 2.1 Hiera-T ONNX graphs in `../exports`:
promptable image segmentation and single-object video tracking, running on ONNX Runtime Web
(WebGPU, WebAssembly fallback). `src/` has no framework and no Node imports; only
`src/browser.ts` touches the DOM, and only when its functions are called.

## API

```ts
import * as ort from "onnxruntime-web";
import {
  configureOrt, createSessions, fetchLoader, parseConstants, pickProvider,
  normalize, resizeToSquare, ImagePredictor, VideoTracker, maskToImageData,
} from "./sam2";

configureOrt(ort, "1.22.0");                       // wasmPaths on jsDelivr, 1 thread, setOrt
const consts = parseConstants(await (await fetch("/sam2/constants_512.json")).text());
const sessions = await createSessions(fetchLoader("/sam2"), {
  provider: await pickProvider(),
  onProgress: (file, frac) => {},
});

// Image mode (createImageSessions loads only the encoder and head).
const S = consts.image_size;
const predictor = new ImagePredictor(sessions, consts);
await predictor.setImage(normalize(resizeToSquare(img, S), S, consts, 4));
const out = await predictor.predict([{ x, y }], [1]);   // points / box in S-pixel space
out.candidates;                                         // three multimask masks + IoUs
const overlay = maskToImageData(out.lowRes, consts.mask_size, w, h, [30, 144, 255, 140]);

// Video mode.
const frames = { count, get: async (i) => normalize(resizeToSquare(frame[i], S), S, consts, 4) };
const tracker = new VideoTracker(sessions, consts, frames, { width, height });
await tracker.addClick(0, [{ x, y }], [1]);              // video pixels; 1 positive, 0 negative
for await (const f of tracker.propagate()) {}            // { frame, lowRes, score, isConditioning }
for await (const f of tracker.propagate(t, true)) {}     // reverse from frame t
```

| Module | Contents |
|--------|----------|
| `constants.ts` | `Constants`, `parseConstants` (tensors via `Float32Array.from`, bit-exact) |
| `sessions.ts` | `setOrt`, `createSessions` / `createImageSessions`, `SessionSet`, file names, minimal ORT types |
| `preprocess.ts` | `normalize` (float64 math, float32 storage), `upsampleBilinear` (half-pixel, float32), `roundBf16` |
| `features.ts` | `encodeImage`, `withMemoryTokens` (`vision_feat`, `curr`) |
| `head.ts` | `runHead`, `selectOutput` (multimask / stability / absent-object rules), `clampMaskInput` |
| `imagePredictor.ts` | `ImagePredictor.setImage` / `predict` (box = leading labels 2 and 3) |
| `videoTracker.ts` | `VideoTracker`: port of `OrtVideoTracker` in `tools/ort_video_loop.py` |
| `browser.ts` | `resizeToSquare`, `pickProvider`, `configureOrt`, `fetchLoader`, `maskToImageData` |
| `index.ts` | re-exports |

Graphs loaded (S = 512 by default, `size` option for 1024): `image_encoder_512.int8.onnx`,
`sam_head_512.onnx`, `memory_encoder_512.onnx`, `memory_attention_512.onnx`,
`object_pointer.onnx`, `object_pointer_tpos.onnx`, all with the session config entry
`session.disable_quant_qdq = "1"` so the int8 encoder weights are dequantised once at load.
The runtime calls only `InferenceSession.create`, `new Tensor` and `session.run`, so any ORT
build works; register it once with `setOrt` (`configureOrt` does this in the browser).

A `VideoTracker` processes one call at a time: await `addClick` and finish (or `break` out
of) a `propagate` loop before the next call.

## Tests

```sh
npm install
npm run typecheck     # whole package, then src/ alone without Node types
npm test              # unit tests + Node parity against exports/browser_reference
SAM2_FULL=1 npm test  # video parity on all 48 frames instead of the first 12
```

The parity suites need the build outputs `../exports/*.onnx`, `../exports/constants_512.json`
and `../exports/browser_reference/` (`tools/export_onnx.py`, then
`PYTHONPATH=src python tools/dump_browser_reference.py`); they are skipped when those are
missing. They use `onnxruntime-web` itself (the version the site ships): its WebAssembly
backend runs in Node 22, so the tests exercise the same kernels the browser's wasm fallback
uses and no second ORT build is needed. Node runs with ORT's default thread count; set
`SAM2_THREADS=1` to match the browser's single thread.

1. Image parity: `image_truck`, three prompts (one point, positive + negative, box) through
   `normalize` + `ImagePredictor`; all four mask tokens within 1e-2 and IoU >= 0.999,
   `iou_predictions` within 1e-3.
2. Video parity: `clip_blackswan`, click on frame 0, forward propagation; per-frame IoU
   >= 0.99 at threshold 0 and object score within 0.1. A second test runs a correction
   click on a tracked frame and a reverse propagation.
3. Unit tests: output selection branches, bfloat16 rounding, bilinear upsampling (hand
   cases and a bit-exact numpy case), `normalize` bit-exactness, constants parsing.

## What the website copies

`src/` becomes the site's `lib/sam2/` verbatim (import from `lib/sam2` or the individual
modules). The site provides `onnxruntime-web@1.22.0`, the graph files and
`constants_512.json` as static assets, and the frame decoding (any `CanvasImageSource`
through `resizeToSquare`).
