# ONNX exports (SAM 2.1 Hiera-T, image half)

The `.onnx` files are build outputs (gitignored). Regenerate and check them with:

```sh
python tools/export_onnx.py --model tiny      # writes the eight files below (about 30 s)
python tools/ort_parity.py --model tiny       # parity + latency table vs the torch modules
```

All graphs: opset 17, legacy TorchScript exporter (`dynamo=False`), simplified with
onnxslim, batch 1, fp32 inputs and outputs. Metadata props `graph`, `model`, `image_size`.

| File | Graph | Weights | Size | Shipped |
|------|-------|---------|-----:|:-------:|
| `image_encoder_1024.onnx` | encoder, S = 1024 | fp32 | 109.1 MB | no (reference) |
| `image_encoder_1024.int8.onnx` | encoder, S = 1024 | int8 weight-only | 32.5 MB | yes |
| `image_encoder_512.onnx` | encoder, S = 512 | fp32 | 109.1 MB | no (reference) |
| `image_encoder_512.int8.onnx` | encoder, S = 512 | int8 weight-only | 32.5 MB | yes |
| `sam_head_1024.onnx` | prompt encoder + mask decoder, S = 1024 | fp32 | 21.1 MB | fallback |
| `sam_head_1024.fp16.onnx` | same | fp16 | 10.6 MB | yes |
| `sam_head_512.onnx` | prompt encoder + mask decoder, S = 512 | fp32 | 17.9 MB | fallback |
| `sam_head_512.fp16.onnx` | same | fp16 | 9.0 MB | yes |

The fp32 encoders are over GitHub's 100 MB limit and are only the parity reference. The
1024 and 512 encoders share every weight; only the static shapes differ (the Hiera
position embedding is resized to the token grid inside the graph and folded by
onnxruntime at session creation). Shipped set per resolution: 32.5 + 9.0 = 41.5 MB (512)
or 32.5 + 10.6 = 43.1 MB (1024).

## `image_encoder_{S}`

| Name | Dir | dtype | Shape |
|------|-----|-------|-------|
| `image` | in | float32 | `[1, 3, S, S]` |
| `image_embed` | out | float32 | `[1, 256, S/16, S/16]` |
| `high_res_feat_s0` | out | float32 | `[1, 32, S/4, S/4]` |
| `high_res_feat_s1` | out | float32 | `[1, 64, S/8, S/8]` |

Preprocessing (as `ImagePredictor.preprocess`): RGB, resize to S x S ignoring aspect
ratio (bilinear, antialiased), divide by 255, subtract mean `(0.485, 0.456, 0.406)` and
divide by std `(0.229, 0.224, 0.225)`, CHW.

`image_embed` is the stride-16 FPN level **with `no_mem_embed` added**, so it feeds the
head directly for single images and for prompted (conditioning) video frames without
memory. The memory path needs the raw level: `raw = image_embed - no_mem_embed`
(checkpoint key `no_mem_embed`, shape `(1, 1, 256)`, broadcast over the grid).
`high_res_feat_s0` / `_s1` are the stride-4 / stride-8 levels after the mask decoder's
`conv_s0` / `conv_s1`.

## `sam_head_{S}`

| Name | Dir | dtype | Shape | Meaning |
|------|-----|-------|-------|---------|
| `image_embed` | in | float32 | `[1, 256, S/16, S/16]` | encoder output |
| `high_res_feat_s0` | in | float32 | `[1, 32, S/4, S/4]` | encoder output |
| `high_res_feat_s1` | in | float32 | `[1, 64, S/8, S/8]` | encoder output |
| `point_coords` | in | float32 | `[1, P, 2]` | (x, y) in S-pixel space, P dynamic |
| `point_labels` | in | int32 | `[1, P]` | -1 pad, 0 negative, 1 positive, 2 box top-left, 3 box bottom-right |
| `mask_input` | in | float32 | `[1, 1, S/4, S/4]` | previous low-res logits |
| `has_mask_input` | in | float32 | `[1]` | 1 uses `mask_input`, 0 uses the learned no-mask embedding |
| `masks` | out | float32 | `[1, 4, S/4, S/4]` | logits of all four mask tokens |
| `iou_predictions` | out | float32 | `[1, 4]` | predicted IoU (already sigmoid) |
| `object_score_logits` | out | float32 | `[1, 1]` | object present if > 0 |
| `sam_tokens` | out | float32 | `[1, 4, 256]` | mask-token outputs (object pointers) |

The low-res mask side is S/4: 256 at 1024 and 128 at 512.

Prompt conventions (mirroring `ImagePredictor.predict`):

* Coordinates: `x * S / W`, `y * S / H` from original pixels. Do not add 0.5; the graph
  does it.
* A box is two points, `(x0, y0)` label 2 and `(x1, y1)` label 3, placed **before** any
  clicks.
* The graph always appends one padding point (label -1) after the given points, as the
  prompt encoder does when called with `boxes=None`. Pass exactly the points the Python
  side passes; never add your own trailing pad.
* No prompt at all (video frames decoded from memory only): pass one point `(0, 0)` with
  label -1.
* With `has_mask_input = 0`, `mask_input` is ignored but must be finite (zeros).
  Feed back previous logits clamped to `[-32, 32]`.

Output selection is left to the caller (token 0 = single mask, tokens 1..3 = multimask):

1. **Multimask** when the prompt has 0 or 1 points (a single click, or a memory-only
   tracked frame in the video loop; a box counts as 2 points): candidates are tokens 1..3
   with scores `iou_predictions[1..3]`, the best is the argmax. The image demo shows all
   three.
2. **Single mask** otherwise (several points or a box): take token 0 if it is
   stable, else fall back to the best multimask candidate. Stability of token 0 is
   `count(logit > 0.05) / count(logit > -0.05)` over its S/4 x S/4 logits (1.0 when the
   denominator is 0); stable means >= 0.98. The fallback replaces both the mask and its
   IoU score.
3. **Object pointer token**: multimask calls use `sam_tokens[best]` (same index as the
   chosen mask); single-mask calls always use `sam_tokens[0]`, even when the stability
   fallback picked a multimask mask.
4. Final mask: bilinear upsample of the chosen logits (align corners false) to the
   original size, threshold at 0. During tracking, a frame with
   `object_score_logits <= 0` counts as "object absent".

## Parity (tiny, ORT 1.31 CPU provider vs torch fp32)

Three images x their manifest prompts, each decoded with and without a mask prompt (18
cases per row). `max|err|` is over every output of that graph (head rows get torch's
encoder features). IoU is the best multimask candidate at threshold 0 on the S/4 logits;
encoder rows are end to end through the fp32 head. `flips` counts best-candidate choices
that differ from torch. Latency is the median single `run` on an Apple M5 laptop, default
threads; it varies by about 20 % between runs.

| Graph | Precision | max\|err\| | IoU min | IoU mean | flips | ORT CPU ms | Criterion |
|-------|-----------|-----------:|--------:|---------:|------:|-----------:|-----------|
| `image_encoder_1024` | fp32 | 2.1e-05 | 0.9999 | 1.0000 | 0/18 | 797 | < 1e-3, >= 0.999 |
| `image_encoder_1024` | int8 | 6.0e-02 | 0.9895 | 0.9979 | 0/18 | 845 | >= 0.98 |
| `sam_head_1024` | fp32 | 1.8e-04 | 0.9999 | 1.0000 | 0/18 | 20.9 | < 1e-3, >= 0.999 |
| `sam_head_1024` | fp16 | 8.4e-02 | 0.9984 | 0.9997 | 0/18 | 30.1 | >= 0.99 |
| int8 encoder + fp16 head | 1024 | | 0.9913 | 0.9980 | 0/18 | | >= 0.98 |
| `image_encoder_512` | fp32 | 9.5e-06 | 1.0000 | 1.0000 | 0/18 | 155 | < 1e-3, >= 0.999 |
| `image_encoder_512` | int8 | 4.4e-02 | 0.9897 | 0.9976 | 0/18 | 187 | >= 0.98 |
| `sam_head_512` | fp32 | 1.6e-04 | 1.0000 | 1.0000 | 0/18 | 5.7 | < 1e-3, >= 0.999 |
| `sam_head_512` | fp16 | 8.3e-02 | 0.9908 | 0.9988 | 0/18 | 8.9 | >= 0.99 |
| int8 encoder + fp16 head | 512 | | 0.9897 | 0.9975 | 0/18 | | >= 0.98 |

The int8 and fp16 files save download size, not CPU time. With default session options
ORT dequantizes the int8 weights on every run (the int8 rows above); with
`session.disable_quant_qdq = "1"` it folds them once at load and the 512 int8 encoder
runs in about 110 ms, bit-identical outputs. The fp16 head is slower than fp32 on CPU.

## int8 encoder: what was tried

Same harness, end-to-end mask IoU (min / mean over the 18 cases). The weakest case is
always the groceries image at 512, which also moves most under fp16.

| Scheme | fp32 kept | Size | IoU 512 | IoU 1024 | WebGPU |
|--------|-----------|-----:|--------:|---------:|:------:|
| `quantize_dynamic`, per-tensor | Conv | 28.9 MB | 0.944 / 0.976 | 0.929 / 0.981 | no |
| `quantize_dynamic`, per-channel | Conv | 29.1 MB | 0.946 / 0.986 | | no |
| `quantize_dynamic`, per-channel | Conv, blocks 0-1 | 30.6 MB | 0.977 / 0.992 | 0.988 / 0.997 | no |
| `quantize_dynamic`, per-channel (`--int8-scheme dynamic`) | Conv, blocks 0-3 | 36.8 MB | 0.982 / 0.996 | 0.995 / 0.998 | no |
| weight-only, per-channel | none | 27.9 MB | 0.918 / 0.983 | 0.979 / 0.994 | yes |
| weight-only, 64-row blocks | Conv | 30.9 MB | 0.985 / 0.996 | 0.987 / 0.996 | yes |
| **weight-only, 64-row blocks (default)** | **Conv, blocks 0-1** | **32.5 MB** | **0.990 / 0.998** | **0.990 / 0.998** | **yes** |

A per-block sweep showed weight rounding, not activation quantization, dominates the
error, concentrated in Hiera blocks 0-3 (the stride-4/8 stages that feed the mask
decoder's skip connections) and in the MLP layers; quantizing the Convs (patch embed, neck
laterals, `conv_s*`) also costs IoU for only 1.3 MB. `quantize_dynamic` emits `MatMulInteger` and
`DynamicQuantizeLinear`, which ORT Web's WebGPU provider does not implement (each would
fall back to WASM with a GPU/CPU copy), and onnxruntime implements `ConvInteger` only for
uint8 weights, hence Conv excluded. The default instead stores int8 weights with one
fp32 scale per output column per 64 input rows, rebuilt in the graph by
`DequantizeLinear` (opset 13 per-axis) plus a constant Reshape/Transpose, with fp32
compute.

## ORT Web notes

* Encoder ops: Add, Conv, DequantizeLinear (int8 only), Div, Erf, LayerNormalization,
  MatMul, MaxPool, Mul, Pad, Reshape, Resize (cubic on the position table, nearest in the
  FPN), Slice, Softmax, Split, Squeeze, Tile, Transpose. Every shape is static; no Shape
  ops remain.
* Head ops: the encoder set minus DequantizeLinear/MaxPool/Pad/Resize/Split/Squeeze/Tile,
  plus Cast, Concat, ConvTranspose, Cos, Equal, Gather, Gemm, Pow, ReduceMean, Relu,
  Sigmoid, Sin, Sqrt, Sub, Unsqueeze, Where (Identity in fp16). The only dynamic dimension
  is P.
* None of these is in the avoid list checked by the export and the tests (MatMulInteger,
  ConvInteger, DynamicQuantizeLinear, NonZero, Unique, Loop, If, Scan,
  NonMaxSuppression). Each has a kernel in ORT Web's WebGPU operator table, so the graphs
  are expected to run without CPU fallback; confirm in the browser at T7/T8 (verbose
  logging lists any node assigned to the CPU provider). The cubic Resize and the Tile only
  touch constants and are folded at session creation.
* fp16 heads need a WebGPU adapter with `shader-f16`. On the WASM backend (or without
  `shader-f16`) load the fp32 head; it is faster on CPU anyway.
* ORT keeps `DequantizeLinear` on constant weights unfolded by default, so the int8 encoder
  dequantizes about 27 M weights every run. Setting the session config entry
  `session.disable_quant_qdq = "1"` folds them once at load (fp32 weights resident, same
  speed as the fp32 graph, identical outputs); in ORT Web pass it through
  `extra.session.disable_quant_qdq`.
