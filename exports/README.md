# ONNX exports (SAM 2.1 Hiera-T)

The `.onnx` and `constants_*.json` files are build outputs (gitignored). Regenerate and
check them with:

```sh
python tools/export_onnx.py --model tiny      # writes every file below (about 20 s)
python tools/ort_parity.py --model tiny       # parity + latency tables (image + video, ~15 min)
python tools/ort_parity.py --parts video --sizes 512   # video rows at one size only
python tools/ort_video_loop.py --size 512 --precision shipped   # the ORT-only video loop
```

All graphs: opset 17, legacy TorchScript exporter (`dynamo=False`), simplified with
onnxslim, batch 1, fp32 inputs and outputs. Metadata props `graph`, `model`, and
`image_size` (not on the two size-independent pointer graphs).

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
| `memory_encoder_1024.onnx` | memory encoder, S = 1024 | fp32 | 6.6 MB | fallback |
| `memory_encoder_1024.fp16.onnx` | same | fp16 | 3.3 MB | yes |
| `memory_encoder_512.onnx` | memory encoder, S = 512 | fp32 | 5.9 MB | fallback |
| `memory_encoder_512.fp16.onnx` | same | fp16 | 2.9 MB | yes |
| `memory_attention_1024.onnx` | memory attention, S = 1024 | fp32 | 24.0 MB | fallback |
| `memory_attention_1024.fp16.onnx` | same | fp16 | 12.1 MB | yes |
| `memory_attention_512.onnx` | memory attention, S = 512 | fp32 | 23.9 MB | fallback |
| `memory_attention_512.fp16.onnx` | same | fp16 | 12.0 MB | yes |
| `object_pointer.onnx` | pointer projection + occlusion switch, any S | fp32 | 0.8 MB | yes |
| `object_pointer_tpos.onnx` | pointer temporal encoding, any S | fp32 | 0.07 MB | yes |
| `constants_1024.json`, `constants_512.json` | loop constants (see below) | JSON | 23 kB | yes |

The fp32 encoders are over GitHub's 100 MB limit and are only the parity reference. The
1024 and 512 encoders share every weight; only the static shapes differ (the Hiera
position embedding is resized to the token grid inside the graph and folded by
onnxruntime at session creation). Shipped image set per resolution: 32.5 + 9.0 = 41.5 MB
(512) or 32.5 + 10.6 = 43.1 MB (1024). The video mode adds the fp16 memory graphs and the
pointer graphs: 2.9 + 12.0 + 0.9 = 15.8 MB at 512 (57.3 MB in total) or
3.3 + 12.1 + 0.9 = 16.3 MB at 1024 (59.4 MB in total). Every file is under 100 MB except
the two fp32 reference encoders.

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

## Video graphs

Exported by the same command (`--graphs memory` for these only). All fp32 I/O, batch 1,
one object per run. `fp16` variants: fp16 weights and compute, fp32 inputs and outputs.

### `memory_encoder_{S}`

| Name | Dir | dtype | Shape | Meaning |
|------|-----|-------|-------|---------|
| `vision_feat` | in | float32 | `[1, 256, S/16, S/16]` | raw stride-16 level: encoder `image_embed - no_mem_embed` |
| `mask_logits` | in | float32 | `[1, 1, S, S]` | high-res logits of the chosen mask (bilinear upsample of the S/4 logits) |
| `object_score_logits` | in | float32 | `[1, 1]` | the head's score for this frame |
| `binarize` | in | float32 | `[1]` | 1 for a clicked (conditioning) frame, 0 for a tracked frame |
| `maskmem_features` | out | float32 | `[1, 64, S/16, S/16]` | memory features (store rounded to bfloat16, see below) |
| `maskmem_pos_enc` | out | float32 | `[1, 64, S/16, S/16]` | sine position encoding; a constant of S, cache it from the first call |

Inside the graph: `m = binarize ? (mask_logits > 0) : sigmoid(mask_logits)`, then
`20 * m - 10`, the memory encoder, and `+ no_obj_embed_spatial` when
`object_score_logits <= 0`.

### `memory_attention_{S}`

| Name | Dir | dtype | Shape | Meaning |
|------|-----|-------|-------|---------|
| `curr` | in | float32 | `[1, HW, 256]` | raw stride-16 tokens, `HW = (S/16)^2`, row-major over the grid (transpose of `vision_feat`) |
| `memory` | in | float32 | `[1, N, 64]` | spatial memories (whole frames of HW tokens) then pointer tokens; N dynamic |
| `memory_pos` | in | float32 | `[1, N, 64]` | their position encodings |
| `num_obj_ptr_tokens` | in | int64 | `[1]` | pointer tokens at the end of `memory` (4 per pointer); excluded from RoPE |
| `fused` | out | float32 | `[1, HW, 256]` | memory-conditioned tokens |
| `image_embed` | out | float32 | `[1, 256, S/16, S/16]` | the same values as a map: the SAM head's `image_embed` for a tracked frame |

`image_embed` is `fused` **transposed** (`[HW, 256] -> [256, HW]`), not reshaped; feed it
to the head directly and ignore `fused`. The stride-16 position encoding (`curr_pos`) is a
constant of S and is built inside the graph, as are the RoPE tables (real arithmetic:
each channel pair `(x0, x1)` becomes `(x0 cos - x1 sin, x0 sin + x1 cos)`; the keys before
the pointer tokens are rotated frame by frame with the query grid's table, so any number
of memory frames works). The attention is split into static query chunks so that one
score matrix stays under 64 MiB for a full bank (7 frames + 16 pointers): 8 chunks of 512
queries at 1024, one chunk at 512. Every extra clicked frame adds HW keys (all
conditioning frames are attended, `max_cond_frames_in_attn = -1`).

### `object_pointer` and `object_pointer_tpos`

Identical at every S, so one file each (not `_{S}`).

| Graph | Name | Dir | dtype | Shape | Meaning |
|-------|------|-----|-------|-------|---------|
| `object_pointer` | `mask_token` | in | float32 | `[1, 256]` | `sam_tokens[i]` of the chosen output (selection rule 3 above) |
| | `object_score_logits` | in | float32 | `[1, 1]` | the head's score |
| | `obj_ptr` | out | float32 | `[1, 256]` | `obj_ptr_proj(token)` if score > 0, else the learned `no_obj_ptr` |
| `object_pointer_tpos` | `t_diff` | in | float32 | `[K]` | temporal distance of each pointer (integers as float, K dynamic) |
| | `t_diff_max` | in | float32 | `[1]` | `min(num_frames, 16) - 1` |
| | `ptr_pos` | out | float32 | `[K, 64]` | `obj_ptr_tpos_proj(sine_256(t_diff / t_diff_max))` |

### `constants_{S}.json`

Everything the loop needs outside the graphs; the demo never reads the checkpoint. Floats
are written exactly (float32 values printed as float64), so `Float32Array.from` restores
them bit for bit.

| Key | Value | Use |
|-----|-------|-----|
| `pixel_mean`, `pixel_std` | `(0.485, 0.456, 0.406)`, `(0.229, 0.224, 0.225)` | frame normalisation |
| `no_mem_embed` | `[256]` | `vision_feat = image_embed - no_mem_embed` (per channel) |
| `maskmem_tpos_enc` | `[7][64]` | temporal slot encodings of spatial memories |
| `no_obj_ptr`, `no_obj_embed_spatial` | `[256]`, `[64]` | baked into the graphs; listed for auditing |
| `no_obj_score` | -1024 | mask logits of a frame where the object is absent |
| `stability_delta`, `stability_thresh` | 0.05, 0.98 | single-mask stability rule |
| `multimask_min_pt_num`, `multimask_max_pt_num` | 0, 1 | multimask when the prompt has 0 or 1 points |
| `mask_input_clamp` | 32 | clamp of a previous mask fed back as `mask_input` |
| `num_maskmem`, `max_obj_ptrs_in_encoder`, `tokens_per_obj_ptr` | 7, 16, 4 | memory bank sizes |
| `memory_temporal_stride_for_eval`, `max_cond_frames_in_attn`, `only_obj_ptrs_in_the_past_for_eval` | 1, -1, true | bank selection (released settings) |
| `sigmoid_scale_for_mem_enc`, `sigmoid_bias_for_mem_enc` | 20, -10 | baked into the memory encoder |
| `image_size`, `embed_grid`, `mask_size` | S, S/16, S/4 | shapes |

## Video loop recipe

`tools/ort_video_loop.py` (`OrtVideoTracker`) runs this with `numpy` + `onnxruntime`
only; the TypeScript port follows it step by step. Notation: `g = S/16`, `HW = g*g`,
`m = S/4`; `d = +1` forward, `-1` reverse; frames are 0-based.

**Per video**

1. Load `constants_{S}.json` and the six sessions.
2. Each frame: RGB, resize to S x S ignoring aspect ratio (PIL bicubic in Python; see
   deviations), `x / 255`, `(x - pixel_mean) / pixel_std`, CHW float32.

**Per frame (once, cached)**

3. `image_encoder` -> `image_embed`, `high_res_feat_s0`, `high_res_feat_s1`.
   `vision_feat = image_embed - no_mem_embed[c]`; `curr[t, c] = vision_feat[c, t]`
   (`[1, HW, 256]`).

**Click on frame `f`** (points in video pixels -> `x * S / W`, `y * S / H`, labels 1/0)

4. `mask_input` = the frame's previous low-res logits clamped to `[-32, 32]` if it already
   has a prediction (`has_mask_input = 1`), else zeros (`has_mask_input = 0`).
5. Embedding: if `f` has never been tracked, the encoder's `image_embed` (no memory).
   Otherwise steps 9-10 in the direction it was last tracked (a correction).
6. `sam_head` with the points (padding is added in the graph). Selection:
   multimask when the point count (a box counts 2) is 0 or 1: `best = 1 + argmax(iou[1..3])`
   (first maximum), mask `masks[best]`, token `sam_tokens[best]`; otherwise token 0's mask
   if `count(logit > 0.05) / count(logit > -0.05) >= 0.98` (float32; 1.0 when the
   denominator is 0), else `masks[best]`, and the token is always `sam_tokens[0]`. If
   `object_score_logits <= 0` the whole mask becomes -1024.
7. `object_pointer(token, score)` -> `obj_ptr`. Keep `{pred_masks [1,1,m,m], obj_ptr,
   score}` as *pending* (no memory yet; further clicks redo 4-7).

**Start of propagation**

8. Each pending frame (ascending): `high_res = upsample(pred_masks, S)` (bilinear,
   half-pixel centres, edge clamp: source coordinate `(i + 0.5) / 4 - 0.5`, clamped at 0);
   `memory_encoder(vision_feat, high_res, score, binarize = 1)`; store
   `bf16(maskmem_features)` (round to nearest even on the top 16 bits:
   `u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000`); cache `maskmem_pos_enc` once as
   tokens `pos[t, c]` (`[HW, 64]`). The frame becomes a *conditioning* frame (removed from
   the tracked set if it was there).

**Track frame `t`** (visit `start, start + d, ...` to the clip end; a conditioning frame
is yielded with its stored prediction and not re-tracked)

9. Memory bank, in this order:
   * every conditioning frame (in the order they became conditioning): tokens
     `feat[t, c]` (`[HW, 64]`, transpose of the stored features), pos
     `pos + maskmem_tpos_enc[6]`;
   * for `t_pos = 1 .. 6`: frame `p = t - d * (7 - t_pos)`; if `p` is a *tracked* frame
     with memory (conditioning frames never fill these slots): tokens of `p`, pos
     `pos + maskmem_tpos_enc[6 - t_pos]` (`t_pos = 6` is the previous frame, row 0);
   * pointers, with `P = min(num_frames, 16)`: every conditioning frame `c` with
     `c <= t` (forward) / `c >= t` (reverse) at distance `(t - c) * d`, then for
     `k = 1 .. P - 1` the tracked frame `t - d * k` at distance `k` (stop at the first
     index outside the clip; skip frames that are not tracked). `ptr_pos =
     object_pointer_tpos(distances, [P - 1])`; pointer `j` contributes 4 tokens
     `obj_ptr[64 i : 64 (i + 1)]`, `i = 0..3`, each with position `ptr_pos[j]`.
   * `memory = concat(...)` `[1, N, 64]`, `memory_pos` likewise,
     `num_obj_ptr_tokens = 4 * #pointers`.
10. `memory_attention(curr, memory, memory_pos, num_obj_ptr_tokens)` -> `image_embed`.
11. `sam_head(image_embed, s0, s1, point (0, 0) label -1, zeros, has_mask_input = 0)`;
    selection as step 6 with 0 points (multimask); -1024 mask when the score is <= 0;
    `object_pointer`.
12. `memory_encoder(vision_feat, upsample(pred_masks, S), score, binarize = 0)`, store
    `bf16(features)`; frame `t` is now tracked (with its direction).
13. Display: bilinear upsample of `pred_masks` to the video size, threshold 0; the object
    is occluded when `score <= 0`.

**Deviations from `video_predictor.py`**: one object (the demo's scope; the torch side
batches objects independently anyway); no mask prompt (`add_mask`) and no box in
`add_click` (the head accepts a box as labels 2 and 3, so adding it is mechanical);
`clear_old_points` is always true. Everything else, including corrections on tracked
frames and the general frame-selection rules (`max_cond_frames_in_attn`, temporal
stride), mirrors the torch code. In the browser, canvas resizing is not PIL bicubic;
that changes the encoder input slightly and is outside this parity check.

## Parity: image half (tiny, ORT 1.31 CPU provider vs torch fp32)

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

## Parity: video half (tiny, ORT 1.31 CPU provider)

Bedroom clip (48 frames, 960 x 540), one positive click at (210, 350): on frame 0
propagated forward (48 frames) and on frame 30 propagated in reverse (31 frames), the
prompts of `tools/dump_reference.py`. Reference: the official package's dump at 1024;
at 512, where no dump exists, this repo's torch `VideoPredictor` built with
`dataclasses.replace(TINY, image_size=512)` (it matches the official package exactly at
1024). Graph rows replay the memory calls of frames 1, 2, 7, 20 and 47 (the memory
encoder rows add three synthetic calls for the branches the clip never takes: a clicked
mask and an absent object). The object is visible on every frame of this clip (lowest
object score 5.9), so no occlusion is exercised end to end. `max|err|` for the memory
encoder is before the bfloat16 storage rounding; memory-attention IoU is the tracking
head call (fp32 head) on the graph's embedding vs on the reference embedding; loop IoU is
per frame on the S/4 logits at 0. Graph `ms` is one call with the frame-47 bank (7
spatial frames + 16 pointers); loop `ms` is the median per tracked frame including the
encoder.

| Graph / loop | S | Precision | MB | max\|err\| | IoU min (frame) | IoU mean | ORT CPU ms | Criterion |
|--------------|--:|-----------|---:|-----------:|----------------:|---------:|-----------:|-----------|
| `memory_encoder_1024` | 1024 | fp32 | 6.6 | 3.8e-05 | | | 40 | < 1e-3: pass |
| `memory_encoder_1024` | 1024 | fp16 | 3.3 | 2.3e-03 | | | 50 | reported |
| `memory_attention_1024` | 1024 | fp32 | 24.0 | 7.7e-05 | 1.0000 | 1.0000 | 1005 | < 1e-3, >= 0.999: pass |
| `memory_attention_1024` | 1024 | fp16 | 12.1 | 5.5e-03 | 1.0000 | 1.0000 | 4375 | >= 0.99: pass |
| loop forward | 1024 | fp32 | 161.7 | | 1.0000 | 1.0000 | 2125 | >= 0.999: pass |
| loop reverse | 1024 | fp32 | 161.7 | | 1.0000 | 1.0000 | 1881 | >= 0.999: pass |
| loop forward | 1024 | shipped | 59.3 | | 0.9620 (44) | 0.9959 | 4029 | >= 0.98: **fail** |
| loop reverse | 1024 | shipped | 59.3 | | 0.7916 (7) | 0.9756 | 3891 | >= 0.98: **fail** |
| `memory_encoder_512` | 512 | fp32 | 5.9 | 5.4e-05 | | | 8 | < 1e-3: pass |
| `memory_encoder_512` | 512 | fp16 | 2.9 | 2.2e-03 | | | 11 | reported |
| `memory_attention_512` | 512 | fp32 | 23.9 | 1.2e-05 | 1.0000 | 1.0000 | 69 | < 1e-3, >= 0.999: pass |
| `memory_attention_512` | 512 | fp16 | 12.0 | 5.7e-03 | 1.0000 | 1.0000 | 140 | >= 0.99: pass |
| loop forward | 512 | fp32 | 157.6 | | 1.0000 | 1.0000 | 220 | >= 0.999: pass |
| loop reverse | 512 | fp32 | 157.6 | | 1.0000 | 1.0000 | 222 | >= 0.999: pass |
| loop forward | 512 | shipped | 57.3 | | 0.9701 (43) | 0.9903 | 308 | >= 0.97: pass |
| loop reverse | 512 | shipped | 57.3 | | 0.9886 (22) | 0.9968 | 326 | >= 0.97: pass |
| `object_pointer` | any | fp32 | 0.8 | 7.2e-07 | | | < 0.1 | < 1e-3: pass |
| `object_pointer_tpos` | any | fp32 | 0.07 | 1.4e-06 | | | < 0.1 | < 1e-3: pass |

Shipped set: int8 encoder, fp16 head, fp16 memory encoder, fp16 memory attention, fp32
pointer graphs (MB column: the files the loop loads). The fp32 loop's largest per-frame
errors are 0.010 (1024) / 0.021 (512) in low-res logits, 5e-4 / 8e-4 in the object score
and 1e-3 / 2e-3 in the object pointer.

Per-graph CPU time inside the loop (median over the clip, ms; Apple M5, default threads):

| S | Set | encoder | memory attention | SAM head | memory encoder | pointer graphs | tracked frame |
|--:|-----|--------:|-----------------:|---------:|---------------:|---------------:|--------------:|
| 1024 | fp32 | 830 | 1222 | 21 | 42 | 0.2 | 2125 |
| 1024 | shipped | 687 | 3266 | 26 | 46 | 0.2 | 4029 |
| 512 | fp32 | 132 | 70 | 4.7 | 10 | 0.1 | 220 |
| 512 | shipped | 134 | 151 | 8.3 | 12 | 0.1 | 308 |

fp16 is slower than fp32 on the CPU provider (as for the head); the fp16 files exist for
download size and for WebGPU with `shader-f16`. The fp32 memory graphs are the fallback,
like the fp32 head.

### Precision study (loop IoU min, forward / reverse)

| Image encoder | SAM head | Memory graphs | 512 | 1024 |
|---------------|----------|---------------|----:|-----:|
| fp32 | fp32 | fp32 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| fp32 | fp32 | fp32, memory kept fp32 (no bfloat16 rounding) | 1.0000 / 0.9978 | 0.9952 / 0.9965 |
| fp32 | fp32 | fp16 encoder, fp32 attention | 1.0000 / 1.0000 | |
| fp32 | fp32 | fp32 encoder, fp16 attention | 1.0000 / 1.0000 | |
| fp32 | fp32 | fp16 | 0.9853 / 1.0000 | 1.0000 / 0.9981 |
| fp32 | fp16 | fp32 | 0.9589 / 0.9824 | 0.9955 / 0.8020 |
| int8 | fp32 | fp32 | 0.9701 / 0.9887 | 0.9620 / 0.8012 |
| int8 | fp16 | fp32 | 0.9701 / 0.9886 | 0.9578 / 0.7916 |
| int8 | fp16 | fp16 encoder, fp32 attention | 0.9701 / 0.9886 | |
| **int8** | **fp16** | **fp16 (shipped)** | **0.9701 / 0.9886** | **0.9620 / 0.7916** |
| fp16 (54.6 MB, not exported) | fp16 | fp16 | 0.9831 / 0.9538 | 0.9955 / 0.9660 |

* The memory graphs are not the problem: in fp16 alone they keep min IoU >= 0.985 at
  512 and >= 0.998 at 1024, so they ship in fp16.
* The bfloat16 rounding of stored memory features is part of the reference behaviour:
  skipping it moves masks (0.9952 at 1024), so the loop keeps it (a two-line integer
  operation, step 8).
* The loss comes from the image side. At 512 the tracked face covers about 70 pixels of
  the 128 x 128 mask grid, so one boundary pixel is about 1.5 % IoU; the weak frames are
  spread over the clip (frame 1 is already 0.971 with the int8 encoder) rather than
  drifting.
* The 1024 reverse failure is a multimask near-tie, not a numerical fault. On reverse
  frames 6 to 16 the reference's predicted IoUs for tokens 1 and 3 are within 0.002 to
  0.01 of each other (frame 7: 0.7916 vs 0.7898) and the two candidates overlap at IoU
  0.80. Any image-side perturbation (fp16 head alone, int8 encoder alone, an fp16
  encoder) picks the other candidate on some of those frames; the run recovers within a
  few frames (mean 0.976). The fp32 head does not help with the int8 encoder, and no
  shippable encoder precision passes 0.98 there. The demo's video default is 512 (D5),
  where the shipped set passes.

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
* Video graph ops: memory encoder Add, Cast, Conv, Div, Erf, Greater, MatMul, Mul, Pow,
  ReduceMean, Reshape, Sigmoid, Sqrt, Sub, Transpose, Where; memory attention Add,
  Concat, Expand, Gather, LayerNormalization, MatMul, Mul, Relu, Reshape, Shape, Slice,
  Softmax, Sub, Transpose, Unsqueeze; pointer graphs Add, Concat, Cos, Div, Gemm,
  Greater, Mul, Relu, Sin, Sub, Unsqueeze, Where (fp16 files add Cast and Identity). None
  is in the avoid list (checked by the export test). The memory attention's only dynamic
  dimension is the bank length N (Shape / Slice / Reshape on it); its RoPE and position
  grids are Expand / Concat of small constant tables, folded at session creation.
* Memory attention buffers: with a full bank the fp32 score matrix per query chunk is
  512 x 28 736 (59 MB, 8 chunks) at 1024 and 1024 x 7 232 (30 MB, one chunk) at 512; the
  self attention at 1024 is 4096 x 4096 (64 MiB). Each extra clicked frame adds HW keys.
  The bank itself is rebuilt by the caller every frame: up to 7.4 MB at 1024 and 1.9 MB
  at 512.
* fp16 memory graphs, like the fp16 heads, need `shader-f16`; otherwise load the fp32
  files.

