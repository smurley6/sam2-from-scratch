# SAM 2 from scratch: program document

Started 2026-10-21. Owner: Steven Murley.

## Purpose (the only requirement)

Two deliverables on stevenmurley.com, nothing else:

1. **A case study** (`/projects/sam2`) describing a from-scratch reimplementation of the
   inference architecture of *SAM 2: Segment Anything in Images and Videos* (Ravi et al.,
   arXiv:2408.00714, v2 2024-10-28), verified numerically against Meta's released weights.
2. **A working demo** on that page: promptable image segmentation and short-video object
   tracking running entirely in the visitor's browser, from ONNX graphs exported from *this*
   reimplementation (not Meta's package).

Everything in this repo exists to feed those two pages. Scope that does not land on the site
is cut.

## Locked decisions

| # | Decision | Why |
|---|----------|-----|
| D1 | Reimplement the **inference** path of SAM 2.1 (image encoder, prompt encoder, mask decoder, memory encoder, memory attention, video propagation loop) in plain PyTorch from the paper. Training on SA-V is **not** reproduced (50.9k videos, 256 A100s); the case study says so in its first paragraph. | Honest scope that fits one laptop. Reimplementing and verifying the architecture is the credible, inspectable claim. |
| D2 | Correctness is defined as **numerical parity** with the official `sam2` package loading the same `sam2.1_hiera_tiny.pt` / `sam2.1_hiera_small.pt` weights: per-module max abs error on identical inputs, mask IoU ≥ 0.99 end to end on images, and identical propagated masks (IoU ≥ 0.98 per frame) on videos. The reference package is a **test-only** dependency; `src/sam2_scratch` imports nothing from it. | Parity is the only ground truth available without retraining. |
| D3 | One external benchmark number: **DAVIS 2017 val J&F** (semi-supervised, first-frame mask prompt) with the official evaluator, for Hiera-T and Hiera-S, reported next to Meta's numbers. Cached frames at 480p. | Gives the case study a recognisable headline that a reader can check. |
| D4 | Demo stack = **ONNX Runtime Web**, WebGPU execution provider with WASM fallback, same loader pattern as the FundusVision demo already on the site. Four exported graphs: `image_encoder`, `sam_head` (prompt encoder + mask decoder), `memory_encoder`, `memory_attention`. The video loop (memory bank selection, object pointers, occlusion handling) is written in TypeScript, mirroring the Python loop. | The site is static on Vercel; there is no backend and there will not be one. |
| D5 | Shipped model = **Hiera-T** (38.9 M params). Encoder exported at the paper's 1024 px and at 512 px. **Revised 2026-10-22:** the site ships the 512 px set only, for both image and video mode (77 MB total: int8 weight-only encoder 32.5 MB, fp32 head 17.9 MB, fp32 memory graphs 29.7 MB, pointer graphs 0.9 MB), because the shipped-precision loop at 1024 fell below the parity floor on an ambiguous clip and fp32 heads run on WASM without shader-f16. The 1024 graphs and the fp16 variants stay in the repo. | Download budget, GitHub file limits, WebGPU op support. |
| D6 | Demo UX: image mode (upload or sample image, positive/negative clicks and a box, three candidate masks with predicted IoU); video mode (three bundled clips of 24 to 48 frames, click the object on any frame, propagate forward and backward with a per-frame progress bar; occlusion score shown). Nothing leaves the device. | Shows both of the paper's contributions, the promptable head and the streaming memory. |
| D7 | Repo goes **public** at github.com/smurley6/sam2-from-scratch when the case study ships. No AI-assistance references anywhere in commits, code or docs. No em dashes in site copy. | Standing rules for all of Steven's repos. |
| D8 | Website edits happen **last**, in a single pass, after the exports exist. Another session holds uncommitted changes in `~/Desktop/personal-website`; this program does not touch that repo until the demo bundle is final. | Avoid two writers in one checkout. |

## Architecture to reproduce (SAM 2.1, from the paper and the released configs)

- **Image encoder**: Hiera (hierarchical ViT, MAE pretrained) with 4 stages, patch embed 7x7
  stride 4, windowed attention with global attention at a fixed set of blocks, q-pooling at
  stage boundaries, learned absolute position embeddings (tiled window + background).
  Hiera-T: embed 96, 1 head, stages [1, 2, 7, 2], global blocks [5, 7, 9], window sizes
  [8, 4, 14, 7]. Hiera-S: stages [1, 2, 11, 2], global blocks [7, 10, 13].
- **FPN neck**: 1x1 conv per stage to 256 channels, top-down fusion into the two highest
  resolution levels (strides 4 and 8 feed the mask decoder as skip features, stride 16 is the
  main image embedding, 64x64 at 1024 px). Sinusoidal 2-D positional encodings.
- **Prompt encoder**: points (positive, negative, box corners) through random Fourier
  position encoding (128 frequencies) plus learned type embeddings; dense mask prompt
  through a small conv stack; a learned no-mask embedding.
- **Mask decoder**: two-layer two-way transformer (token-to-image and image-to-token
  cross attention, 8 heads, dim 256, MLP 2048), output tokens = IoU token + 4 mask tokens +
  object-score token; upscaling via two transposed convs with the stride-4 and stride-8 skip
  features added; hypernetwork MLPs produce per-mask dot-product kernels; IoU head; object
  score head (occlusion prediction); multimask output for ambiguous single-click prompts.
- **Memory encoder**: downsample the predicted mask (4 strided convs), add to the image
  embedding, two "CX" fuser blocks (depthwise 7x7 conv, LayerNorm, MLP, layer scale), then
  a 1x1 projection to 64 channels.
- **Memory attention**: 4 transformer layers over the current frame tokens: self attention,
  cross attention to the concatenated memory bank (spatial memory features of up to 7 frames
  with temporal position embeddings, plus up to 16 object pointer tokens), MLP. 2-D rotary
  position embeddings in attention (RoPE), with the object-pointer tokens excluded from RoPE.
- **Video loop**: per frame, encode once; if the frame has prompts, run the SAM head on the
  raw image embedding (conditioning frame); otherwise condition the embedding on the memory
  bank, run the SAM head with no prompts, pick the single mask, encode the result into
  memory. Memory bank = all conditioning frames plus the 6 most recent non-conditioning
  frames (num_maskmem = 7), object pointers from the mask-token output of the last 16
  frames; occluded frames write a "no object" memory via a learned embedding.

Exact hyperparameters are read from the released YAML configs at parity-test time and pinned
in `src/sam2_scratch/configs.py`; the list above is the design summary for the case study.

## Milestones and tasks

Each task is one commit series with its own parity test. Order is fixed; nothing moves to
the website before M5.

| Task | Deliverable | Done when |
|------|-------------|-----------|
| T1 | Repo skeleton: package layout, venv, lint/type/test config, weight download + hash script, `tools/dump_reference.py` that saves reference intermediate tensors for a fixed image | `pytest` runs; reference dumps exist for tiny and small |
| T2 | Hiera image encoder + FPN neck + weight loader (official state-dict key mapping) | Max abs error vs reference backbone features < 1e-4 fp32 on 3 images, both sizes |
| T3 | Prompt encoder + mask decoder + image predictor API (`predict(image, points, boxes, mask)`) | Masks, IoU predictions and object scores match reference within 1e-4; multimask IoU ≥ 0.99 on 20 prompts across 5 images |
| T4 | Memory encoder + memory attention + video propagator (`init_state`, `add_prompt`, `propagate`) | Per-frame mask IoU ≥ 0.98 vs reference on 3 clips, forward and backward, including an occlusion |
| T5 | DAVIS 2017 val evaluation with the official J&F evaluator; results table for T and S | Numbers within 1.0 J&F of Meta's reported values (or the gap is explained) |
| T6 | ONNX export of the four graphs at 1024 and 512 px, int8 encoder + fp16 heads, ORT parity harness in Python | ORT outputs match torch within 1e-2 (fp16) / IoU ≥ 0.98 (int8); file sizes within D5 budget |
| T7 | Demo bundle (`bundle/`: three sample clips, three images, credits) and the browser runtime `web/` (framework-free TypeScript port of the onnxruntime loop, tested in Node with onnxruntime-web against `tools/dump_browser_reference.py` outputs) | `cd web && npm test` passes with per-frame IoU 1.0 against the Python loop |
| T8 | Website: `components/Sam2Demo.tsx` (image + video modes), case-study entry in `content/projects.ts`, architecture SVG diagram, figures (parity table, DAVIS table, speed table), `/projects/sam2` page wiring | `npm run typecheck && npm run lint && npm run build` green; demo verified in a browser on a sample clip; em-dash CI rule holds |
| T9 | Public release: README with results, license notes (Apache-2.0 for Meta's weights and this code), repo visibility flip, site deploy | Live on stevenmurley.com |

## Measurement plan for the case study

- Parity tables per module (fp32 max abs error, mask IoU) for Hiera-T and Hiera-S.
- DAVIS 2017 val J&F for both, next to the paper's Table (semi-supervised VOS).
- Speed: frames per second on M5 GPU (torch MPS) at 1024 and 512, and in-browser frame
  times on WebGPU and WASM for the tiny model.
- Export cost: int8 encoder vs fp32 mask IoU on the parity set.
- Ablations that are cheap and informative: memory bank size (num_maskmem 1/3/7), object
  pointers on/off, resolution 512 vs 1024, all measured as DAVIS J&F.

## Non-goals

Training or fine-tuning on SA-V; the data engine; the SA-V dataset; multi-object batching in
the browser (one object per demo run); mobile-phone performance targets.

## Change log

- 2026-10-22 (later): T6b landed (memory encoder, memory attention, pointer graphs; pure
  onnxruntime loop exact in fp32 at 1024 and 512; shipped-precision study in
  exports/README.md). DAVIS 2017 val Hiera-T J&F 87.65; the official package scores
  identically on a 3-sequence subset (docs/results/davis_reference_subset.json); the paper's
  89.4 (Table 17d) is for a model trained with internal data. T7 landed as the `web/`
  TypeScript runtime with Node parity tests (IoU 1.0 on all 48 clip frames and every image
  prompt). Repo moved from ~/Desktop (iCloud sets a hidden flag on .pth files, which Python
  3.12 then skips) to ~/sam2-from-scratch. D5 revised: the site ships the 512 set with fp32
  heads. Ablation chain (small base; tiny 512, mem1, noptr, mem3) running; T8 (site page +
  React demo) in progress.

- 2026-10-22: T2, T3, T4 landed with exact (0.0) parity against the official package on
  tiny and small: image encoder, SAM heads and image predictor, memory encoder, memory
  attention, assembled model (every checkpoint key consumed) and video predictor (bedroom
  clip forward and reverse, min IoU 1.0). Two reference details not in the YAML: the video
  predictor binarises prompted-frame masks before memory encoding, and memory features are
  stored in bfloat16. T6a landed: ONNX graphs for the image half at 1024 and 512 (int8
  weight-only encoder 32.5 MB using plain DequantizeLinear ops because ORT Web's WebGPU
  provider has no MatMulInteger kernel; fp16 heads ~10 MB; see exports/README.md). The
  mask grid at 512 is 128x128, not 256x256. DAVIS 2017 val downloaded (the ETH cvlvision
  URL is dead; the csergi mirror works) and the official evaluator installed;
  tools/eval_davis.py drives it.

- 2026-10-21: program created. Decisions D1 to D8 locked. Venv on Python 3.12 with torch
  (MPS), reference `sam2` package installed test-only, SAM 2.1 tiny and small checkpoints
  downloaded and hashed (`checkpoints/SHA256SUMS`).
