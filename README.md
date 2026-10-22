# SAM 2 from scratch

A from-scratch PyTorch reimplementation of the inference architecture of
**SAM 2: Segment Anything in Images and Videos** (Ravi et al., 2024, [arXiv:2408.00714](https://arxiv.org/abs/2408.00714)),
verified to zero numerical error against Meta's released SAM 2.1 weights, scored on DAVIS 2017,
exported to ONNX, and running as an interactive image and video segmentation demo in the browser at
[stevenmurley.com/projects/sam2](https://stevenmurley.com/projects/sam2).

Training is not reproduced (Meta trained the released model on 50.9K videos with 256 A100 GPUs for
108 hours). Everything that runs at inference time is.

## What is here

| Path | Contents |
|------|----------|
| `src/sam2_scratch/` | The model: `hiera.py` (hierarchical ViT with windowed and global attention), `image_encoder.py` (FPN neck), `prompt_encoder.py`, `transformer.py` (two-way transformer), `mask_decoder.py`, `memory_encoder.py`, `memory_attention.py` (2-D RoPE), `model.py` (assembly, strict checkpoint loading, per-frame tracking step), `image_predictor.py`, `video_predictor.py`. About 2,700 lines, typed, torch-only. |
| `tests/` | Parity tests against intermediate tensors dumped from the official package (`tools/dump_reference.py`), plus unit tests. |
| `tools/` | Reference dumps, DAVIS evaluation (`eval_davis.py`), ONNX export (`export_onnx.py`), onnxruntime parity (`ort_parity.py`), a pure-onnxruntime video loop (`ort_video_loop.py`), browser ground truth, figures. |
| `web/` | The browser runtime: a framework-free TypeScript port of the onnxruntime loop with Node parity tests. The website copies `web/src` into its `lib/sam2/`. |
| `exports/README.md` | Graph interfaces, the step-by-step video loop recipe, quantisation study, ORT Web notes. |
| `docs/PROGRAM.md` | Scope, decisions, task log. `docs/results/` holds the benchmark JSON. |

## Results

**Parity.** Every module, both released sizes (Hiera-T, Hiera-S), three images, a 48-frame clip
tracked forward and in reverse: maximum absolute error **0.0** against the official `sam2`
package on CPU (same operations in the same order; Apple MPS differs by about 1e-5). Loading
consumes every checkpoint key. Two inference details that the paper and configs do not state
were recovered from the numbers: masks of prompted frames are binarised before memory encoding,
and memory features are stored in bfloat16.

**DAVIS 2017 val, semi-supervised (first-frame masks), official evaluator.**

| Model | J&F | J | F | Notes |
|-------|----:|--:|--:|-------|
| Hiera-T (this repo, SAM 2.1 weights) | 87.65 | 84.89 | 90.42 | 30 sequences, 1024 px |
| Hiera-S (this repo, SAM 2.1 weights) | 87.96 | 85.31 | 90.62 | |
| Hiera-T, SAM 2 paper Table 17(d) | 89.4 | 85.8 | 92.9 | model trained with additional internal data, not released |
| Hiera-S, SAM 2 paper Table 17(d) | 89.6 | 86.3 | 92.9 | same |

On three sequences re-run with the official package and the public checkpoints, the official
code scores the same as this repo to within 0.002 points (`docs/results/davis_reference_subset.json`),
so the gap to the paper is the checkpoint, not the implementation.

**Ablations, Hiera-T on DAVIS 2017 val:** {{ABLATION_TABLE}}

**Browser.** Six ONNX graphs exported from this implementation at 512 px (and 1024 px in the
repo): image encoder (int8 weight-only, 32.5 MB), prompt encoder + mask decoder, memory encoder,
memory attention, two object-pointer graphs. A pure-onnxruntime loop reproduces the torch
predictor exactly in fp32; the TypeScript port reproduces that loop with IoU 1.0 on every frame
of the demo clip. Standard dynamic int8 quantisation of the encoder was rejected (0.93 mask IoU
against fp32, and ORT Web's WebGPU provider has no kernels for its operators); the shipped scheme
keeps the patch embedding and first two blocks in fp32 and reaches 0.99 IoU.

## Reproduce

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pip install --no-deps "git+https://github.com/facebookresearch/sam2.git"   # test-only reference
curl -o checkpoints/sam2.1_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
shasum -a 256 -c checkpoints/SHA256SUMS
python tools/dump_reference.py --model tiny      # intermediate tensors from the official package
pytest                                           # parity + unit tests
python tools/export_onnx.py --model tiny         # ONNX graphs into exports/
python tools/ort_parity.py --model tiny          # onnxruntime vs torch
python tools/eval_davis.py --model tiny          # needs data/davis/DAVIS (2017 trainval 480p)
cd web && npm install && npm test                # TypeScript runtime vs the Python loop
```

The sample media in `bundle/` come from the DAVIS 2017 dataset (CC BY-NC 4.0) and the
facebookresearch/sam2 notebook images (Apache-2.0); see `bundle/CREDITS.md`.

## License

Code: Apache-2.0. Model weights are Meta's, released under the Apache-2.0 license of the
[sam2](https://github.com/facebookresearch/sam2) repository.
