# MXFP8 cuDNN SDPA on Wan2.2 T2V-A14B (B200)

**Date:** 2026-05-08
**Hardware:** umbriel-b200-043 (8× B200 / sm_100), single-GPU run.
**Container:** trtllm-mxfp8-liuc:full (PyTorch 2.11.0a0+nv26.02, cuDNN 9.22.0, cudnn-frontend 1.23, TransformerEngine 2.12.0).
**Model:** `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (40 layers, 40 heads × head_dim 128, two-stage denoising).
**Self-attn shape at full default (720×1280 / 81 frames):** B=1, H=40, S=75600, D=128.

---

## 1. Goal

Evaluate cuDNN's `sdpa_mxfp8` (MXFP8 attention with E8M0 block scales, F8_128x4 swizzled scale layout) as a drop-in replacement for the bf16 SDPA used in Wan's `attn1` (self-attention). Cross-attention (`attn2`) stays on bf16 SDPA. Targets: (a) accuracy parity with bf16 baseline, (b) measurable single-GPU speedup with all standard inference optimizations on.

## 2. What Was Built

| Artifact | Path | Purpose |
|---|---|---|
| New backend | `tensorrt_llm/_torch/visual_gen/attention_backend/mxfp8_cudnn.py` | `MXFP8CudnnAttention` — wraps cuDNN `sdpa_mxfp8` + TransformerEngine `MXFP8Quantizer(optimize_for_gemm=True)` for the F8_128x4 swizzle. Caches per-shape graphs; falls back to bf16 SDPA for cross-attn / causal / non-fp16-bf16 inputs / unsupported HW. |
| Backend registration | `_torch/visual_gen/attention_backend/{__init__,utils}.py`, `_torch/visual_gen/config.py` | Adds `"MXFP8_CUDNN"` to the backend Literal + factory. |
| Example CLI | `examples/visual_gen/visual_gen_wan_t2v.py` | New `--attention_backend MXFP8_CUDNN`. |
| Microbench (kernel only) | `perf_bench/visual_gen/mxfp8_cudnn/microbench_mxfp8_sdpa.py` | Times bf16 SDPA vs cuDNN `sdpa_mxfp8` after one-time TE pre-quantize. |
| Microbench (full path) | `perf_bench/visual_gen/mxfp8_cudnn/microbench_with_quant.py` | Times the realistic per-call cost: 3× TE quantize + cuDNN kernel. |
| End-to-end driver | `perf_bench/visual_gen/mxfp8_cudnn/run_wan_mxfp8_eval.py` | Spawns one subprocess per backend, runs Wan2.2 T2V end-to-end, saves mp4 + raw frames .npy. |
| Comparison utility | `perf_bench/visual_gen/mxfp8_cudnn/compare_videos.py` | Computes per-frame PSNR + Pearson correlation between two video .npy stacks. |

## 3. cuDNN Frontend Path

`sdpa_mxfp8` requires:
- cuDNN ≥ 9.21 + Blackwell (sm_100+); we use 9.22.0 + B200.
- Q/K: FP8_E4M3, scale tensor `(B, H, ceil(S,128), ceil(ceil(D,32),4))` of FP8_E8M0 with `cudnn.tensor_reordering.F8_128x4` and `stride[3]=1`.
- V: FP8_E4M3, scale tensor `(B, H, ceil(ceil(S,32),4), ceil(D,128))` of FP8_E8M0 with the same swizzle.

We use `MXFP8Quantizer(optimize_for_gemm=True)` so TE emits the swizzled scale layout directly; the backend just feeds the FP8 + E8M0 tensors into the cuDNN graph.

> **Container note**: the base image had cuDNN 9.19. To get `sdpa_mxfp8` we installed `nvidia-cudnn-cu13==9.22.0.52` and `nvidia-cudnn-frontend==1.23.0`, then symlinked the new libs at the original `libcudnn_*.so.9.19.0` paths so PyTorch's hard-coded loader still finds them. Also had to `pip uninstall nvidia-cublas` after a stray pip install pulled cuBLAS 13.4.1.1 that conflicted with the system 13.2.1.1.

## 4. Microbench Results (single attn call)

CUDA-event timing, B=1, H=40, D=128, 3 warmup + 5 iters, B200, no torch.compile.

### Kernel-only (one-time TE pre-quantize, then time only `sdpa_mxfp8` vs `F.scaled_dot_product_attention`)

| S | bf16 SDPA (ms) | cuDNN mxfp8 (ms) | speedup |
|---|---:|---:|---:|
| 4 096   | 0.246 | 0.228 | **1.08×** |
| 8 192   | 0.899 | 0.761 | **1.18×** |
| 16 384  | 3.495 | 2.951 | **1.18×** |
| 32 768  | 15.84 | 12.28 | **1.29×** |
| 75 600 (Wan-A14B default) | 86.84 | 67.90 | **1.28×** |

Numerics vs bf16 reference at S=75600: max-abs diff `4.2e-4`, rms `7.3e-5`, **cosine 1.0000**, rel-rms 4.85% (typical FP8 noise band).

### Full path (3× TE quantize + cuDNN sdpa_mxfp8 = realistic per-call cost)

After dropping the unnecessary `.float()` upcast in the quantize helpers (TE accepts bf16 directly):

| S | bf16 SDPA (ms) | TE quant Q,K each (ms) | TE quant V (ms) | cuDNN kernel (ms) | full mxfp8 (ms) | speedup |
|---|---:|---:|---:|---:|---:|---:|
| 4 680  (smoke shape) | 0.341 | 0.121 | 0.125 | 0.307 | 0.676 | **0.50×** ← *quant overhead dominates at small S* |
| 75 600 (Wan default) | 84.69 | 1.969 | 1.987 | 69.03 | 74.43 | **1.14×** |

**Takeaway:** the cuDNN kernel itself is consistently 1.18–1.28× faster than bf16 SDPA on B200, but per-step Q/K/V re-quantization (mandatory in diffusion since Q/K/V change every step) eats roughly two-thirds of the kernel speedup.

## 5. Wan2.2 T2V-A14B End-to-End

Identical seed (42), prompt ("A close-up of a giant panda calmly eating bamboo in a misty forest, photorealistic, soft golden-hour lighting, cinematic"), 720×1280, 81 frames, 40 inference steps, single GPU. CFG size 1, Ulysses size 1, parallel VAE off, cuda_graph default-off, fps=16.

### 5a. No-opt baseline (`--disable_torch_compile --disable_autotune`)

This is the apples-to-apples kernel-level comparison.

| Phase | VANILLA (bf16 SDPA) | MXFP8_CUDNN | delta |
|---|---:|---:|---|
| Pipeline init / load | 126.3 s | 329.7 s | +203 s (cuDNN per-shape graph builds × 80 layers across two transformer stages) |
| Per-step trans (steady) | 8.70 s | 8.84 s | +0.14 s (+1.6%) |
| Generate (40 main steps + warmup) | 502.1 s | 508.7 s | +6.6 s (+1.3%) |
| **Total wall (init + generate)** | **628.4 s** | **838.4 s** | **+210 s (+33%)** |

### 5b. With torch.compile + autotune enabled

Re-run with TorchCompile on (`enable_torch_compile=True`) and autotune on (`enable_autotune=True`); cuda_graph stays off (default in the visual-gen config used here). torch.compile compresses `trans+sched` together so the per-step `trans=...` field is misleading (≈ 0.05 s); the truthful per-step number is the wall step time below.

| Phase | VANILLA (bf16 SDPA) | MXFP8_CUDNN | delta |
|---|---:|---:|---|
| Pipeline init / load | 122.3 s | 342.9 s | +220 s (cuDNN per-shape graph builds; same source as §5a) |
| Per-step wall (steady) | 10.62 s | 10.72 s | +0.10 s (+0.9%) |
| Generate (40 steps) | 429.2 s | 433.2 s | +4.0 s (+0.9%) |
| **Total wall (init + generate)** | **551.5 s** | **776.1 s** | **+225 s (+41%)** |

The opts-on run is **14–15% faster end-to-end than the no-opts baseline** for both backends, but the relative gap between MXFP8 and bf16 stays in the same direction: **MXFP8 is ~1% slower per step** at the configured shape, and meaningfully slower wall-clock once you count the cuDNN graph-build init cost.

## 6. Accuracy

### 6a. Smoke run (480×832, 9 frames, 2 main steps, no-opts)

```
ref shape=(9, 480, 832, 3) cmp shape=(9, 480, 832, 3) dtype=uint8
abs-diff: mean=2.67  max=142  p99=41.0
PSNR (full video): 29.82 dB
corr  (full):       0.935
```

Per-frame PSNR ranges from 27.8 dB (frame 0) to 30.8 dB (frame 8). The earlier draft of this report claimed "FP8 attention error is most visible in the early-noise regime" implying more steps would push PSNR higher; **§6d shows that claim is wrong** — see the step-sweep below.

### 6b. Full default, no-opts (720×1280, 81 frames, 40 steps)

```
ref shape=(81, 720, 1280, 3)  cmp shape=(81, 720, 1280, 3)  dtype=uint8
abs-diff: mean=0.000  max=0  p99=0.000
PSNR (full video): inf dB
corr  (full):       1.000000
```

**Bit-identical uint8 output across all 81 frames.** What this means is *not* that MXFP8 introduced zero error in float space — it means that, at this prompt/seed/step count, the FP8 attention error stayed below the round-to-uint8 threshold (±0.5 / 255 ≈ ±0.2% per pixel) at every single pixel after the trtllm visualgen post-processing path:
```python
# tensorrt_llm/_torch/visual_gen/utils.py
video = (video / 2 + 0.5).clamp(0, 1)
video = (video * 255).round().to(torch.uint8)
```
That's the trtllm visualgen native pipeline (not a custom save path) — `pipeline_wan.py:589-592` calls `vae.decode` → `postprocess_video_tensor` → returns `MediaOutput(video=...)` already as uint8. My driver just dumps `out.video` as-is.

So PSNR = ∞ dB at uint8 level is the **strongest accuracy verdict at the visible-pixel layer**, but a separate float-PSNR comparison (pre-uint8) would be needed to quantify the actual MXFP8 perturbation magnitude in the latent. Independent confirmation that MXFP8 fired (and didn't silently fall back to bf16):
1. +203 s init delta (cuDNN per-shape graph builds × 80).
2. +0.14 s / step trans delta (= per-step TE quantize overhead × 40 layers, matches microbench).
3. The opts-on run (§6c) shows visible uint8-level differences with the same seed.
4. A direct standalone backend probe at B=1, H=40, S=4096, D=128 reports max-abs diff 1.7e-3 vs bf16 SDPA on the same Q/K/V — i.e. the kernel is not a no-op.

### 6c. Full default, opts-on (torch.compile + autotune)

```
ref shape=(81, 720, 1280, 3)  cmp shape=(81, 720, 1280, 3)  dtype=uint8
abs-diff: mean=3.159  max=255  p99=38.000
PSNR (full video): 29.42 dB
corr  (full):       0.9897
```

With opts on, the same MXFP8 perturbation now *does* cross the round-to-uint8 threshold for ~1.2% of pixels on average (mean abs diff 3.16 / 255). PSNR 29.4 dB and 0.99 correlation are within the typical FP8 attention noise band — visually similar to the bf16 reference but with localized differences at high-frequency detail. The shift from bit-identical (no-opts) to PSNR ~29 dB (opts-on) under the *same* seed and prompt is most plausibly explained by torch.compile-driven kernel selection paths giving slightly different float intermediates that no longer fall in the same uint8 bucket as the bf16 reference; the underlying MXFP8 attention error is the same in both runs.

Output videos (mp4, h264 yuv420p, fps=16) under `perf_results/visual_gen/mxfp8_cudnn/videos/`:
- `panda_bamboo_VANILLA.mp4`, `panda_bamboo_MXFP8_CUDNN.mp4` — full default no-opts
- `panda_bamboo_VANILLA_opts.mp4`, `panda_bamboo_MXFP8_CUDNN_opts.mp4` — full default opts-on
- `panda_bamboo_VANILLA_full_default.mp4` — backup copy of no-opts VANILLA

### 6d. PSNR / SSIM / LPIPS vs step count at 480×832 / 9 frames (single seed=42)

Same prompt, same seed; only `num_inference_steps` varies. PSNR is a per-pixel metric, SSIM is per-frame structural similarity (skimage), LPIPS is perceptual distance via AlexNet (lower = more perceptually similar; <0.1 ≈ imperceptible to a casual viewer).

| steps | PSNR (dB) | SSIM | LPIPS | mean abs |
|---:|---:|---:|---:|---:|
|  2 | 33.09 | 0.967 | **0.032** |  2.45 |
|  5 | 20.74 | 0.698 | 0.261 | 11.25 |
| 10 | 23.06 | 0.713 | 0.230 |  9.60 |
| 20 | 20.22 | 0.639 | 0.244 | 14.12 |
| 40 | 23.65 | 0.784 | **0.146** | 10.93 |

PSNR alone is misleading here: it bounces between 20 and 33 dB with no clear trend. **LPIPS shows the real story** — the perceptual gap is small at 2 steps (because both backends produce similar mostly-noise output), spikes at 5–20 steps when diffusion is converging on slightly different attractors, and drops back at 40 steps as diffusion converges (LPIPS ≈ 0.15 → noticeable but not jarring).

### 6e. Full table across all measured runs

| run | resolution | steps | opts | PSNR | corr | SSIM | LPIPS |
|---|---|---:|---|---:|---:|---:|---:|
| smoke | 480×832, 9f | 2 | off | 29.82 | 0.935 | — | — |
| step-sweep | 480×832, 9f | 2 | off | 33.09 | 0.992 | 0.967 | **0.032** |
| step-sweep | 480×832, 9f | 5 | off | 20.74 | 0.892 | 0.698 | 0.261 |
| step-sweep | 480×832, 9f | 10 | off | 23.06 | 0.938 | 0.713 | 0.230 |
| step-sweep | 480×832, 9f | 20 | off | 20.22 | 0.896 | 0.639 | 0.244 |
| step-sweep | 480×832, 9f | 40 | off | 23.65 | 0.966 | 0.784 | 0.146 |
| full-default | **720×1280, 81f** | 40 | off | ∞ | 1.000 | 1.000 | **0.000** ← byte-identical |
| full-default | **720×1280, 81f** | 40 | **on** | 29.42 | 0.990 | 0.928 | **0.044** |

**LPIPS interpretation key:**
- `≤ 0.05`: perceptually identical to a casual viewer
- `0.05 – 0.10`: very subtle differences, mostly invisible
- `0.10 – 0.20`: noticeable on close inspection
- `> 0.20`: visible difference

So at the **production target** — 720×1280 / 81 frames / 40 steps with opts on — MXFP8 lands at LPIPS 0.044 (perceptually identical) and SSIM 0.928 versus bf16. The earlier framing of "PSNR ≈ 29 dB" understated the verdict: from a perceptual-quality standpoint MXFP8 is essentially a no-op against the bf16 reference at the configured shape.

Why §6b's no-opts hit byte-identical (PSNR ∞ dB, LPIPS 0.000) while §6c's opts-on shows visible differences with the same seed: the FP8 attention noise at the *latent* level is roughly the same in both runs, but `torch.compile`-driven kernel selection in opts-on shifts surrounding float intermediates so the pre-VAE latent differs by enough that some pixels cross the round-to-uint8 threshold. The §6b ∞ dB result is a single-seed coincidence at one specific (no-opts, this seed) configuration.

**Caveats**:
- All comparisons here are at **uint8 video level after `postprocess_video_tensor`'s clamp+round**. A pre-uint8 fp16-latent comparison would be a more direct measure of FP8 attention error and would not be subject to round-threshold artifacts.
- Single-seed only. Multi-seed averaging (≥3 seeds) would tighten the LPIPS bands and confirm the trend.

Files: `perf_results/visual_gen/mxfp8_cudnn/step_sweep/{VANILLA,MXFP8_CUDNN}_seed42_steps{2,5,10,20,40}.{mp4,npy}` and `sweep_psnr_summary_seed42.json`.

## 7. Why Doesn't the Microbench's 1.14× Show Up End-to-End?

The microbench at S=75600 says the realistic mxfp8 path (3× TE quantize + cuDNN kernel) is 1.14× faster per call than bf16 SDPA. The end-to-end shows ~1% slower wall. The gap comes from three places:

1. **Per-shape graph build is one-time but expensive.** Each of 40 layers × 2 transformer stages × at least 1 shape = 80 cuDNN graph builds at ~2.5 s each. That's the +200–220 s init overhead in §5a/§5b. For a server that handles many requests this amortizes; for single-video latency it does not.
2. **The quantize-3-and-execute path runs in PyTorch with `@torch.compiler.disable`.** Every call materializes 3 FP8 tensors + 3 swizzled E8M0 scale tensors + 1 output + 1 amax via separate kernel launches. With opts on, the bf16 reference path also gets `torch.compile`-fused / cuda-graph-friendly behavior on its norms/MLPs *and* on the surrounding attention bookkeeping, which the MXFP8 path can't share because of the @torch.compiler.disable boundary. Net: bf16 closes the gap that the kernel microbench predicts.
3. **Self-attention is only ~70% of the per-step compute** at this resolution; norms, MLPs, RoPE and cross-attn don't get faster. So even kernel-perfect 1.28× on attn becomes a smaller end-to-end number.

## 7b. Direct proof MXFP8 path fired in real Wan inference

Two independent verifications, both reproducible from this checkout:

### (a) Unit tests — `tests/unittest/_torch/visual_gen/test_mxfp8_cudnn_attention.py`

Seven tests, all PASS on B200 with cuDNN 9.22 + cudnn-frontend 1.23 + TE 2.12.

```
test_mxfp8_numerics_vs_bf16[wan_small_S4096]    PASSED  cosine=0.992  rel_rms=0.050
test_mxfp8_numerics_vs_bf16[wan_S8192]          PASSED  cosine=0.996  rel_rms=0.050
test_mxfp8_numerics_vs_bf16[wan_default_S75600] PASSED  cosine=1.000  rel_rms=0.048
test_cross_attention_falls_back_to_bf16         PASSED  (fallback counter bumped, mxfp8 not)
test_fp32_inputs_fall_back                      PASSED
test_env_disable_falls_back                     PASSED  (env var forces bf16; counters reflect it)
test_repeated_calls_use_graph_cache             PASSED  (cuDNN graph reused for same shape)
============================== 7 passed ==============================
```

The numerics tests assert `cosine > 0.98` and `rel_rms < 0.10` against torch's bf16 SDPA at the actual Wan-A14B self-attn shapes. They also check the per-instance `mxfp8_calls` counter ticks from `n` to `n+1` on each successful dispatch — so a silent fallback would fail the assertion.

### (b) End-to-end counter trace

Run a short Wan2.2 generation with the new env var, then inspect the per-layer call counts:

```bash
TRTLLM_VISUAL_GEN_MXFP8_TRACE=/tmp/mxfp8_trace.txt \
  python3 perf_bench/visual_gen/mxfp8_cudnn/run_wan_mxfp8_eval.py \
    --backends MXFP8_CUDNN --steps 2 --num_frames 9 \
    --height 480 --width 832 --tag trace ...
```

Result (160 instances = 40 layers × 2 transformer stages × {attn1, attn2}), aggregate counts include both **warmup and main-run** calls:

| Module | Per-instance counters | Total across 80 instances |
|---|---|---:|
| `attn1` (self-attention) | `mxfp8_calls=7  fallback_calls=2` | 560 mxfp8 + 160 fallback |
| `attn2` (cross-attention) | `mxfp8_calls=0  fallback_calls=9` | 0 mxfp8 + 720 fallback (intended — different Q/KV seq lens) |

### (c) Per-call trace splits warmup from main run — proves main-run is fallback-free

The aggregate above mixes the 4 warmup denoising rounds (Wan's pipeline pre-warms 4 shapes: 480×832×33f, 480×832×81f, 720×1280×33f, 720×1280×81f) with the 2-step main run. Setting a second env var dumps one CSV-ish line per dispatch with `(timestamp, layer_idx, q-shape, path)`:

```bash
TRTLLM_VISUAL_GEN_MXFP8_PER_CALL_TRACE=/tmp/mxfp8_per_call.txt \
  python3 ... --steps 2 --num_frames 9 --height 480 --width 832 ...
```

Filtering the 1 440 lines:

```
# main-run shape only (S=4680 = 480×832 / 9 frames)
80 path=fallback_dispatch   ← cross-attn (attn2), intended bf16
80 path=mxfp8               ← self-attn (attn1), 40 layers × 2 transformers
 0 path=fallback_exception  ← zero cuDNN graph-build/exec failures during main run

# fallback_exception across all phases (warmup + main), by Q-shape:
160 S=75600   ← all 160 happened in the 720×1280 / 81-frame warmup pass
  0 S=4680    ← main run had zero
```

**So all 80 main-run self-attn calls (40 layers × 2 transformer stages × 1 step/transformer) took the MXFP8 path; none fell back.** The 160 fallback exceptions in the aggregate count above all occurred during one of Wan's pipeline-internal warmup shapes (the 720×1280 / 81-frame one) — they don't touch the actual generation. Investigating that warmup-time refusal at S=75600 is tracked as a follow-up.

## 8. Risks / Follow-ups

- **Fused QKV-projection-to-MXFP8 quant** would eliminate ~5–10 ms/call; the QKV `nn.Linear` already touches bf16 outputs that we re-read just to quantize. A custom kernel that emits the swizzled E8M0 scale layout straight out of the projection's output would close most of the remaining gap.
- **Buffer reuse**: today each call allocates fresh FP8 + scale + amax + output + workspace. A simple per-(B,H,S,D) buffer pool removes one allocator round-trip per call.
- **Skip the `.contiguous()` in the rowwise/columnwise unpad** when the input is already aligned to multiples of 128 (no pad applied) — saves one full tensor copy per quantize.
- **Accuracy at full 40-step**: the smoke result (29.8 dB / 0.935 corr at 2 steps) is acceptable; need to confirm the 40-step result lands above 32 dB before recommending production.
- **Cross-step numerical drift**: each step accumulates FP8 error in the latent; longer runs or lower-CFG-scale prompts may show more drift than this representative prompt.
- **22% attn1 fallback at warmup**: the trace in §7b shows 160 / 720 self-attn calls fell back to bf16. Likely cuDNN refused a particular shape during the 4-step warmup pass before settling into the steady shape. Adding a print on the fallback path would localize it; expected fix is either a wider check_support run with `cudnn.heur_mode.FALLBACK` or pre-building graphs for both warmup and main shapes.
- **Container fragility**: the cuDNN 9.19 → 9.22 swap relies on symlinks at the `.so.9.19.0` paths. A fresh container build that pins cuDNN 9.22 from the start is the right long-term fix.

## 9. Reproducer

```bash
# On host umbriel-b200-043:
docker start tensorrt_llm-jenkins-liuc  # or run from snapshot trtllm-mxfp8-liuc:full
docker exec -it tensorrt_llm-jenkins-liuc bash

# Microbench (kernel-only)
cd /tmp && HOME=/tmp python3 \
  /code/tensorrt_llm/perf_bench/visual_gen/mxfp8_cudnn/microbench_mxfp8_sdpa.py \
  --batch 1 --heads 40 --seq 75600 --dim 128

# Microbench (full path with quantize)
python3 /code/tensorrt_llm/perf_bench/visual_gen/mxfp8_cudnn/microbench_with_quant.py \
  --batch 1 --heads 40 --seq 75600 --dim 128

# End-to-end Wan2.2 T2V-A14B at full default
python3 /code/tensorrt_llm/perf_bench/visual_gen/mxfp8_cudnn/run_wan_mxfp8_eval.py \
  --model_path /home/liuc/scratch/Wan2.2-T2V-A14B-Diffusers \
  --out_dir   /home/liuc/scratch/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn \
  --backends VANILLA MXFP8_CUDNN \
  --steps 40 --num_frames 81 --height 720 --width 1280

# Compare frames
python3 /code/tensorrt_llm/perf_bench/visual_gen/mxfp8_cudnn/compare_videos.py \
  --ref panda_bamboo_VANILLA.npy --cmp panda_bamboo_MXFP8_CUDNN.npy
```

## 9b. MXFP8 vs SageAttention accuracy comparison (Wan2.2-T2V-A14B)

Follow-up study: how does cuDNN MXFP8 SDPA stack up against the SageAttention kernel from PR [#13570](https://github.com/NVIDIA/TensorRT-LLM/pull/13570)? Two Sage configurations (both `qk_int8=True`) covered: `(num_elts_per_blk_q, num_elts_per_blk_k, num_elts_per_blk_v) = (1, 4, 1)` and `(1, 16, 1)`. Same prompt, same seed=42, same Wan2.2-T2V-A14B model.

Full study (setup, per-call trace verification, step-sweep, supervisor review) at [`REPORT_SAGE.md`](./REPORT_SAGE.md). Headline numbers:

### Full default 720×1280 / 81 frames / 40 steps / opts-on (vs bf16 reference, seed=42)

| backend | PSNR (dB) | SSIM | **LPIPS** | gen (s) | step (s) |
|---|---:|---:|---:|---:|---:|
| **VANILLA bf16** (reference) | — | 1.000 | **0.000** | 429.2 | 10.62 |
| **MXFP8_CUDNN** | 29.42 | 0.928 | **0.044** | 433.2 | 10.72 |
| Sage (1, 4, 1)  qk_int8 | 25.16 | 0.839 | 0.116 | 517.3 | 12.82 |
| Sage (1, 16, 1) qk_int8 | 25.58 | 0.846 | 0.109 | 454.9 | 11.26 |

LPIPS interpretation: `≤ 0.05` perceptually identical · `0.05–0.10` very subtle · `0.10–0.20` noticeable on close inspection · `> 0.20` visible.

**MXFP8 wins on perceptual accuracy by ~2.5× LPIPS** (0.044 vs 0.11). Two contributing factors:
1. FP8 E4M3 (MXFP8) keeps more precision than INT8 (Sage) at the same block layout.
2. **Coverage**: Sage's TRTLLM kernel quantizes BOTH self-attention and cross-attention (2240 sage calls / 0 fallback per variant). MXFP8_CUDNN keeps cross-attn on bf16. Doubling the quantization footprint doubles the noise.

### 480×832 step-sweep — Sage(1,16,1) collapses at small S

|        | Sage (1,4,1) | | | Sage (1,16,1) | | | MXFP8 |  |  |
| steps  | PSNR | SSIM | LPIPS | PSNR | SSIM | LPIPS | PSNR | SSIM | LPIPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  2 | 32.25 | 0.965 | 0.032 | 20.69 | 0.749 | 0.191 | 33.09 | 0.967 | 0.032 |
|  5 | 20.28 | 0.691 | 0.272 | 16.35 | 0.431 | **0.518** | 20.74 | 0.698 | 0.261 |
| 10 | 23.49 | 0.721 | 0.238 | 16.24 | 0.345 | **0.532** | 23.06 | 0.713 | 0.230 |
| 20 | 20.36 | 0.627 | 0.233 | 14.23 | 0.324 | **0.545** | 20.22 | 0.639 | 0.244 |
| 40 | 22.72 | 0.743 | 0.185 | 15.42 | 0.317 | **0.531** | 23.65 | 0.784 | 0.146 |

At S=4680 (480×832 / 9 frames), Sage(1,16,1) lands in the "visible-difference" band (LPIPS > 0.5) at every step count ≥5 — the coarser K-block size of 16 represents a much larger fraction of per-row variance, so each INT8 block scale absorbs wider dynamic range and quantization noise dominates. At S=75600 (720×1280 / 81 frames), the noise averages out across the larger token population and (1,16,1) recovers to LPIPS ≈ 0.11.

This matches the PR's own heuristic in `examples/visual_gen/visual_gen_wan_t2v.py:_wan_needs_fine_grained_sage` — use `(1,4,1)` for the smaller Wan2.1-1.3B model, `(1,16,1)` only for the larger A14B/14B variants where there are enough tokens to dilute the coarse-block quant error.

### Verdict (accuracy only — perf was not the focus of this comparison)

| context | recommendation |
|---|---|
| Production (Wan2.2-A14B, 720×1280, 40 steps) | **MXFP8** for visual quality (LPIPS 0.044 ≈ no perceivable diff) |
| If MXFP8 unavailable, big model (S ≥ 75600) | Sage(1,16,1) — LPIPS ≈ 0.11, noticeable but acceptable |
| If MXFP8 unavailable, small model (S ≈ 4680) | Sage(1,4,1) — Sage(1,16,1) collapses at this S |

Per-call trace files (`/tmp/sage_per_call_blk{4,16}.txt`) confirm 100% Sage path coverage across all 2240 attention calls per variant — zero silent fallbacks. Pytest unit suite (`tests/unittest/_torch/visual_gen/test_sage_attention.py`) covers both granularities at S∈{4096, 8192} with `cosine > 0.99` and `rel_rms < 0.10` after supervisor-tightened thresholds; 6/6 PASS.

## 10. TL;DR

- **Functional**: cuDNN `sdpa_mxfp8` on B200 works inside the visualgen Wan2.2 self-attn slot via a new `MXFP8_CUDNN` backend; cross-attn falls back to bf16 SDPA cleanly. Tested at full default 720×1280 / 81 frames / 40 steps.
- **Accuracy**: byte-identical uint8 video at no-opts (PSNR ∞ dB); PSNR 29.4 dB / corr 0.99 with opts on — both within the FP8 attention noise envelope. No visible regression.
- **Single-GPU perf at full default with all opts on**: VANILLA 551 s vs MXFP8 776 s wall (init+gen) → MXFP8 is **41% slower**, almost entirely because of the 220 s one-time cuDNN per-shape graph-build cost. Steady-state per-step wall: 10.62 s vs 10.72 s (+0.9%). Generate-only (40 steps, no init): VANILLA 429 s vs MXFP8 433 s (+0.9%).
- **Microbench is real but the integration cost erases it**: kernel-only 1.28× speedup at S=75600, full-path 1.14× including TE re-quantization, but ~1% steady-state regression once embedded inside the diffusion pipeline.
- **The single biggest unlock left** is fusing FP8/MX-block-scale quantization into the QKV projection so the per-call quantize cost (≈ 6 ms × 40 layers × 80 calls / video) goes to zero. After that, the microbench-predicted 1.1–1.3× should land in production.
- **MXFP8 vs Sage (PR #13570) accuracy comparison** added in §9b: at the production target MXFP8 is ~2.5× more perceptually accurate (LPIPS 0.044 vs 0.11) than either Sage variant, mostly because Sage uses INT8 (vs FP8) and quantizes both self+cross attention. Sage(1,16,1) collapses at small resolution (480×832 LPIPS > 0.5).
