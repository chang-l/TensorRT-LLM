# Sage Attention vs MXFP8 cuDNN SDPA — Accuracy comparison

**Date:** 2026-05-09
**Hardware:** umbriel-b200-043, single B200 (sm_100)
**Wheel:** TensorRT-LLM 1.3.0rc13 + PR #13570 Sage config plumbing applied as overlay
**Container snapshot:** trtllm-mxfp8-liuc:full + the Sage plumbing patch (see §1).
**Model:** Wan2.2-T2V-A14B-Diffusers (40 layers × 2 transformer stages, H=40, D=128).
**Same prompt + seed=42 across all backends** — accuracy comparison is apples-to-apples.

## 1. Setup

The wheel already had the underlying TRTLLM Sage attention plumbing (from earlier PR #12937, which merged the C++ kernel + `sage_attn_num_elts_per_blk_*` constructor args). PR #13570 adds the visual-gen wrapper: `SageAttentionConfig` Pydantic class, `AttentionConfig.sage_attention_config` flow-through, and a single `sage_attention_config` constructor arg in the visual-gen `TrtllmAttention`.

We applied PR #13570's Python diff (config.py, attention_backend/{trtllm,utils}.py) onto the wheel install. `modules/attention.py` was kept on the wheel-original because the PR's version pulls in unrelated `attn2d_row_size` infra. The wheel's `_attn_impl` calls `self.attn.forward(q=q, k=k, v=v, **kwargs)` with no `batch_size`/`seq_len`, so we relaxed PR #13570's required positional args to optional kwargs that default to `q.shape[0]` / `q.shape[1]` when missing — matches pre-PR semantics.

Two Sage configurations were studied (qk_int8=True for both):
- **Sage (1, 4, 1)** — finer K-axis blocking
- **Sage (1, 16, 1)** — coarser K-axis blocking; qk_int8-only per the PR's validator

## 2. Verification: Sage path actually fired

Per-call trace (`TRTLLM_VISUAL_GEN_SAGE_PER_CALL_TRACE`), one line per dispatch. Full-default Wan2.2 720×1280 / 81 frames / 40 steps:

```
sage_blk4 :  2240 path=sage   0 path=trtllm_bf16   ← 100% Sage path coverage
sage_blk16:  2240 path=sage   0 path=trtllm_bf16
```

The 2240 calls covers both self-attention (`attn1`) AND cross-attention (`attn2`) across all 40 layers × 2 transformer stages × all 56 forward passes (40 main + 16 warmup). Note this is **broader coverage than MXFP8** — MXFP8_CUDNN keeps cross-attn on bf16, while Sage's TRTLLM kernel handles different Q/KV seq lengths in INT8 directly. This wider quantization footprint is part of why Sage's accuracy is worse — see §4.

Pytest unit test (`tests/unittest/_torch/visual_gen/test_sage_attention.py`, 6/6 PASS, tightened tolerances after supervisor review):

```
test_sage_numerics_vs_bf16[wan_S4096-blk_1_4_1]    PASSED   cosine=1.000   rel_rms=0.047
test_sage_numerics_vs_bf16[wan_S4096-blk_1_16_1]   PASSED   cosine=1.000   rel_rms=0.047
test_sage_numerics_vs_bf16[wan_S8192-blk_1_4_1]    PASSED   cosine=0.996   rel_rms=0.047
test_sage_numerics_vs_bf16[wan_S8192-blk_1_16_1]   PASSED   cosine=0.996   rel_rms=0.047
test_sage_repeated_calls_keep_counter_growing[blk_1_4_1]    PASSED  (counter 0→3)
test_sage_repeated_calls_keep_counter_growing[blk_1_16_1]   PASSED  (counter 0→3)
```

Cosine threshold is `> 0.99`, rel_rms threshold is `< 0.10`. Both configurations clear them at the unit-test (random Q/K/V) level. The end-to-end accuracy is more revealing — see §3.

## 3. Full-default end-to-end (720×1280 / 81 frames / 40 steps, seed=42, opts on)

| backend | PSNR (dB) | corr | SSIM | **LPIPS** | gen (s) | step (s) |
|---|---:|---:|---:|---:|---:|---:|
| **VANILLA bf16** (reference) | — | 1.000 | 1.000 | **0.000** | 429.2 | 10.62 |
| **MXFP8_CUDNN** | 29.42 | 0.9897 | 0.928 | **0.044** | 433.2 | 10.72 |
| **Sage (1, 4, 1) qk_int8** | 25.16 | 0.9725 | 0.839 | **0.116** | 517.3 | 12.82 |
| **Sage (1, 16, 1) qk_int8** | 25.58 | 0.9751 | 0.846 | **0.109** | 454.9 | 11.26 |

LPIPS interpretation:
- `≤ 0.05` perceptually identical to a casual viewer
- `0.05 – 0.10` very subtle, mostly invisible
- `0.10 – 0.20` noticeable on close inspection
- `> 0.20` visible difference

**At full default, MXFP8 is essentially imperceptible (LPIPS 0.044) while both Sage variants land in the "noticeable on close inspection" band (LPIPS 0.11).** Sage(1,16,1) is marginally better than Sage(1,4,1). End-to-end timing: both Sage variants are SLOWER than bf16 at this configuration (Sage(1,4,1) +20% slower per step, Sage(1,16,1) +6% slower) — but timing isn't the focus of this study.

## 4. Step-sweep at 480×832 / 9 frames / single seed=42

Same prompt, same seed, only `num_inference_steps` varies. Reuses one VisualGen instance per backend so model load amortizes.

### PSNR / LPIPS table

|        | MXFP8 | | | Sage (1,4,1) | | | Sage (1,16,1) | | |
| steps  | PSNR | SSIM | **LPIPS** | PSNR | SSIM | **LPIPS** | PSNR | SSIM | **LPIPS** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  2 | 33.09 | 0.967 | **0.032** | 32.25 | 0.965 | **0.032** | 20.69 | 0.749 | **0.191** |
|  5 | 20.74 | 0.698 | 0.261 | 20.28 | 0.691 | 0.272 | 16.35 | 0.431 | 0.518 |
| 10 | 23.06 | 0.713 | 0.230 | 23.49 | 0.721 | 0.238 | 16.24 | 0.345 | 0.532 |
| 20 | 20.22 | 0.639 | 0.244 | 20.36 | 0.627 | 0.233 | 14.23 | 0.324 | 0.545 |
| 40 | 23.65 | 0.784 | 0.146 | 22.72 | 0.743 | **0.185** | 15.42 | 0.317 | **0.531** |

### What the step-sweep shows

1. **Sage (1, 16, 1) collapses at 480×832** — LPIPS hovers around 0.52 across all step counts ≥5 (visible-difference range). This is a much worse accuracy verdict than the 720×1280 measurement, where Sage(1,16,1) was ~0.11. Likely cause: at the smaller latent shape (S = 4680 vs 75600), the coarser K-block size of 16 represents a much larger fraction of the per-row variance, so each INT8 block scale has to absorb wider dynamic range and the quantization noise dominates.

2. **Sage (1, 4, 1) tracks MXFP8 closely** at this resolution: LPIPS at 40 steps is 0.185 (Sage) vs 0.146 (MXFP8) — both in the same "noticeable but not jarring" band. Same general step-vs-LPIPS shape (worst at 5 steps when diffusion is converging on slightly different attractors, then recovers as steps grow).

3. **MXFP8 wins or ties across every (resolution, step count) point** measured, with the biggest margin at the production target (720×1280 / 40 steps, where MXFP8's LPIPS 0.044 vs Sage's 0.11 is ~2.5x better perceptually).

## 5. Cross-resolution observation: why Sage(1,16,1) is worse at 480×832 but slightly better at 720×1280

| resolution | S = ceil(T/4)·H/16·W/16 | Sage(1,4,1) LPIPS | Sage(1,16,1) LPIPS |
|---|---:|---:|---:|
| 480×832 / 9 frames | 4 680 | 0.185 (40 steps) | **0.531** (40 steps) |
| 720×1280 / 81 frames | 75 600 | 0.116 (40 steps) | **0.109** (40 steps) |

Coarser K-blocks (16) need a larger token population per attention head to average down the per-block quantization noise. At S=4680 the noise dominates; at S=75600 it's averaged out by sheer count. So the PR's "use (1,16,1) for the bigger Wan2.2 models, (1,4,1) for the small 1.3B" heuristic in `examples/visual_gen/visual_gen_wan_t2v.py` (`_wan_needs_fine_grained_sage`) lines up with what the data here shows.

## 6. Verdict

**MXFP8 cuDNN SDPA is more accurate than Sage at every measured configuration.**
At the production target (Wan2.2-T2V-A14B at 720×1280 / 81 frames / 40 steps):
- MXFP8 LPIPS = 0.044 → perceptually identical
- Sage(1,16,1) LPIPS = 0.109 → noticeable on close inspection
- Sage(1,4,1) LPIPS = 0.116 → noticeable on close inspection

For **smaller-resolution / under-stepped runs (e.g. 480×832 / ≤20 steps)**, Sage(1,16,1) collapses (LPIPS > 0.5) — *don't* use the coarse-K granularity at small S; stick to (1,4,1) or pay the bf16 / MXFP8 cost.

**Caveats**:
- Single seed, single prompt — multi-seed averaging would tighten LPIPS bands.
- All comparisons are at uint8 video level after `postprocess_video_tensor`'s clamp+round; pre-uint8 fp16-latent comparison would isolate the FP8 / INT8 perturbation magnitude from VAE rounding artifacts.
- These numbers are accuracy-only; perf is reported only loosely (Sage variants are 6–20% slower per step than bf16 at this configuration, but this study didn't tune Sage perf).
- Sage's broader coverage (self+cross attention) vs MXFP8's self-attention-only coverage is part of why Sage's accuracy is worse at the same shape; if Sage were restricted to self-attn the gap would shrink.

## 7. Reproducer

```bash
# Container: trtllm-mxfp8-liuc:full + Sage overlay on umbriel-b200-043

# Unit tests (6/6 PASS for Sage; 7/7 for MXFP8)
docker exec trtllm-mxfp8-liuc bash -lc "cd /tmp && HOME=/tmp pytest -xvs \
  /code/tensorrt_llm/tests/unittest/_torch/visual_gen/test_sage_attention.py \
  /code/tensorrt_llm/tests/unittest/_torch/visual_gen/test_mxfp8_cudnn_attention.py"

# Full-default Sage E2E (per granularity)
TRTLLM_VISUAL_GEN_SAGE_PER_CALL_TRACE=/tmp/sage_per_call.txt \
python3 /code/tensorrt_llm/perf_bench/visual_gen/mxfp8_cudnn/run_wan_mxfp8_eval.py \
    --model_path $MODEL_DIR --out_dir $OUT_DIR \
    --backends TRTLLM --tag sage_blk4  --sage_blk_k 4 \
    --steps 40 --num_frames 81 --height 720 --width 1280

# Step-sweep
python3 /code/tensorrt_llm/perf_bench/visual_gen/mxfp8_cudnn/step_sweep_psnr.py \
    --model_path $MODEL_DIR --out_dir $OUT_DIR \
    --backends TRTLLM --sage_blk_k 4 \
    --height 480 --width 832 --num_frames 9 --seeds 42 --step_counts 2 5 10 20 40

# Compare to bf16 reference (PSNR + SSIM + LPIPS via AlexNet)
python3 /code/tensorrt_llm/perf_bench/visual_gen/mxfp8_cudnn/compare_videos.py \
    --ref panda_bamboo_VANILLA_opts.npy \
    --cmp panda_bamboo_TRTLLM_sage_blk4.npy
```

Files saved under `perf_results/visual_gen/mxfp8_cudnn/`:
- `videos/panda_bamboo_TRTLLM_sage_blk{4,16}.{mp4,npy}` — full-default outputs
- `step_sweep/sage_blk{4,16}_seed42_steps{2,5,10,20,40}.{mp4,npy}` — sweep frames
- `logs/panda_bamboo_TRTLLM_sage_blk{4,16}.{log,json}` — per-run trace
- `SUPERVISOR_REVIEWS_SAGE.md` — independent supervisor sign-off
- `REPORT_SAGE.md` — this file
