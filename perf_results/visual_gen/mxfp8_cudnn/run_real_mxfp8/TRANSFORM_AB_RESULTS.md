# Do QK^T pre-transforms improve MXFP8 attention accuracy? — YES (real pipeline)

Generated 2026-06-10 on umbriel-b200-027 (faithful study container, cuDNN 9.22),
patched backend: **no silent fallback** + **chunked quantize** + env-gated Q/K
pre-transforms. 8-GPU parallel, 40 steps, seed 42, LPIPS-alex (native, all frames)
vs bf16 VANILLA. All MXFP8 variants verified REAL (per-call trace = `path=mxfp8` +
cross-attn bf16, **0 `fallback_exception`**); transforms confirmed ACTIVE in logs.

**Both transforms are exactly QK^T-invariant** (validated fp32, TF32-off: Q'·K'^T
rel-err ~1e-7) and applied **only to Q/K** (never V). So they change only the FP8
quantization error, not the attention math.

## 480p/33f (S=14040), 3 prompts — LPIPS vs bf16 VANILLA (lower better)

| variant | mean LPIPS | mean PSNR | Δ vs baseline |
|---|---|---|---|
| MXFP8 (none) | 0.1696 | 22.3 | — |
| + SmoothQuant **K** (α=0) | 0.1890 | 21.8 | **+11% (worse)** |
| + SmoothQuant **QK** (α=0.5) | 0.1450 | 23.8 | **−15% (better)** |
| + **Hadamard** rotation | **0.1370** | 23.6 | **−19% (best)** |

## 720p/81f (S=75600), 2 prompts — LPIPS vs bf16 VANILLA

| variant | mean LPIPS | mean PSNR | Δ vs baseline |
|---|---|---|---|
| MXFP8 (none) | 0.3140 | 18.7 | — |
| + **Hadamard** rotation | **0.2650** | 20.0 | **−16% (better)** |

## Verdict

- **Hadamard rotation works** — the staged-plan step-1. It cuts MXFP8↔bf16 LPIPS by
  **~19% at 480p and ~16% at 720p** (PSNR +1.3 dB), consistent per-prompt. It is the
  best single lever and reuses the repo's existing `hadamard_transform`.
- **SmoothQuant: direction matters.** K-only **hurts** (+11%); **balanced QK (α=0.5)
  helps ~15%** at 480p. This *corrects* the earlier synthetic probe ("≤2.2%, no help"):
  on real Wan Q/K, balanced migration does help — the synthetic gaussian+injected-
  outlier data missed the real channel structure. (Lesson: validate on the real pipeline.)
- Hadamard ≈ SmoothQuant-QK at 480p, both clearly above baseline; Hadamard scales to 720p.

## Caveats

- 2–3 prompts, single seed (42). A ~0.10 bf16 trajectory-divergence floor sits under
  every number (each variant is a separate generation), so treat magnitudes as ±0.02–0.03;
  the *ranking* is consistent per-prompt but more seeds/prompts would firm up the deltas.
- Dynamic per-call SmoothQuant scale (from current activation amax), not a calibrated scale.
- Hadamard adds two (D×D) matmuls per self-attn call (D=128, cheap) before quantization.

Backend: `tensorrt_llm/_torch/visual_gen/attention_backend/mxfp8_cudnn.py`
(`_smoothquant_qk`, `_hadamard_qk`). Run: `repro/run_transform_ab.sh`; invariance gate:
`repro/run_invcheck.sh`. Env: `TRTLLM_VISUAL_GEN_MXFP8_SMOOTHQUANT=k|qk|q`,
`TRTLLM_VISUAL_GEN_MXFP8_HADAMARD=1`.
