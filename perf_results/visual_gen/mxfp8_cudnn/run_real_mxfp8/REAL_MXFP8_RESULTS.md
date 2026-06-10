# REAL MXFP8 cuDNN SDPA vs bf16 VANILLA — Wan2.2-T2V-A14B (B200)

Generated 2026-06-10 on umbriel-b200-027, faithful study container
(`trtllm-mxfp8-liuc:027`, cuDNN 9.22, TE 2.12), patched backend
(**no silent fallback** + **chunked quantize** so MXFP8 actually runs at 720p),
8-GPU parallel (VANILLA + MXFP8 × 720p + 480p × 6 prompts). 40 steps, seed 42.

## Proof it is REAL MXFP8 (not bf16 fallback)

Per-call trace, both resolutions: **10,880 `path=mxfp8` + 10,880
`path=cross_or_unsupported_bf16` + 0 `fallback_exception`.** Every self-attention
call ran the cuDNN `sdpa_mxfp8` kernel; cross-attention is bf16 in both backends
(architectural, cancels in the comparison). The silent `except→bf16` fallback was
deleted, so these numbers cannot be secretly bf16.

## LPIPS (alex, native res, all frames) — REAL MXFP8 vs bf16 VANILLA

### 720p/81f (S=75600) — the production shape that previously silently fell back to bf16
| prompt | PSNR dB | LPIPS | %pix≠ |
|---|---|---|---|
| busy_street | 15.10 | 0.3419 | 95.9 |
| cat_windowsill | 22.31 | 0.2805 | 95.8 |
| dancer_jump | 20.85 | 0.2487 | 93.4 |
| drone_city_night | 19.22 | 0.2136 | 81.9 |
| flower_blooming | 16.72 | 0.3503 | 90.8 |
| ocean_sunset | 20.62 | 0.1888 | 88.4 |
| **mean** | **19.1** | **0.270** | **91%** |

### 480p/33f (S=14040)
| prompt | PSNR dB | LPIPS | %pix≠ |
|---|---|---|---|
| busy_street | 22.10 | 0.1904 | 89.5 |
| cat_windowsill | 21.76 | 0.1853 | 94.8 |
| dancer_jump | 23.06 | 0.1331 | 85.6 |
| drone_city_night | 23.01 | 0.1607 | 80.4 |
| flower_blooming | 21.77 | 0.2293 | 84.5 |
| ocean_sunset | 27.91 | 0.0806 | 88.2 |
| **mean** | **23.3** | **0.163** | **87%** |

## Headline

- **Real 720p MXFP8 ≈ 0.27 mean LPIPS (PSNR ~19 dB) — clearly visible degradation.**
  This is the colleague's "video 2" regime, now quantified. The old report's 720p
  "MXFP8 ≈ 0.10" was **bf16-vs-bf16** (silent fallback) — it **understated the true
  MXFP8 cost by ~2.7×**.
- **MXFP8 error grows with sequence length:** 720p (S=75600) ≈ 0.27 vs 480p
  (S=14040) ≈ 0.16 — FP8 attention error accumulates over the longer sequence.
- Both are well above the bf16 trajectory-divergence floor (~0.10), so this is a
  real FP8 effect, not just chaos — strongest at 720p.

Videos (3-panel VANILLA | MXFP8 | diff×4): `diff_videos/REAL720p_*.mp4`,
`REAL480p_busy_street.mp4`. Raw frames/mp4: `720p/prompts/`, `480p/prompts/`.
