# Hadamard Rotation To Recover MXFP8 Attention Accuracy

## Original Idea

now, you have the context, mxfp8 is also not so good, wondering why? smooth quant would not help on the accuracy? why? and based one someone's point: in theory, attention's Bmm1 is more sensitive to mantissa bits than exponent bits due to fact that softmax is actually normalized exponential function, that extending dynamic range alone with MX might not immediately help with precision.
can you giver me someidea to improve mxfp8 accuracy?

## Primary Direction: Hadamard / Rotation Pre-Transform on head_dim

### Rationale

A dense orthogonal rotation (e.g. Hadamard) on the D=head_dim axis, applied identically to Q and K, cancels exactly in the QK^T contraction (like SmoothQuant) — but, unlike SmoothQuant's diagonal per-channel scaling, it reduces intra-32-block kurtosis/outliers so E4M3's 3 mantissa bits are better utilized within each MX block. This directly attacks the mantissa bottleneck that SmoothQuant cannot, which is exactly why SmoothQuant gave ≤2.2% in the probe: it only redistributes per-channel magnitude (a dynamic-range game that MX block-scaling already wins), while a rotation reshapes the value *distribution* toward something the limited mantissa can represent.

### Approach Summary

Insert a fixed orthogonal rotation R (head_dim × head_dim, e.g. a Walsh–Hadamard matrix scaled by D^-0.5) applied identically to Q and K on the D axis, immediately before MXFP8 quantization, in the self-attention path.

1. Hook point: `tensorrt_llm/_torch/visual_gen/modules/attention.py`, between the QK-norm and the backend `_attn_impl()` dispatch — apply `Q' = Q @ Rᵀ`, `K' = K @ Rᵀ` per head.
2. Invariance: because R is orthogonal and applied identically to Q and K, `Q'·K'ᵀ = Q·R·Rᵀ·Kᵀ = Q·Kᵀ` in exact arithmetic — the softmax logits are unchanged, so V and the rest of attention need no inverse transform (only Q/K are rotated; the rotation is purely a pre-quantization conditioning step).
3. Quantization benefit: the rotated Q'/K' have lower per-channel kurtosis (energy spread across D), so within each 32-element MX block the shared E8M0 scale covers a tighter effective range and E4M3's 3 mantissa bits land where the values actually are.
4. Reuse the existing `hadamard_transform` wrapper already vendored in the repo; gate behind a config flag (default off) so it is opt-in and A/B-testable against the committed REAL_MXFP8 LPIPS baseline.
5. No change to the cuDNN `sdpa_mxfp8` graph — the rotation lives upstream of `_quantize_q_or_k_along_d`, so the kernel and its scale layout are untouched.

### Objective Evidence

- Existing, currently-UNUSED rotation primitive ready to extend: `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` defines `rotate_activation()` (≈lines 237–253) calling `fast_hadamard_transform.hadamard_transform(x, scale=hidden_size**-0.5)` (import ≈line 49); grep confirms it is defined but never invoked — a drop-in Hadamard transform already in the dependency set.
- Clean pre-quantization hook: `tensorrt_llm/_torch/visual_gen/modules/attention.py` — `apply_qk_norm(q,k)` (≈244–248), the unfused QK-norm path (≈361–362), and the `_attn_impl(q,k,v)` dispatch (≈374) bracket the exact insertion point, upstream of the MXFP8 backend.
- Quantization entry points are upstream-hookable: `tensorrt_llm/_torch/visual_gen/attention_backend/mxfp8_cudnn.py` `_quantize_q_or_k_along_d()` (≈124–161) is called on Q and K just before the cuDNN graph; a rotation before this call needs no cuDNN-graph change.
- Orthogonal-transform-on-attention precedent: RoPE infra in `tensorrt_llm/_torch/modules/rotary_embedding.py` and `WanRotaryPosEmb` in `tensorrt_llm/_torch/visual_gen/models/wan/transformer_wan.py` (≈35–92) already apply fixed per-position rotations to Q/K.
- Direct study endorsement: `perf_results/visual_gen/mxfp8_cudnn/repro/SMOOTHQUANT_PROBE.md` names "Hadamard/rotation (QuaRot/SpinQuant-style) on D, shared between Q and K" as the most promising next step over SmoothQuant, for "mantissa utilization within each MX block."
- Target shape is compatible: Wan2.2 self-attn head_dim D=128 (and 64 elsewhere) are powers of two → directly Hadamard-transformable.

### Known Risks

- `fast_hadamard_transform` requires power-of-2 head_dim; D=128 satisfies this but a manual Hadamard-matrix fallback is needed for portability / CI environments where the wheel is absent.
- MX computes its block scale *along D*, the same axis the rotation acts on; the block-scale may partially re-absorb the rotation's benefit. The kurtosis-reduction → LPIPS-improvement link is mechanistically sound but unmeasured on real Wan Q/K (the SmoothQuant probe used synthetic outliers).
- Only helps self-attention (attn1); cross-attention stays bf16 in both backends, so the win is bounded to the self-attn share.
- Correctness is unforgiving: R must be exactly orthogonal and applied identically to Q and K, or the logits drift; a test must assert QK^T is bit-identical pre/post rotation in fp32.
- Magnitude of benefit is unknown — the 720p gap is ~0.27 LPIPS; rotation may close only a fraction, leaving a residual that still needs mixed precision.

## Alternative Directions Considered

### Alt-1: Selective Mixed Precision by Matmul (bf16 QK^T, MXFP8 PV)
- Gist: Keep the mantissa-sensitive Bmm1 (QK^T → softmax) in bf16 and apply MXFP8 only to the less-sensitive Bmm2 (softmax @ V), directly honoring the "softmax is mantissa-sensitive" insight by never quantizing the logit inputs.
- Objective Evidence:
  - `tensorrt_llm/_torch/visual_gen/attention_backend/mxfp8_cudnn.py` builds the cuDNN `sdpa_mxfp8` graph with Q/K/V all FP8_E4M3 and a hardcoded BFLOAT16 output — cudnn-frontend 1.23 exposes no mixed-input-precision knob, so this requires a manual Bmm1/Bmm2 split.
  - Separate per-matmul FP8 scales (`fmha_bmm1_scale` / `fmha_bmm2_scale`) appear in `tensorrt_llm/_torch/attention_backend/trtllm_gen.py`, and CUTLASS ships `examples/.../blackwell_mixed_mxfp8_bf16_gemm` — precedent that mixed-precision Bmm is a real kernel pattern.
  - The three `_quantize_*()` calls in mxfp8_cudnn.py already decouple Q/K vs V quantization.
- Why not primary: cuDNN's single `sdpa_mxfp8` op can't run mixed precision, so this forces a manual torch.bmm decomposition that likely erases the MXFP8 speedup — it is the surest *accuracy* win but the weakest *speed* story, the opposite of MXFP8's purpose.

### Alt-2: Finer MX Scaling Granularity (sub-32 block / two-level scale)
- Gist: Shrink the shared-scale block from 32 to 16 (or 8) elements, or add a second finer scale level, so fewer elements share one E8M0 exponent and the E4M3 mantissa covers a tighter intra-block range.
- Objective Evidence:
  - FP4 already templates the block size: `cpp/tensorrt_llm/kernels/quantization.h` (`template <typename T, int SF_VEC_SIZE = 16>`) and `quantization.cu` instantiate both 16 and 32 — proven dual-block-size machinery to mirror for MXFP8.
  - MXFP8 hardcodes 32: `cpp/tensorrt_llm/thop/mxFp8Quantize.cpp` (`static constexpr int SF_VEC_SIZE = 32`) and `quantization.cu` (≈line 197); the warp-reduce in `quantization.cuh` (≈617–623) already branches for 16 vs 32.
  - Two-level scaling precedent (NVFP4 per-16 E4M3 group scales + FP32 global) in `cpp/tensorrt_llm/kernels/fusedMoeCommKernels.cu` and `tensorrt_llm/_torch/auto_deploy/.../quant.py` (`TRTLLM_NVFP4_SCALING_VECTOR_SIZE = 16`).
- Why not primary: the strongest raw code precedent, but the cuDNN `sdpa_mxfp8` kernel likely assumes a 32-element block internally — if so, finer blocks can't reach the cuDNN path without re-blocking, which is the critical unverified blocker.

### Alt-3: Element-Format Exploration (E5M2 / NVFP4 / MXFP4)
- Gist: Test whether a different low-bit element encoding (E5M2, NVFP4, MXFP4) helps the mantissa-sensitive QK^T relative to E4M3.
- Objective Evidence:
  - E4M3 has 3 mantissa bits; E5M2 has 2; NVFP4/MXFP4 (E2M1) have 1 — every available low-bit format has *fewer* mantissa bits than E4M3.
  - cuDNN `sdpa_mxfp8` is hardcoded to FP8_E4M3 (`io_data_type` in mxfp8_cudnn.py); TE `MXFP8Quantizer` exposes E5M2 but the cuDNN op does not.
  - `SMOOTHQUANT_PROBE.md` already states "Neither NVFP4 nor MXFP4 will help — both have fewer mantissa bits than E4M3."
- Why not primary: essentially a negative control — it confirms the diagnosis (E4M3 is already the best low-bit option; the only way up in mantissa is more bits, i.e. mixed precision) rather than providing a fix.

### Alt-4: Stochastic Rounding + Per-Step Error Feedback
- Gist: Replace round-to-nearest FP8 casting with stochastic rounding and carry the quantization residual (bf16 − dequant(fp8)) across the 40 denoising steps, de-biasing the error that accumulates with sequence length and step count.
- Objective Evidence:
  - Production SR precedent: `tensorrt_llm/_torch/modules/mamba/replay_selective_state_update.py` `_stochastic_round_fp16x2()` (Philox + PTX `cvt.rs.f16x2`), config-gated via `mamba_ssm_stochastic_rounding`.
  - Per-step residual-carry precedent: `tensorrt_llm/_torch/visual_gen/cache/teacache.py` already caches `prev_residual` across timesteps; `cache/base.py` `CacheAccelerator.refresh()` is a ready hook.
  - REAL_MXFP8_RESULTS.md shows error scaling with sequence length (720p 0.27 > 480p 0.16) → accumulation is a real component.
- Why not primary: strong infra and orthogonal to the others, but needs a new E4M3 SR kernel (Mamba's is fp16-only) and only helps if the error is rounding *bias*; the probe suggests a mantissa *floor*, which SR alone cannot lift.

### Alt-5: Diffusion-Schedule-Aware Precision (bf16 on sensitive steps/stages)
- Gist: Run MXFP8 on coarse high-noise/early steps and bf16 on perceptually-critical low-noise/late steps, exploiting Wan2.2's existing two-transformer (high-noise/low-noise) structure.
- Objective Evidence:
  - Wan2.2 already dispatches between `transformer` and `transformer_2` by a boundary timestep (`pipeline_wan.py` forward; `split_wan22_inference_steps` in `cache/cache_dit_enablers.py`); the denoise loop in `pipeline.py` is step-indexed.
  - Backend selection is centralized in `attention_backend/utils.py` (`get_attention_backend`/`create_attention`) and per-model `attention_metadata_state` exists in `config.py` — two transformers can hold different backends.
- Why not primary: a pragmatic E2E speed/quality tradeoff, but it sidesteps MXFP8 on the hard steps rather than making MXFP8 itself more accurate — a different question than the one asked.

## Synthesis Notes

The primary (rotation) and Alt-2 (finer granularity) and Alt-4 (stochastic rounding) are complementary, not competing: they attack three orthogonal error sources — distribution shape / mantissa utilization (rotation), intra-block dynamic range (finer blocks), and per-step accumulation bias (SR) — and can stack. Alt-3 is the diagnostic control that explains *why* format-swapping and SmoothQuant fail (the E4M3 mantissa floor), reinforcing why the primary reshapes the distribution instead of rescaling it. If rotation alone cannot close the ~0.27 720p gap, the safety net is the mixed-precision pair: Alt-1 keeps the softmax-feeding QK^T in bf16, and Alt-5 keeps the perceptually-critical late steps in bf16 — both trade some of MXFP8's speedup for a guaranteed accuracy floor. A natural staged plan: (1) validate rotation first — cheapest, QK^T-invariant, reuses `rotate_activation`, measurable against the committed REAL_MXFP8 LPIPS; (2) add finer granularity + SR if rotation under-delivers and the cuDNN block-size constraint allows; (3) fall back to schedule-aware or per-matmul mixed precision to hit a hard quality target.
