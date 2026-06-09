# Does SmoothQuant (K / Q/K) improve MXFP8 attention accuracy? — No (synthetic probe)

**Question (user):** would SmoothQuant on K, or on Q/K, improve MXFP8 attention accuracy?

**Answer: essentially no.** On a synthetic B200 probe (umbriel-b200-072,
`smoothquant_mxfp8_probe.py`, cuDNN 9.22 `sdpa_mxfp8`, B=1/S=14040, mean over 3 seeds),
per-channel SmoothQuant on the QK^T matmul never bought more than ~2% and usually hurt.

`s_d = amax_d|Q|^α / amax_d|K|^(1-α)`, `Q'=Q/s_d`, `K'=K·s_d` (exact in fp32;
α→0 = "smooth K", α=0.5 = "smooth Q/K", α→1 = "smooth Q"):

| variant | rel-err vs bf16, **no outliers** | Δ | rel-err, **shared Q/K outliers (30×/8×)** | Δ |
|---|---|---|---|---|
| baseline (no smooth) | 0.3201 | — | 1.5180 | — |
| smooth K (α=0.0) | 0.3591 | +12.2% | 1.4842 | **−2.2%** |
| α=0.25 | 0.3751 | +17.2% | 1.5191 | +0.1% |
| Q/K (α=0.5) | 0.3206 | +0.2% | 1.5176 | −0.0% |
| α=0.75 | 0.3727 | +16.5% | 1.5058 | −0.8% |
| smooth Q (α=1.0) | 0.3620 | +13.1% | 1.5298 | +0.8% |

**Why SmoothQuant can't help here (mechanism, empirically supported):**

1. **MXFP8 already block-scales every 32 elements along D.** SmoothQuant exists to
   fix the dynamic-range/outlier problem of *coarse* (per-tensor/per-token) int8
   quant. MX's per-32-block E8M0 scale already absorbs almost all of that, so
   SmoothQuant's per-channel rescale is largely redundant — and on clean data it
   *manufactures* per-channel disparity that increases intra-block range → **worse**.
2. **The dominant MXFP8 error is E4M3 mantissa precision (3 mantissa bits ≈ 6%
   per-element), not dynamic range.** SmoothQuant moves magnitude around; it cannot
   add mantissa bits. This is exactly the colleague's point that softmax (a
   normalized exponential) is mantissa-sensitive, so MX's extra exponent range
   doesn't buy attention accuracy. The ~0.32 baseline rel-err is a mantissa floor.
3. **With outliers the error is catastrophic (~1.5) regardless** — the outlier
   channels dominate the QK^T dot product and, once in E4M3, are mantissa-starved.
   SmoothQuant only shuffles the outlier between Q and K; the product still needs
   those large values represented in 3 mantissa bits.

**More promising directions than SmoothQuant** (not yet tested):
- **Hadamard / rotation (QuaRot/SpinQuant-style) on D**, shared between Q and K so
  it cancels in QK^T like SmoothQuant does. Rotations reduce kurtosis and spread
  energy across channels, improving *mantissa utilization within each MX block* —
  which is the actual bottleneck, unlike SmoothQuant's diagonal scaling.
- **Mixed precision**: keep the mantissa-sensitive QK^T in bf16 (or MXFP8 only for
  the AV matmul / V), since the softmax amplifies QK^T error most.
- Neither NVFP4 nor MXFP4 will help — both have *fewer* mantissa bits than E4M3.

**Caveat:** synthetic Q/K (gaussian + injected shared-channel outliers). A real
verdict wants Q/K captured from a Wan forward (a forward hook on `attn1`), but the
mechanism (range-vs-mantissa) is format-level and unlikely to change the conclusion.

Raw: `smoothquant_probe.log`. Script: `smoothquant_mxfp8_probe.py`.
