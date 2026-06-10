# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""MXFP8 cuDNN SDPA backend for Visual Generation (Blackwell, sm100+).

Wraps cuDNN >= 9.21's `sdpa_mxfp8` Python frontend op with TransformerEngine's
`MXFP8Quantizer(optimize_for_gemm=True)` for the F8_128x4 swizzled scale layout
the kernel requires.

Falls back to `F.scaled_dot_product_attention` (bf16) for cross-attention
(seq_q != seq_kv), unsupported sm levels, missing dependencies, or any
graph-build failure. Self-attention (Wan's `attn1`) is the intended target.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ...attention_backend.interface import PredefinedAttentionMask
from .interface import AttentionBackend, AttentionTensorLayout

# Lazy-imported to keep import cost low and to fall back gracefully when
# the optional deps are missing on non-Blackwell systems.
_CUDNN = None
_TE_QUANTIZER = None
_TEX = None
_IMPORT_ERR: Optional[Exception] = None


def _try_import_deps() -> bool:
    """Lazy import TE + cuDNN. Order matters per cudnn-frontend tests."""
    global _CUDNN, _TE_QUANTIZER, _TEX, _IMPORT_ERR
    if _CUDNN is not None:
        return True
    if _IMPORT_ERR is not None:
        return False
    try:
        import cudnn  # imported AFTER TE on purpose
        import transformer_engine  # noqa: F401
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

        _TE_QUANTIZER = MXFP8Quantizer
        _TEX = tex
        _CUDNN = cudnn
        return True
    except Exception as e:  # pragma: no cover - environment dependent
        _IMPORT_ERR = e
        return False


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _mxfp8_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    if not _try_import_deps():
        return False
    cap = torch.cuda.get_device_capability()
    if cap[0] < 10:  # Blackwell or newer required
        return False
    cudnn = _CUDNN
    try:
        return cudnn.backend_version() >= 92100
    except Exception:
        return False


# --- TE quantization helpers ---------------------------------------------------

# Max rows the TE MXFP8 quantize CUDA kernel can launch in one call. At B=2/S=75600
# the pre-padded (B*H*s_pad, 128) tensor is ~6.05M rows and the kernel aborts with
# "CUDA Error: invalid argument" (a launch/indexing limit; B=1 ~3.03M rows works).
# We chunk along whole (B*H) groups -- chunk boundaries land on s_padded (a multiple
# of 128), so the per-row (rowwise Q/K) and per-32-row-block (columnwise V) MXFP8
# results are bit-identical to the unchunked call. Overridable for testing.
_MAX_QUANT_ROWS = int(os.environ.get("TRTLLM_VISUAL_GEN_MXFP8_MAX_QUANT_ROWS", str(3_000_000)))


def _te_mxfp8_quantize_2d(x2d: torch.Tensor, s_padded: int, rowwise: bool):
    """MXFP8-quantize a pre-padded (L*s_padded, d_padded) tensor.

    Chunks along whole (B*H) groups when the row count would crash the TE kernel.
    rowwise=True -> Q/K (quantized along D); rowwise=False -> V (columnwise, along S).
    """
    MXFP8Quantizer = _TE_QUANTIZER
    tex = _TEX

    def _quant(sub: torch.Tensor):
        quantizer = MXFP8Quantizer(
            fp8_dtype=tex.DType.kFloat8E4M3, rowwise=rowwise, columnwise=not rowwise
        )
        quantizer.optimize_for_gemm = True
        r = quantizer(sub)
        data = r._rowwise_data if rowwise else r._columnwise_data
        sf = r._rowwise_scale_inv if rowwise else r._columnwise_scale_inv
        return data, sf

    total_rows = x2d.shape[0]
    if total_rows <= _MAX_QUANT_ROWS:
        return _quant(x2d)
    chunk_L = max(1, _MAX_QUANT_ROWS // s_padded)
    chunk_rows = chunk_L * s_padded
    datas, sfs = [], []
    for start in range(0, total_rows, chunk_rows):
        d, s = _quant(x2d[start : start + chunk_rows].contiguous())
        datas.append(d)
        sfs.append(s)
    return torch.cat(datas, dim=0), torch.cat(sfs, dim=0)


# Optional SmoothQuant on the QK^T (Bmm1) matmul, env-gated for A/B testing.
# Per-head (H), per-channel (D) scale s computed from current activation amax:
#   s = amax_{B,S}(|Q|)^a / amax_{B,S}(|K|)^(1-a)  ;  Q' = Q/s, K' = K*s.
# This is EXACTLY invariant in QK^T (Q'.K'^T == Q.K^T per head), so only the FP8
# quantization error changes -- never V, never the softmax math. a selects the
# migration: a=0 normalizes K ("k"), a=0.5 balanced ("qk"), a=1 normalizes Q ("q").
_SQ_ENV = "TRTLLM_VISUAL_GEN_MXFP8_SMOOTHQUANT"
_SQ_ALPHA = {"k": 0.0, "qk": 0.5, "q": 1.0}
_SQ_LOGGED = False


def _smoothquant_qk(q: torch.Tensor, k: torch.Tensor):
    """Return (q', k') with per-head per-channel SmoothQuant applied iff env set."""
    global _SQ_LOGGED
    mode = os.environ.get(_SQ_ENV, "")
    if not mode:
        return q, k
    if mode in _SQ_ALPHA:
        alpha = _SQ_ALPHA[mode]
    else:
        try:
            alpha = float(mode)
        except ValueError:
            return q, k
    # q, k are (B, H, S, D). Per-head, per-channel amax over batch+seq -> (H, D).
    aq = q.float().abs().amax(dim=(0, 2)).clamp(min=1e-6)
    ak = k.float().abs().amax(dim=(0, 2)).clamp(min=1e-6)
    s = (aq.pow(alpha) / ak.pow(1.0 - alpha)).clamp(min=1e-4, max=1e4).to(q.dtype)
    s = s.unsqueeze(0).unsqueeze(2)  # (1, H, 1, D) broadcast over B, S
    if not _SQ_LOGGED:
        import sys

        print(
            f"[MXFP8] SmoothQuant ACTIVE: mode={mode} alpha={alpha} (per-head, QK^T-invariant)",
            file=sys.stderr,
            flush=True,
        )
        _SQ_LOGGED = True
    return q / s, k * s


# Optional Hadamard (Walsh) rotation on the D=head_dim axis, env-gated for A/B.
# A normalized Sylvester-Hadamard R (D x D, symmetric, orthogonal: R@R == I) applied
# identically to Q and K: Q' = Q@R, K' = K@R -> Q'.K'^T == Q.K^T exactly (invariant).
# Unlike SmoothQuant's diagonal rescale, this DENSE rotation spreads per-channel
# energy across D, lowering intra-32-block kurtosis so E4M3's 3 mantissa bits are
# better used. Requires D to be a power of two (Wan D=128 ok). Env TRTLLM_VISUAL_GEN_MXFP8_HADAMARD=1.
_HAD_ENV = "TRTLLM_VISUAL_GEN_MXFP8_HADAMARD"
_HAD_CACHE: dict = {}
_HAD_LOGGED = False


def _hadamard_matrix(n: int, device, dtype):
    cache_key = (n, str(device), dtype)
    R = _HAD_CACHE.get(cache_key)
    if R is None:
        m = torch.ones(1, 1, dtype=torch.float32)
        while m.shape[0] < n:
            m = torch.cat([torch.cat([m, m], dim=1), torch.cat([m, -m], dim=1)], dim=0)
        R = (m / (n**0.5)).to(device=device, dtype=dtype)  # normalized -> orthogonal
        _HAD_CACHE[cache_key] = R
    return R


def _hadamard_qk(q: torch.Tensor, k: torch.Tensor):
    """Return (q@R, k@R) with a shared orthogonal Hadamard R iff env set; else passthrough."""
    global _HAD_LOGGED
    if os.environ.get(_HAD_ENV, "0") != "1":
        return q, k
    D = q.shape[-1]
    if D & (D - 1) != 0:  # not a power of two -> cannot build Hadamard, skip safely
        return q, k
    R = _hadamard_matrix(D, q.device, q.dtype)
    if not _HAD_LOGGED:
        import sys

        print(f"[MXFP8] Hadamard rotation ACTIVE on D={D} (QK^T-invariant)", file=sys.stderr, flush=True)
        _HAD_LOGGED = True
    return q @ R, k @ R


def _quantize_q_or_k_along_d(x_bhsd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize Q or K (BHSD bf16/half) along the D axis (rowwise).

    Returns (fp8_data, swizzled_e8m0_scale). Padding is applied internally to
    match the F8_128x4 layout cuDNN requires; the FP8 data is then cropped back
    to the original (B,H,S,D).
    """
    cudnn = _CUDNN  # noqa: F841 (kept for potential future uses)
    MXFP8Quantizer = _TE_QUANTIZER
    tex = _TEX

    B, H, S, D = x_bhsd.shape
    L = B * H
    block = 32

    d_scale = _ceil_div(D, block)
    d_scale_padded = _ceil_div(d_scale, 4) * 4
    d_padded = d_scale_padded * block
    s_scale = _ceil_div(S, block)
    s_scale_padded = _ceil_div(s_scale, 4) * 4
    s_padded = s_scale_padded * block

    x = x_bhsd.reshape(L, S, D)
    pad_d = d_padded - D
    pad_s = s_padded - S
    if pad_s or pad_d:
        x = torch.nn.functional.pad(x, (0, pad_d, 0, pad_s))
    x2d = x.reshape(L * s_padded, d_padded)

    data, sf = _te_mxfp8_quantize_2d(x2d, s_padded, rowwise=True)  # sf: raw uint8 F8_128x4

    fp8 = (
        data.reshape(L, s_padded, d_padded)[:, :S, :D]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(B, H, S, D)
    )
    return fp8, sf


def _quantize_v_along_s(x_bhsd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize V (BHSD bf16/half) along the S axis (columnwise)."""
    MXFP8Quantizer = _TE_QUANTIZER
    tex = _TEX

    B, H, S, D = x_bhsd.shape
    L = B * H
    block = 32

    d_scale = _ceil_div(D, block)
    d_scale_padded = _ceil_div(d_scale, 4) * 4
    d_padded = d_scale_padded * block
    s_scale = _ceil_div(S, block)
    s_scale_padded = _ceil_div(s_scale, 4) * 4
    s_padded = s_scale_padded * block

    x = x_bhsd.reshape(L, S, D)
    pad_d = d_padded - D
    pad_s = s_padded - S
    if pad_s or pad_d:
        x = torch.nn.functional.pad(x, (0, pad_d, 0, pad_s))
    x2d = x.reshape(L * s_padded, d_padded)

    data, sf = _te_mxfp8_quantize_2d(x2d, s_padded, rowwise=False)

    fp8 = (
        data.reshape(L, s_padded, d_padded)[:, :S, :D]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(B, H, S, D)
    )
    return fp8, sf


# --- cuDNN graph construction --------------------------------------------------

# UIDs for the variant_pack
_UID_Q = 0
_UID_K = 1
_UID_V = 2
_UID_SFQ = 5
_UID_SFK = 6
_UID_SFV = 7
_UID_O = 3
_UID_STATS = 4
_UID_AMAX_O = 12


def _build_sdpa_mxfp8_graph(B, H, S, D, attn_scale, handle):
    cudnn = _CUDNN
    s_pad = _ceil_div(S, 128) * 128
    d_scale_padded = _ceil_div(_ceil_div(D, 32), 4) * 4
    s_scale_padded = _ceil_div(_ceil_div(S, 32), 4) * 4
    d_padded = _ceil_div(D, 128) * 128

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )

    def fp8_in(uid):
        return g.tensor(
            uid=uid,
            dim=(B, H, S, D),
            stride=(H * S * D, S * D, D, 1),
            data_type=cudnn.data_type.FP8_E4M3,
        )

    q = fp8_in(_UID_Q)
    k = fp8_in(_UID_K)
    v = fp8_in(_UID_V)

    sf_q = g.tensor(
        uid=_UID_SFQ,
        dim=(B, H, s_pad, d_scale_padded),
        stride=(H * s_pad * d_scale_padded, s_pad * d_scale_padded, d_scale_padded, 1),
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    sf_k = g.tensor(
        uid=_UID_SFK,
        dim=(B, H, s_pad, d_scale_padded),
        stride=(H * s_pad * d_scale_padded, s_pad * d_scale_padded, d_scale_padded, 1),
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    sf_v = g.tensor(
        uid=_UID_SFV,
        dim=(B, H, s_scale_padded, d_padded),
        stride=(H * s_scale_padded * d_padded, s_scale_padded * d_padded, d_padded, 1),
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )

    o, _stats, amax_o = g.sdpa_mxfp8(
        q=q,
        k=k,
        v=v,
        descale_q=sf_q,
        descale_k=sf_k,
        descale_v=sf_v,
        attn_scale=attn_scale,
        generate_stats=False,
    )
    o.set_uid(_UID_O).set_output(True).set_dim((B, H, S, D)).set_stride(
        (H * S * D, S * D, D, 1)
    ).set_data_type(cudnn.data_type.BFLOAT16)
    if amax_o is not None:
        amax_o.set_uid(_UID_AMAX_O).set_output(True).set_dim((1, 1, 1, 1)).set_stride(
            (1, 1, 1, 1)
        ).set_data_type(cudnn.data_type.FLOAT)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    g.check_support()
    g.build_plans()
    return g


class MXFP8CudnnAttention(AttentionBackend):
    """Self-attention via cuDNN MXFP8 SDPA. Cross-attention falls back to bf16 SDPA."""

    _force_fallback_env = "TRTLLM_VISUAL_GEN_DISABLE_MXFP8_CUDNN"
    # Per-instance counters so callers (tests, eval drivers) can verify the
    # mxfp8 path actually fired vs silently fell back to bf16 SDPA.
    # `mxfp8_calls` increments ONLY when sdpa_mxfp8 is actually invoked;
    # `fallback_calls` increments on every dispatch that took the bf16 path.
    #
    # Setting the env var below to a writable path makes every backend
    # instance append its (layer_idx, mxfp8_calls, fallback_calls) on
    # destruction. Useful to confirm the mxfp8 path fired across all layers
    # during a real Wan/diffusion run from the worker subprocess.
    _trace_env = "TRTLLM_VISUAL_GEN_MXFP8_TRACE"
    # When set, every dispatch (mxfp8 or fallback) appends one CSV-ish line with
    # wall-clock timestamp, layer_idx, q-shape, and which path was taken.
    # Use this to verify that *main-run* steps all took the mxfp8 path
    # (not just aggregate-across-warmup-and-main counts).
    _per_call_trace_env = "TRTLLM_VISUAL_GEN_MXFP8_PER_CALL_TRACE"

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype
        self.scale = 1.0 / math.sqrt(head_dim)
        self._preferred_layout = AttentionTensorLayout.HND

        # Lazy state
        self._cudnn_handle = None
        self._graph_cache: dict[Tuple[int, int, int, int], object] = {}
        self._workspace: Optional[torch.Tensor] = None
        self._enabled = os.environ.get(self._force_fallback_env, "0") != "1" and _mxfp8_supported()
        # Counters (see class docstring fields above).
        self.mxfp8_calls: int = 0
        self.fallback_calls: int = 0

    def __del__(self):
        # Append counters to the trace file when env-var is set. Best-effort:
        # don't raise from a destructor.
        try:
            path = os.environ.get(self._trace_env)
            if path:
                with open(path, "a") as f:
                    f.write(
                        f"layer_idx={getattr(self, 'layer_idx', '?')} "
                        f"H={getattr(self, 'num_heads', '?')} "
                        f"D={getattr(self, 'head_dim', '?')} "
                        f"mxfp8_calls={getattr(self, 'mxfp8_calls', 0)} "
                        f"fallback_calls={getattr(self, 'fallback_calls', 0)}\n"
                    )
        except Exception:
            pass

    # --- public API ---

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> torch.Tensor:
        is_causal = attention_mask == PredefinedAttentionMask.CAUSAL
        same_seq_len = k.shape[2] == q.shape[2] and v.shape[2] == q.shape[2]
        # Cross-attention (q/kv seq differ), causal, or non-fp16/bf16 cannot use the
        # self-attention MXFP8 graph and run bf16 -- this is architectural and is
        # ALSO bf16 in the VANILLA backend, so it cancels in any MXFP8-vs-VANILLA
        # comparison. It is an explicit, counted path, NOT a silent failure mask.
        if not self._enabled or is_causal or not same_seq_len or q.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            self.fallback_calls += 1
            self._per_call_log(q, "cross_or_unsupported_bf16")
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=self.scale)
        # Self-attention: NO silent fallback. Real MXFP8 runs or this raises loudly,
        # so a generated "MXFP8" video can never be secretly bf16. (Deleted the old
        # `except -> bf16` mask that hid the B=2/S=75600 TE-quantizer crash at 720p.)
        out = self._mxfp8_forward(q, k, v)
        self.mxfp8_calls += 1
        self._per_call_log(q, "mxfp8")
        return out

    def _per_call_log(self, q, path):
        path_var = os.environ.get(self._per_call_trace_env)
        if not path_var:
            return
        try:
            import time as _time

            B, H, S, D = q.shape
            with open(path_var, "a") as f:
                f.write(
                    f"{_time.time():.6f} layer_idx={self.layer_idx} "
                    f"path={path} B={B} H={H} S={S} D={D} dtype={q.dtype}\n"
                )
        except Exception:
            pass

    # --- internals ---

    @torch.compiler.disable
    def _mxfp8_forward(self, q, k, v):
        B, H, S, D = q.shape
        cudnn = _CUDNN
        if self._cudnn_handle is None:
            self._cudnn_handle = cudnn.create_handle()

        key = (B, H, S, D)
        graph = self._graph_cache.get(key)
        if graph is None:
            graph = _build_sdpa_mxfp8_graph(B, H, S, D, self.scale, self._cudnn_handle)
            self._graph_cache[key] = graph

        q, k = _smoothquant_qk(q, k)  # optional, env-gated; QK^T-invariant; V untouched
        q, k = _hadamard_qk(q, k)  # optional, env-gated; QK^T-invariant; V untouched
        qf, sfq = _quantize_q_or_k_along_d(q)
        kf, sfk = _quantize_q_or_k_along_d(k)
        vf, sfv = _quantize_v_along_s(v)

        out = torch.empty(B, H, S, D, dtype=torch.bfloat16, device=q.device)
        amax_o = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device=q.device)

        ws_bytes = graph.get_workspace_size()
        if self._workspace is None or self._workspace.numel() < ws_bytes:
            self._workspace = torch.empty(ws_bytes, dtype=torch.uint8, device=q.device)

        var_pack = {
            _UID_Q: qf,
            _UID_K: kf,
            _UID_V: vf,
            _UID_SFQ: sfq,
            _UID_SFK: sfk,
            _UID_SFV: sfv,
            _UID_O: out,
            _UID_AMAX_O: amax_o,
        }
        graph.execute(var_pack, self._workspace, handle=self._cudnn_handle)
        return out.to(q.dtype)
