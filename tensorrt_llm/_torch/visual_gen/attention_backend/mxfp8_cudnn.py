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

    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    quantizer.optimize_for_gemm = True
    res = quantizer(x2d)

    fp8 = (
        res._rowwise_data.reshape(L, s_padded, d_padded)[:, :S, :D]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(B, H, S, D)
    )
    sf = res._rowwise_scale_inv  # raw uint8 in F8_128x4 layout
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

    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=False, columnwise=True)
    quantizer.optimize_for_gemm = True
    res = quantizer(x2d)

    fp8 = (
        res._columnwise_data.reshape(L, s_padded, d_padded)[:, :S, :D]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(B, H, S, D)
    )
    sf = res._columnwise_scale_inv
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
        if (
            not self._enabled
            or is_causal  # MXFP8 path here is built without causal mask; punt
            or not same_seq_len  # cross-attention -> bf16 fallback
            or q.dtype not in (torch.bfloat16, torch.float16)
        ):
            self.fallback_calls += 1
            self._per_call_log(q, "fallback_dispatch")
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=self.scale)
        try:
            out = self._mxfp8_forward(q, k, v)
            self.mxfp8_calls += 1
            self._per_call_log(q, "mxfp8")
            return out
        except Exception as e:
            # On any failure (graph build, exec), fall back transparently.
            self.fallback_calls += 1
            self._per_call_log(q, f"fallback_exception:{type(e).__name__}")
            return F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scale)

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
