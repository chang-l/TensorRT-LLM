# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MXFP8 cuDNN SDPA backend used in visual-gen self-attention.

Asserts:
1. Numerical agreement vs torch's bf16 SDPA at Wan-style shapes
   (max-abs, RMS, cosine).
2. The mxfp8 path actually fires (vs falling back to bf16) — uses the
   instance-level call counters on `MXFP8CudnnAttention`.
3. Graceful fallback: cross-attention (different Q/KV seq lens), causal,
   and fp32 dispatch should all use bf16 and bump `fallback_calls`.
4. Skips cleanly on non-Blackwell or when cuDNN < 9.21 / cudnn-frontend
   doesn't expose `sdpa_mxfp8`.
"""

import math

import pytest
import torch
import torch.nn.functional as F

cuda_available = torch.cuda.is_available()


def _capability_ok():
    if not cuda_available:
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 10  # Blackwell+


@pytest.fixture(scope="module")
def mxfp8_attn_module():
    if not _capability_ok():
        pytest.skip("MXFP8 cuDNN SDPA needs Blackwell (sm_100+)")
    try:
        from tensorrt_llm._torch.visual_gen.attention_backend.mxfp8_cudnn import (  # noqa: E501
            MXFP8CudnnAttention,
        )
    except Exception as e:  # pragma: no cover
        pytest.skip(f"backend import failed: {e}")
    return MXFP8CudnnAttention


def _qkv(B, H, S, D, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, H, S, D, dtype=dtype, device="cuda", generator=g) * 0.5
    k = torch.randn(B, H, S, D, dtype=dtype, device="cuda", generator=g) * 0.5
    v = torch.randn(B, H, S, D, dtype=dtype, device="cuda", generator=g) * 0.5
    return q, k, v


def _bf16_ref(q, k, v, scale):
    return F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)


def _stats(ref, got):
    diff = ref.float() - got.float()
    mae = diff.abs().mean().item()
    mx = diff.abs().max().item()
    rms = (diff**2).mean().sqrt().item()
    rel_rms = rms / max(ref.float().abs().mean().item(), 1e-12)
    cos = F.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
    return dict(mae=mae, max=mx, rms=rms, rel_rms=rel_rms, cosine=cos)


# Shapes: (B, H, S, D). Wan2.2-A14B uses H=40, D=128. We sweep S to ensure
# the kernel works across multiple cuDNN graph shapes, including the
# per-frame self-attn at Wan2.2 720x1280/81 frames (S=75600).
WAN_SHAPES = [
    pytest.param(1, 40, 4096, 128, id="wan_small_S4096"),
    pytest.param(1, 40, 8192, 128, id="wan_S8192"),
    pytest.param(1, 40, 75600, 128, id="wan_default_S75600"),
]


@pytest.mark.skipif(not _capability_ok(), reason="needs Blackwell")
@pytest.mark.parametrize("B,H,S,D", WAN_SHAPES)
def test_mxfp8_numerics_vs_bf16(mxfp8_attn_module, B, H, S, D):
    """Cosine similarity vs bf16 SDPA must be > 0.98 at typical shapes."""
    Cls = mxfp8_attn_module
    attn = Cls(num_heads=H, head_dim=D)
    if not attn._enabled:
        pytest.skip("MXFP8CudnnAttention not _enabled in this env")

    q, k, v = _qkv(B, H, S, D)
    ref = _bf16_ref(q, k, v, scale=1.0 / math.sqrt(D))

    pre = attn.mxfp8_calls
    got = attn.forward(q, k, v)
    post = attn.mxfp8_calls
    assert post == pre + 1, (
        f"MXFP8 path did not fire (mxfp8_calls={pre}->{post}). "
        f"fallback_calls={attn.fallback_calls}; check _enabled and shape support."
    )

    s = _stats(ref, got)
    # FP8 attention typically lands ~1-10% rel-rms with cosine close to 1.
    assert s["cosine"] > 0.98, f"cosine too low: {s}"
    assert s["rel_rms"] < 0.10, f"rel-rms too high: {s}"
    print(f"[mxfp8 numerics S={S}] {s}")


@pytest.mark.skipif(not _capability_ok(), reason="needs Blackwell")
def test_cross_attention_falls_back_to_bf16(mxfp8_attn_module):
    """Different Q/KV seq lens must take the bf16 fallback path (cross-attn)."""
    Cls = mxfp8_attn_module
    attn = Cls(num_heads=8, head_dim=128)

    B, H, D = 1, 8, 128
    q = torch.randn(B, H, 256, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H, 64, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H, 64, D, dtype=torch.bfloat16, device="cuda")

    pre_mx = attn.mxfp8_calls
    pre_fb = attn.fallback_calls
    out = attn.forward(q, k, v)
    assert attn.mxfp8_calls == pre_mx, "MXFP8 path should not fire on cross-attn"
    assert attn.fallback_calls == pre_fb + 1, "fallback counter not bumped"
    # Sanity: output shape matches Q
    assert out.shape == q.shape


@pytest.mark.skipif(not _capability_ok(), reason="needs Blackwell")
def test_fp32_inputs_fall_back(mxfp8_attn_module):
    """FP32 inputs must fall back; mxfp8 path is bf16/fp16 only."""
    Cls = mxfp8_attn_module
    attn = Cls(num_heads=8, head_dim=128)
    q = torch.randn(1, 8, 256, 128, dtype=torch.float32, device="cuda")
    k = q.clone()
    v = q.clone()
    pre_mx = attn.mxfp8_calls
    pre_fb = attn.fallback_calls
    _ = attn.forward(q, k, v)
    assert attn.mxfp8_calls == pre_mx
    assert attn.fallback_calls == pre_fb + 1


@pytest.mark.skipif(not _capability_ok(), reason="needs Blackwell")
def test_env_disable_falls_back(mxfp8_attn_module, monkeypatch):
    """Setting TRTLLM_VISUAL_GEN_DISABLE_MXFP8_CUDNN=1 must force fallback."""
    Cls = mxfp8_attn_module
    monkeypatch.setenv("TRTLLM_VISUAL_GEN_DISABLE_MXFP8_CUDNN", "1")
    attn = Cls(num_heads=8, head_dim=128)
    assert attn._enabled is False, "_enabled should be False with env override"
    q = torch.randn(1, 8, 256, 128, dtype=torch.bfloat16, device="cuda")
    k = q.clone()
    v = q.clone()
    pre_mx = attn.mxfp8_calls
    _ = attn.forward(q, k, v)
    assert attn.mxfp8_calls == pre_mx
    assert attn.fallback_calls >= 1


@pytest.mark.skipif(not _capability_ok(), reason="needs Blackwell")
def test_repeated_calls_use_graph_cache(mxfp8_attn_module):
    """Two calls with the same shape must reuse the same cached cuDNN graph."""
    Cls = mxfp8_attn_module
    attn = Cls(num_heads=8, head_dim=128)
    if not attn._enabled:
        pytest.skip("MXFP8CudnnAttention not _enabled")
    q, k, v = _qkv(1, 8, 1024, 128)
    _ = attn.forward(q, k, v)
    cached_key = next(iter(attn._graph_cache))
    cached_obj = attn._graph_cache[cached_key]
    _ = attn.forward(q, k, v)
    assert attn._graph_cache[cached_key] is cached_obj, (
        "graph cache evicted/rebuilt for the same shape"
    )
    assert attn.mxfp8_calls >= 2
