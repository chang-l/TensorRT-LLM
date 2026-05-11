# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SageAttention (PR #13570) integrated into Wan visual-gen.

Mirrors test_mxfp8_cudnn_attention.py: verifies numerical agreement vs
torch's bf16 SDPA at Wan-style shapes for both Sage granularities
(1, 4, 1) and (1, 16, 1) with qk_int8=True, plus the path-actually-fired
counter assertion.
"""

import math

import pytest
import torch
import torch.nn.functional as F


def _capability_ok():
    if not torch.cuda.is_available():
        return False
    return True  # Sage works on Ampere+; B200 is fine


@pytest.fixture(scope="module")
def sage_cls_and_meta():
    if not _capability_ok():
        pytest.skip("CUDA not available")
    try:
        from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention
        from tensorrt_llm._torch.visual_gen.config import (
            SageAttentionConfig,
            create_attention_metadata_state,
        )
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Sage import failed: {e}")
    return TrtllmAttention, SageAttentionConfig, create_attention_metadata_state


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


# Wan-A14B-style shapes. H=40, D=128 matches the configured model.
WAN_SHAPES = [
    pytest.param(1, 40, 4096, 128, id="wan_S4096"),
    pytest.param(1, 40, 8192, 128, id="wan_S8192"),
    # S=75600 (720x1280/81f) is omitted from the fast-path test because the
    # TRTLLM backend allocates a much larger metadata workspace; covered
    # implicitly by the e2e Wan run.
]

SAGE_GRANULARITIES = [
    pytest.param((1, 4, 1), id="blk_1_4_1"),
    pytest.param((1, 16, 1), id="blk_1_16_1"),
]


def _make_sage_attn(Cls, ConfigCls, MetaFn, num_heads, head_dim, granularity):
    nq, nk, nv = granularity
    cfg = ConfigCls(
        num_elts_per_blk_q=nq,
        num_elts_per_blk_k=nk,
        num_elts_per_blk_v=nv,
        qk_int8=True,
    )
    meta_state = MetaFn()
    return Cls(
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_heads,
        dtype=torch.bfloat16,
        attention_metadata_state=meta_state,
        sage_attention_config=cfg,
    )


@pytest.mark.skipif(not _capability_ok(), reason="needs CUDA")
@pytest.mark.parametrize("granularity", SAGE_GRANULARITIES)
@pytest.mark.parametrize("B,H,S,D", WAN_SHAPES)
def test_sage_numerics_vs_bf16(sage_cls_and_meta, B, H, S, D, granularity):
    """Cosine similarity vs bf16 SDPA must be > 0.95 at typical shapes.

    Note: TRTLLM Sage backend's `forward` expects NHD layout [B, S, H, D],
    not HND. We reshape accordingly.
    """
    Cls, ConfigCls, MetaFn = sage_cls_and_meta
    attn = _make_sage_attn(Cls, ConfigCls, MetaFn, H, D, granularity)

    # Per-shape distinct seed so each parametrized case exercises different
    # samples (otherwise S=4096 and S=8192 share the same RNG state and the
    # tests provide redundant signal).
    q_hnd, k_hnd, v_hnd = _qkv(B, H, S, D, seed=S)  # [B, H, S, D]
    ref = _bf16_ref(q_hnd, k_hnd, v_hnd, scale=1.0 / math.sqrt(D))

    # Sage backend wants NHD: [B, S, H, D]
    q = q_hnd.transpose(1, 2).contiguous()
    k = k_hnd.transpose(1, 2).contiguous()
    v = v_hnd.transpose(1, 2).contiguous()

    pre = attn.sage_calls
    out_flat = attn.forward(q=q, k=k, v=v, batch_size=B, seq_len=S)
    post = attn.sage_calls
    assert post == pre + 1, (
        f"Sage path did not fire (sage_calls={pre}->{post}); fallback_calls={attn.fallback_calls}"
    )

    # Output is [B, S, H*D]; reshape to [B, H, S, D] to compare with ref.
    out = out_flat.view(B, S, H, D).transpose(1, 2).contiguous()
    s = _stats(ref, out)
    # Tightened per supervisor review: observed cos ≈ 0.996 / rel_rms ≈ 0.048,
    # so cos > 0.99 and rel_rms < 0.10 leave only ~2x slack. A real regression
    # (4x block-size mistake, off-by-one tile boundary) will trip this.
    assert s["cosine"] > 0.99, f"cosine too low: {s} ({granularity})"
    assert s["rel_rms"] < 0.10, f"rel-rms too high: {s} ({granularity})"
    print(f"[sage {granularity} S={S}] {s}")


@pytest.mark.skipif(not _capability_ok(), reason="needs CUDA")
@pytest.mark.parametrize("granularity", SAGE_GRANULARITIES)
def test_sage_repeated_calls_keep_counter_growing(sage_cls_and_meta, granularity):
    """Sanity check that counters strictly increment across repeated calls."""
    Cls, ConfigCls, MetaFn = sage_cls_and_meta
    attn = _make_sage_attn(
        Cls, ConfigCls, MetaFn, num_heads=8, head_dim=128, granularity=granularity
    )
    B, H, S, D = 1, 8, 512, 128
    q_hnd, k_hnd, v_hnd = _qkv(B, H, S, D)
    q = q_hnd.transpose(1, 2).contiguous()
    k = k_hnd.transpose(1, 2).contiguous()
    v = v_hnd.transpose(1, 2).contiguous()
    n_iters = 3
    for _ in range(n_iters):
        attn.forward(q=q, k=k, v=v, batch_size=B, seq_len=S)
    assert attn.sage_calls == n_iters, (
        f"expected {n_iters} sage_calls, got {attn.sage_calls} "
        f"(fallback_calls={attn.fallback_calls})"
    )
