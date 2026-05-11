# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 cuDNN SDPA microbench for B200.

Validates and benchmarks cudnn-frontend `sdpa_mxfp8` (cuDNN >= 9.21, Blackwell)
against torch bf16 SDPA at Wan2.2-style attention shapes.

Pre-quantization is done via TransformerEngine's MXFP8Quantizer with
optimize_for_gemm=True so scales are emitted in F8_128x4 swizzled layout
that cuDNN's sdpa_mxfp8 expects.
"""

import argparse
import math

import cudnn  # noqa: E402
import torch
import torch.nn.functional as F

# Import TE BEFORE cudnn (per cudnn-frontend/test/python/sdpa/mxfp8.py guidance)
import transformer_engine  # noqa: F401
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=40)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def ceil_div(a, b):
    return (a + b - 1) // b


def make_qkv(B, H, S, D, dtype=torch.bfloat16, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, H, S, D, dtype=dtype, device=device, generator=g) * 0.5
    k = torch.randn(B, H, S, D, dtype=dtype, device=device, generator=g) * 0.5
    v = torch.randn(B, H, S, D, dtype=dtype, device=device, generator=g) * 0.5
    return q, k, v


def reference_bf16_sdpa(q, k, v, scale):
    return F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)


def time_op(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def report_diff(label, ref, got):
    mae = (ref.float() - got.float()).abs().mean().item()
    mx = (ref.float() - got.float()).abs().max().item()
    rms = ((ref.float() - got.float()) ** 2).mean().sqrt().item()
    rel = rms / max(ref.float().abs().mean().item(), 1e-12)
    cos = F.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
    print(f"[{label}] mae={mae:.4e} max={mx:.4e} rms={rms:.4e} rel_rms={rel:.4e} cosine={cos:.6f}")


# ---------- TE MXFP8 quantization (returns swizzled E8M0 scales) ----------


def te_quantize_mxfp8(x_bhsd, fp8_dtype=torch.float8_e4m3fn, block_size=32):
    """Quantize a (B,H,S,D) bf16 tensor.

    Returns:
      fp8_d: rowwise (along D) FP8, shape (B,H,S,D)
      sf_d_swizzle: swizzled E8M0 scale tensor, F8_128x4 layout (raw uint8 view)
      fp8_s: columnwise (along S) FP8, shape (B,H,S,D), used for V
      sf_s_swizzle: swizzled E8M0 scale for s-axis quantization
    """
    B, H, S, D = x_bhsd.shape
    L = B * H

    te_dtype = tex.DType.kFloat8E4M3 if fp8_dtype == torch.float8_e4m3fn else tex.DType.kFloat8E5M2

    d_scale = ceil_div(D, block_size)
    d_scale_padded = ceil_div(d_scale, 4) * 4
    d_padded = d_scale_padded * block_size
    s_scale = ceil_div(S, block_size)
    s_scale_padded = ceil_div(s_scale, 4) * 4
    s_padded = s_scale_padded * block_size

    x = x_bhsd.float().reshape(L, S, D)
    pad_d = d_padded - D
    pad_s = s_padded - S
    if pad_s > 0 or pad_d > 0:
        x = torch.nn.functional.pad(x, (0, pad_d, 0, pad_s))
    x2d = x.reshape(L * s_padded, d_padded)

    quantizer = MXFP8Quantizer(fp8_dtype=te_dtype, rowwise=True, columnwise=True)
    quantizer.optimize_for_gemm = True
    res = quantizer(x2d)

    fp8_d_flat = res._rowwise_data
    fp8_d = fp8_d_flat.reshape(L, s_padded, d_padded)[:, :S, :D].contiguous()
    fp8_d = fp8_d.view(fp8_dtype).reshape(B, H, S, D)
    sf_d_swizzle = res._rowwise_scale_inv  # uint8 raw, hardware layout

    fp8_s_flat = res._columnwise_data
    fp8_s = fp8_s_flat.reshape(L, s_padded, d_padded)[:, :S, :D].contiguous()
    fp8_s = fp8_s.view(fp8_dtype).reshape(B, H, S, D)
    sf_s_swizzle = res._columnwise_scale_inv

    return fp8_d, sf_d_swizzle, fp8_s, sf_s_swizzle


# ---------- cuDNN sdpa_mxfp8 graph builder ----------

UID_Q = 0
UID_K = 1
UID_V = 2
UID_SFQ = 5
UID_SFK = 6
UID_SFV = 7
UID_O = 3
UID_STATS = 4
UID_AMAX_O = 12


def build_mxfp8_sdpa_graph(B, H, S, D, attn_scale, handle):
    s_pad = ceil_div(S, 128) * 128
    d_scale_padded = ceil_div(ceil_div(D, 32), 4) * 4
    s_scale_padded = ceil_div(ceil_div(S, 32), 4) * 4
    d_padded = ceil_div(D, 128) * 128

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

    q = fp8_in(UID_Q)
    k = fp8_in(UID_K)
    v = fp8_in(UID_V)

    sf_q = g.tensor(
        uid=UID_SFQ,
        dim=(B, H, s_pad, d_scale_padded),
        stride=(H * s_pad * d_scale_padded, s_pad * d_scale_padded, d_scale_padded, 1),
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    sf_k = g.tensor(
        uid=UID_SFK,
        dim=(B, H, s_pad, d_scale_padded),
        stride=(H * s_pad * d_scale_padded, s_pad * d_scale_padded, d_scale_padded, 1),
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    sf_v = g.tensor(
        uid=UID_SFV,
        dim=(B, H, s_scale_padded, d_padded),
        stride=(H * s_scale_padded * d_padded, s_scale_padded * d_padded, d_padded, 1),
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )

    o, stats, amax_o = g.sdpa_mxfp8(
        q=q,
        k=k,
        v=v,
        descale_q=sf_q,
        descale_k=sf_k,
        descale_v=sf_v,
        attn_scale=attn_scale,
        generate_stats=True,
    )
    o.set_uid(UID_O).set_output(True).set_dim((B, H, S, D)).set_stride(
        (H * S * D, S * D, D, 1)
    ).set_data_type(cudnn.data_type.BFLOAT16)
    stats.set_uid(UID_STATS).set_output(True).set_dim((B, H, S, 1)).set_stride(
        (H * S, S, 1, 1)
    ).set_data_type(cudnn.data_type.FLOAT)
    amax_o.set_uid(UID_AMAX_O).set_output(True).set_dim((1, 1, 1, 1)).set_stride(
        (1, 1, 1, 1)
    ).set_data_type(cudnn.data_type.FLOAT)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    g.check_support()
    g.build_plans()
    return g


def main():
    args = parse_args()
    print(f"shape B={args.batch} H={args.heads} S={args.seq} D={args.dim} bs={args.block_size}")
    torch.cuda.set_device(0)

    q_bf, k_bf, v_bf = make_qkv(args.batch, args.heads, args.seq, args.dim, seed=args.seed)
    attn_scale = 1.0 / math.sqrt(args.dim)

    # bf16 reference
    o_ref = reference_bf16_sdpa(q_bf, k_bf, v_bf, attn_scale)

    def bf_fn():
        return reference_bf16_sdpa(q_bf, k_bf, v_bf, attn_scale)

    bf_t = time_op(bf_fn, args.warmup, args.iters)
    print(f"[bf16 SDPA]   median={sorted(bf_t)[len(bf_t) // 2]:.3f} ms  min={min(bf_t):.3f}")

    # MXFP8 quantize via TE
    qf_d, sfq, _, _ = te_quantize_mxfp8(q_bf)
    kf_d, sfk, _, _ = te_quantize_mxfp8(k_bf)
    _, _, vf_s, sfv = te_quantize_mxfp8(v_bf)

    # Build cudnn sdpa_mxfp8 graph
    handle = cudnn.create_handle()
    try:
        g = build_mxfp8_sdpa_graph(args.batch, args.heads, args.seq, args.dim, attn_scale, handle)
    except Exception as e:
        print(f"[mxfp8] graph build FAILED: {type(e).__name__}: {e}")
        return

    o_buf = torch.empty(
        args.batch, args.heads, args.seq, args.dim, dtype=torch.bfloat16, device="cuda"
    )
    stats_buf = torch.empty(args.batch, args.heads, args.seq, 1, dtype=torch.float32, device="cuda")
    amax_o_buf = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device="cuda")
    workspace = torch.empty(g.get_workspace_size(), dtype=torch.uint8, device="cuda")

    var_pack = {
        UID_Q: qf_d,
        UID_K: kf_d,
        UID_V: vf_s,
        UID_SFQ: sfq,
        UID_SFK: sfk,
        UID_SFV: sfv,
        UID_O: o_buf,
        UID_STATS: stats_buf,
        UID_AMAX_O: amax_o_buf,
    }

    def fn():
        g.execute(var_pack, workspace, handle=handle)

    fn()
    torch.cuda.synchronize()
    report_diff("mxfp8", o_ref, o_buf)

    ts = time_op(fn, args.warmup, args.iters)
    med = sorted(ts)[len(ts) // 2]
    bf_med = sorted(bf_t)[len(bf_t) // 2]
    print(f"[mxfp8]       median={med:.3f} ms  min={min(ts):.3f}  speedup={bf_med / med:.2f}x")


if __name__ == "__main__":
    main()
