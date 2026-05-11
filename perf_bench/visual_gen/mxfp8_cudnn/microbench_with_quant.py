"""Microbench MXFP8 SDPA INCLUDING quantization, to measure end-to-end cost.

This is the realistic per-call cost of using MXFP8_CUDNN inside Wan's
self-attention, since Q/K/V change every step and must be quantized fresh.
"""

import argparse
import math

import cudnn
import torch
import torch.nn.functional as F
import transformer_engine  # noqa
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer


def ceil_div(a, b):
    return (a + b - 1) // b


# Pre-built quantize via TE
def quantize_mxfp8_rowwise(x_bhsd, fp8_dtype=torch.float8_e4m3fn, block_size=32):
    B, H, S, D = x_bhsd.shape
    L = B * H
    d_scale = ceil_div(D, block_size)
    d_scale_padded = ceil_div(d_scale, 4) * 4
    d_padded = d_scale_padded * block_size
    s_scale = ceil_div(S, block_size)
    s_scale_padded = ceil_div(s_scale, 4) * 4
    s_padded = s_scale_padded * block_size
    x = x_bhsd.reshape(L, S, D)
    pad_d = d_padded - D
    pad_s = s_padded - S
    if pad_s or pad_d:
        x = F.pad(x, (0, pad_d, 0, pad_s))
    x2d = x.reshape(L * s_padded, d_padded)
    q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    q.optimize_for_gemm = True
    res = q(x2d)
    fp8 = (
        res._rowwise_data.reshape(L, s_padded, d_padded)[:, :S, :D]
        .contiguous()
        .view(fp8_dtype)
        .reshape(B, H, S, D)
    )
    sf = res._rowwise_scale_inv
    return fp8, sf


def quantize_mxfp8_colwise(x_bhsd, fp8_dtype=torch.float8_e4m3fn, block_size=32):
    B, H, S, D = x_bhsd.shape
    L = B * H
    d_scale = ceil_div(D, block_size)
    d_scale_padded = ceil_div(d_scale, 4) * 4
    d_padded = d_scale_padded * block_size
    s_scale = ceil_div(S, block_size)
    s_scale_padded = ceil_div(s_scale, 4) * 4
    s_padded = s_scale_padded * block_size
    x = x_bhsd.reshape(L, S, D)
    pad_d = d_padded - D
    pad_s = s_padded - S
    if pad_s or pad_d:
        x = F.pad(x, (0, pad_d, 0, pad_s))
    x2d = x.reshape(L * s_padded, d_padded)
    q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=False, columnwise=True)
    q.optimize_for_gemm = True
    res = q(x2d)
    fp8 = (
        res._columnwise_data.reshape(L, s_padded, d_padded)[:, :S, :D]
        .contiguous()
        .view(fp8_dtype)
        .reshape(B, H, S, D)
    )
    sf = res._columnwise_scale_inv
    return fp8, sf


UID_Q, UID_K, UID_V = 0, 1, 2
UID_SFQ, UID_SFK, UID_SFV = 5, 6, 7
UID_O, UID_STATS, UID_AMAX_O = 3, 4, 12


def build_graph(B, H, S, D, attn_scale, gen_stats=False, handle=None):
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

    def fp8(uid):
        return g.tensor(
            uid=uid,
            dim=(B, H, S, D),
            stride=(H * S * D, S * D, D, 1),
            data_type=cudnn.data_type.FP8_E4M3,
        )

    q, k, v = fp8(UID_Q), fp8(UID_K), fp8(UID_V)

    sfq = g.tensor(
        uid=UID_SFQ,
        dim=(B, H, s_pad, d_scale_padded),
        stride=(H * s_pad * d_scale_padded, s_pad * d_scale_padded, d_scale_padded, 1),
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    sfk = g.tensor(
        uid=UID_SFK,
        dim=(B, H, s_pad, d_scale_padded),
        stride=(H * s_pad * d_scale_padded, s_pad * d_scale_padded, d_scale_padded, 1),
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    sfv = g.tensor(
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
        descale_q=sfq,
        descale_k=sfk,
        descale_v=sfv,
        attn_scale=attn_scale,
        generate_stats=gen_stats,
    )
    o.set_uid(UID_O).set_output(True).set_dim((B, H, S, D)).set_stride(
        (H * S * D, S * D, D, 1)
    ).set_data_type(cudnn.data_type.BFLOAT16)
    if stats is not None:
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


def time_op(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        s[i].record()
        fn()
        e[i].record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) for a, b in zip(s, e)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=40)
    ap.add_argument("--seq", type=int, default=4680)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--gen_stats", action="store_true")
    args = ap.parse_args()

    B, H, S, D = args.batch, args.heads, args.seq, args.dim
    print(f"shape B={B} H={H} S={S} D={D} gen_stats={args.gen_stats}")
    g = torch.Generator(device="cuda").manual_seed(0)
    qb = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda", generator=g) * 0.5
    kb = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda", generator=g) * 0.5
    vb = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda", generator=g) * 0.5
    attn_scale = 1.0 / math.sqrt(D)

    # bf16 baseline
    def bf_fn():
        return F.scaled_dot_product_attention(qb, kb, vb, scale=attn_scale)

    bf_t = time_op(bf_fn)
    print(f"[bf16 SDPA]                    median={sorted(bf_t)[len(bf_t) // 2]:.3f} ms")

    # build cudnn graph once
    handle = cudnn.create_handle()
    graph = build_graph(B, H, S, D, attn_scale, gen_stats=args.gen_stats, handle=handle)
    o_buf = torch.empty(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    stats_buf = torch.empty(B, H, S, 1, dtype=torch.float32, device="cuda")
    amax_o = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device="cuda")
    ws = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")

    # ============ time each phase ============
    # Phase A: TE rowwise quantize (Q or K)
    qf, sfq = quantize_mxfp8_rowwise(qb)

    def a_fn():
        return quantize_mxfp8_rowwise(qb)

    a_t = time_op(a_fn)
    print(f"[TE rowwise quantize Q (or K)] median={sorted(a_t)[len(a_t) // 2]:.3f} ms")

    # Phase B: TE colwise quantize (V)
    vf, sfv = quantize_mxfp8_colwise(vb)

    def b_fn():
        return quantize_mxfp8_colwise(vb)

    b_t = time_op(b_fn)
    print(f"[TE colwise quantize V]        median={sorted(b_t)[len(b_t) // 2]:.3f} ms")

    # Phase C: cuDNN sdpa_mxfp8 kernel only (pre-quantized)
    kf, sfk = quantize_mxfp8_rowwise(kb)
    var_pack = {
        UID_Q: qf,
        UID_K: kf,
        UID_V: vf,
        UID_SFQ: sfq,
        UID_SFK: sfk,
        UID_SFV: sfv,
        UID_O: o_buf,
        UID_AMAX_O: amax_o,
    }
    if args.gen_stats:
        var_pack[UID_STATS] = stats_buf

    def kfn():
        graph.execute(var_pack, ws, handle=handle)

    c_t = time_op(kfn)
    print(f"[sdpa_mxfp8 kernel only]       median={sorted(c_t)[len(c_t) // 2]:.3f} ms")

    # Phase E: TOTAL realistic per-call (3 quantize + 1 kernel)
    def total_fn():
        qf, sfq = quantize_mxfp8_rowwise(qb)
        kf, sfk = quantize_mxfp8_rowwise(kb)
        vf, sfv = quantize_mxfp8_colwise(vb)
        var_pack[UID_Q] = qf
        var_pack[UID_K] = kf
        var_pack[UID_V] = vf
        var_pack[UID_SFQ] = sfq
        var_pack[UID_SFK] = sfk
        var_pack[UID_SFV] = sfv
        graph.execute(var_pack, ws, handle=handle)

    e_t = time_op(total_fn)
    bf_med = sorted(bf_t)[len(bf_t) // 2]
    e_med = sorted(e_t)[len(e_t) // 2]
    print(
        f"[FULL mxfp8 path 3xquant+SDPA] median={e_med:.3f} ms  (vs bf16 {bf_med:.3f}ms = {bf_med / e_med:.2f}x)"
    )


if __name__ == "__main__":
    main()
