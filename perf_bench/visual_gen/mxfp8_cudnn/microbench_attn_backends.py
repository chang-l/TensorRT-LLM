# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-only per-call attention latency across all production backends at Wan-2.2 self-attn shapes.

The timed region for each backend covers *only the SDPA kernel itself*:

  - VANILLA            torch.nn.functional.scaled_dot_product_attention (bf16)
                       Already kernel-only by construction (no setup per call).
  - MXFP8_CUDNN        cuDNN sdpa_mxfp8 graph.execute() with Q/K/V *pre-quantized
                       once* outside the timing loop. Matches REPORT.html §4a
                       "Kernel-only" measurement protocol.
  - Sage (1, 16, 1)    TRTLLM Sage attention, qk_int8=True. The int8 Q/K quant
                       is *fused inside* the kernel — there is no separable
                       pre-quant step. Metadata is prepared during warmup and
                       cached for subsequent calls (same shape), so wrapper
                       forward() time equals kernel time after warmup.
  - Sage (1, 4, 1)     Same, K block size 4.

Shapes are Wan-2.2 T2V-A14B self-attention: H=40, D=128. S is swept across the
shapes the diffusion pipeline actually hits.

Output:
  - One row per (shape, backend) pair: median, min, p10/p90, TFLOPS, speedup
  - JSON dump to --out for downstream tabulation
"""

import argparse
import json
import math
import time
from pathlib import Path

# For kernel-only MXFP8 timing we use cudnn-frontend directly (same approach as
# microbench_mxfp8_sdpa.py / §4a in REPORT.html). The MXFP8CudnnAttention
# wrapper re-quantizes Q/K/V every call, which would not be kernel-only.
import cudnn  # noqa: E402
import torch
import torch.nn.functional as F
import transformer_engine  # noqa: F401, E402  - must import before cudnn
import transformer_engine_torch as tex  # noqa: E402
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer  # noqa: E402

# --- TRT-LLM imports gated by env (only available inside the container) ---
from tensorrt_llm._torch.visual_gen.attention_backend import TrtllmAttention  # noqa: E402
from tensorrt_llm._torch.visual_gen.config import SageAttentionConfig  # noqa: E402

WAN_SHAPES = [
    # (label,    S,     description)
    ("S=4680", 4680, "warmup shape (480x832 / 9f)"),
    ("S=14040", 14040, "480x832 / 33f (mid-S)"),
    ("S=27000", 27000, "720x1280 / 33f (mid+S)"),
    ("S=75600", 75600, "720x1280 / 81f (production target)"),
]

BACKENDS = ["VANILLA", "MXFP8_CUDNN", "sage_blk16", "sage_blk4"]
PRETTY = {
    "VANILLA": "VANILLA bf16 (torch SDPA)",
    "MXFP8_CUDNN": "MXFP8_CUDNN (full path)",
    "sage_blk16": "Sage (1, 16, 1) qk_int8",
    "sage_blk4": "Sage (1, 4, 1) qk_int8",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=40)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="microbench_attn_backends.json")
    p.add_argument(
        "--backends",
        nargs="+",
        default=BACKENDS,
        help=f"Subset to run. Default: {BACKENDS}",
    )
    p.add_argument(
        "--shapes",
        nargs="+",
        default=None,
        help="Subset of shape labels (e.g. S=4680). Default: all 4.",
    )
    return p.parse_args()


def time_op(fn, warmup, iters):
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


def percentile(vals, q):
    vals = sorted(vals)
    k = (len(vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def attn_flops(B, H, S, D):
    """Theoretical FLOPS for one non-causal SDPA call.

    Two matmuls dominate: QK^T (B*H*S*S*D mul-adds) and AV (B*H*S*S*D mul-adds).
    Counting one mul-add as 2 flops: total = 4 * B * H * S^2 * D.
    Softmax (~5 * B*H*S^2 flops) is excluded by convention.
    """
    return 4 * B * H * S * S * D


def tflops(B, H, S, D, latency_ms):
    """Achieved TFLOPS for a single attention call given median latency in ms."""
    return attn_flops(B, H, S, D) / (latency_ms * 1e-3) / 1e12


def bench_vanilla(B, H, S, D, q_nhd, k_nhd, v_nhd, warmup, iters):
    """bf16 torch SDPA. q_nhd/k_nhd/v_nhd are (B, S, H, D) — convert to (B, H, S, D)."""
    q = q_nhd.permute(0, 2, 1, 3).contiguous()
    k = k_nhd.permute(0, 2, 1, 3).contiguous()
    v = v_nhd.permute(0, 2, 1, 3).contiguous()
    scale = 1.0 / math.sqrt(D)

    def fn():
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)

    return time_op(fn, warmup, iters)


def _ceil_div(a, b):
    return (a + b - 1) // b


def _te_quantize_for_mxfp8_sdpa(x_bhsd):
    """Pre-quantize a (B, H, S, D) bf16 tensor for cuDNN sdpa_mxfp8.

    Returns (fp8_rowwise, sf_rowwise, fp8_colwise, sf_colwise) — rowwise is for
    Q/K (quantized along D), colwise is for V (quantized along S). Scales are in
    swizzled F8_128x4 layout that sdpa_mxfp8 expects.
    """
    B, H, S, D = x_bhsd.shape
    L = B * H
    bs = 32
    d_scale_padded = _ceil_div(_ceil_div(D, bs), 4) * 4
    d_padded = d_scale_padded * bs
    s_scale_padded = _ceil_div(_ceil_div(S, bs), 4) * 4
    s_padded = s_scale_padded * bs

    x = x_bhsd.reshape(L, S, D)
    pad_d = d_padded - D
    pad_s = s_padded - S
    if pad_s or pad_d:
        x = F.pad(x, (0, pad_d, 0, pad_s))
    x2d = x.reshape(L * s_padded, d_padded)

    q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    q.optimize_for_gemm = True
    res = q(x2d)
    fp8_dtype = torch.float8_e4m3fn

    fp8_row = (
        res._rowwise_data.reshape(L, s_padded, d_padded)[:, :S, :D]
        .contiguous()
        .view(fp8_dtype)
        .reshape(B, H, S, D)
    )
    sf_row = res._rowwise_scale_inv
    fp8_col = (
        res._columnwise_data.reshape(L, s_padded, d_padded)[:, :S, :D]
        .contiguous()
        .view(fp8_dtype)
        .reshape(B, H, S, D)
    )
    sf_col = res._columnwise_scale_inv
    return fp8_row, sf_row, fp8_col, sf_col


def _build_cudnn_mxfp8_sdpa_graph(B, H, S, D, attn_scale, handle):
    """Build the cuDNN sdpa_mxfp8 execution graph (one-time per shape)."""
    UID_Q, UID_K, UID_V = 0, 1, 2
    UID_SFQ, UID_SFK, UID_SFV = 5, 6, 7
    UID_O, UID_AMAX_O = 3, 12

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
    o, _stats, amax_o = g.sdpa_mxfp8(
        q=q,
        k=k,
        v=v,
        descale_q=sfq,
        descale_k=sfk,
        descale_v=sfv,
        attn_scale=attn_scale,
        generate_stats=False,
    )
    o.set_uid(UID_O).set_output(True).set_dim((B, H, S, D)).set_stride(
        (H * S * D, S * D, D, 1)
    ).set_data_type(cudnn.data_type.BFLOAT16)
    amax_o.set_uid(UID_AMAX_O).set_output(True).set_dim((1, 1, 1, 1)).set_stride(
        (1, 1, 1, 1)
    ).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    g.check_support()
    g.build_plans()
    return g, {
        "UID_Q": UID_Q,
        "UID_K": UID_K,
        "UID_V": UID_V,
        "UID_SFQ": UID_SFQ,
        "UID_SFK": UID_SFK,
        "UID_SFV": UID_SFV,
        "UID_O": UID_O,
        "UID_AMAX_O": UID_AMAX_O,
    }


def bench_mxfp8(B, H, S, D, q_nhd, k_nhd, v_nhd, warmup, iters):
    """CuDNN sdpa_mxfp8 *kernel only*: pre-quantize Q/K/V once, time graph.execute()."""
    # NHD (B, S, H, D) -> BHSD (B, H, S, D) for cuDNN
    q = q_nhd.permute(0, 2, 1, 3).contiguous()
    k = k_nhd.permute(0, 2, 1, 3).contiguous()
    v = v_nhd.permute(0, 2, 1, 3).contiguous()

    # Pre-quantize OUTSIDE the timing loop.
    qf, sfq, _, _ = _te_quantize_for_mxfp8_sdpa(q)
    kf, sfk, _, _ = _te_quantize_for_mxfp8_sdpa(k)
    _, _, vf, sfv = _te_quantize_for_mxfp8_sdpa(v)

    attn_scale = 1.0 / math.sqrt(D)
    handle = cudnn.create_handle()
    graph, uids = _build_cudnn_mxfp8_sdpa_graph(B, H, S, D, attn_scale, handle)

    o_buf = torch.empty(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    amax_o = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device="cuda")
    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
    var_pack = {
        uids["UID_Q"]: qf,
        uids["UID_K"]: kf,
        uids["UID_V"]: vf,
        uids["UID_SFQ"]: sfq,
        uids["UID_SFK"]: sfk,
        uids["UID_SFV"]: sfv,
        uids["UID_O"]: o_buf,
        uids["UID_AMAX_O"]: amax_o,
    }

    def fn():
        graph.execute(var_pack, workspace, handle=handle)

    return time_op(fn, warmup, iters)


def bench_sage(B, H, S, D, q_nhd, k_nhd, v_nhd, warmup, iters, k_block):
    """Sage attention via TrtllmAttention with (1, k_block, 1) qk_int8=True."""
    sage_cfg = SageAttentionConfig(
        num_elts_per_blk_q=1,
        num_elts_per_blk_k=k_block,
        num_elts_per_blk_v=1,
        qk_int8=True,
    )
    # TRTLLM expects a model-scoped mutable dict for cross-layer metadata sharing.
    metadata_state = {"metadata": None, "capacity": (0, 0)}
    backend = TrtllmAttention(
        layer_idx=0,
        num_heads=H,
        head_dim=D,
        dtype=torch.bfloat16,
        max_batch_size=B,
        max_seq_len=S,
        attention_metadata_state=metadata_state,
        sage_attention_config=sage_cfg,
    )

    def fn():
        return backend.forward(q_nhd, k_nhd, v_nhd)

    return time_op(fn, warmup, iters)


def main():
    args = parse_args()
    torch.cuda.set_device(0)
    dev_name = torch.cuda.get_device_name(0)
    cuda_cap = torch.cuda.get_device_capability(0)
    print(f"device: {dev_name} (cc={cuda_cap[0]}.{cuda_cap[1]})")
    print(f"backends: {args.backends}")
    print(
        f"shapes:   {[s for s in [x[0] for x in WAN_SHAPES] if not args.shapes or s in args.shapes]}"
    )
    print()

    rows = []
    for label, S, desc in WAN_SHAPES:
        if args.shapes is not None and label not in args.shapes:
            continue
        B, H, D = args.batch, args.heads, args.dim
        print(f"=== {label} ({desc}) — B={B} H={H} D={D} ===")

        g = torch.Generator(device="cuda").manual_seed(args.seed)
        q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda", generator=g) * 0.5
        k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda", generator=g) * 0.5
        v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda", generator=g) * 0.5

        results_this_shape = {}
        for backend in args.backends:
            try:
                if backend == "VANILLA":
                    ts = bench_vanilla(B, H, S, D, q, k, v, args.warmup, args.iters)
                elif backend == "MXFP8_CUDNN":
                    ts = bench_mxfp8(B, H, S, D, q, k, v, args.warmup, args.iters)
                elif backend == "sage_blk16":
                    ts = bench_sage(B, H, S, D, q, k, v, args.warmup, args.iters, k_block=16)
                elif backend == "sage_blk4":
                    ts = bench_sage(B, H, S, D, q, k, v, args.warmup, args.iters, k_block=4)
                else:
                    print(f"  [{backend}] unknown backend, skipping")
                    continue
            except Exception as e:
                print(f"  [{backend}] FAILED: {type(e).__name__}: {e}")
                rows.append(
                    {
                        "shape": label,
                        "S": S,
                        "backend": backend,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                continue
            med = percentile(ts, 0.5)
            results_this_shape[backend] = med
            spk = (results_this_shape["VANILLA"] / med) if "VANILLA" in results_this_shape else None
            achieved_tflops = tflops(B, H, S, D, med)
            print(
                f"  [{PRETTY[backend]:<32}] "
                f"med={med:7.3f}ms min={min(ts):7.3f} "
                f"p10={percentile(ts, 0.10):7.3f} p90={percentile(ts, 0.90):7.3f} "
                f"TFLOPS={achieved_tflops:6.1f}"
                + (f"  speedup={spk:.2f}x" if spk is not None else "")
            )
            rows.append(
                {
                    "shape": label,
                    "S": S,
                    "backend": backend,
                    "median_ms": med,
                    "min_ms": min(ts),
                    "max_ms": max(ts),
                    "p10_ms": percentile(ts, 0.10),
                    "p90_ms": percentile(ts, 0.90),
                    "iters": args.iters,
                    "warmup": args.warmup,
                    "speedup_vs_vanilla": spk,
                    "tflops_achieved": achieved_tflops,
                    "flops_per_call": attn_flops(B, H, S, D),
                }
            )
        print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "device": dev_name,
            "cuda_capability": list(cuda_cap),
            "B": args.batch,
            "H": args.heads,
            "D": args.dim,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "rows": rows,
        },
        open(out_path, "w"),
        indent=2,
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
