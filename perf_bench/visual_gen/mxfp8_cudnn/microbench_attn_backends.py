# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare per-call attention latency across all production backends at Wan-2.2 self-attn shapes.

Backends compared (all consume the *same* bf16 Q/K/V tensors so per-call cost
includes whatever pre-quantization the backend needs internally — matches what
Wan's pipeline actually does at every denoising step):

  - VANILLA            torch.nn.functional.scaled_dot_product_attention (bf16)
  - MXFP8_CUDNN        cuDNN sdpa_mxfp8 with TE pre-quantize (matches REPORT.html §4 "full mxfp8")
  - Sage (1, 16, 1)    TRTLLM Sage attention, qk_int8=True, K block size 16
  - Sage (1, 4, 1)     TRTLLM Sage attention, qk_int8=True, K block size 4

Shapes are Wan-2.2 T2V-A14B self-attention: H=40, D=128. S is swept across the
shapes the diffusion pipeline actually hits.

Output:
  - One row per (shape, backend) pair: median, min, p10/p90, speedup vs VANILLA
  - JSON dump to --out for downstream tabulation
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# --- TRT-LLM imports gated by env (only available inside the container) ---
from tensorrt_llm._torch.visual_gen.attention_backend import (  # noqa: E402
    MXFP8CudnnAttention,
    TrtllmAttention,
)
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


def bench_mxfp8(B, H, S, D, q_nhd, k_nhd, v_nhd, warmup, iters):
    """MXFP8_CUDNN backend with internal TE pre-quantize on every call."""
    backend = MXFP8CudnnAttention(
        layer_idx=0,
        num_heads=H,
        head_dim=D,
        dtype=torch.bfloat16,
        max_batch_size=B,
        max_seq_len=S,
    )

    def fn():
        return backend.forward(q_nhd, k_nhd, v_nhd)

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
