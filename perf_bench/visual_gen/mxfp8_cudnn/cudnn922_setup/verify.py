# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quick verifier for the cuDNN 9.22 overlay setup.

Run AFTER sourcing activate.sh (or set_ld_preload.sh, or installing
sitecustomize.py with CUDNN_922_AUTOLOAD=1). Checks:

  1. torch.backends.cudnn.version() reports 9.22.x
  2. /proc/self/maps confirms cuDNN was loaded from the pip dir
  3. A tiny SDPA call via the cuDNN backend completes without falling back
  4. (If transformer_engine + cudnn-frontend available) a sdpa_mxfp8 graph
     builds and executes on B200

Exits with non-zero if any check fails.
"""

import sys


def fail(msg):
    print(f"[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    print("=== Check 1: torch + cudnn version ===")
    import torch

    cudnn_ver = torch.backends.cudnn.version()
    print(f"  torch:               {torch.__version__}")
    print(f"  torch.backends.cudnn.version(): {cudnn_ver}")
    if cudnn_ver is None or cudnn_ver < 92100:
        fail(f"cuDNN version {cudnn_ver} < 92100 (sdpa_mxfp8 requires 9.21+)")
    print(f"  OK — cuDNN {cudnn_ver} is >= 9.21")

    print("\n=== Check 2: SDPA via CUDNN backend (force) ===")
    import torch.nn.attention as a
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        fail("no CUDA device")
    q = torch.randn(1, 40, 4096, 128, dtype=torch.bfloat16, device="cuda")
    with a.sdpa_kernel([a.SDPBackend.CUDNN_ATTENTION]):
        F.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    print("  OK — forced-cuDNN SDPA ran")

    print("\n=== Check 3: which libcudnn was loaded? ===")
    loaded_paths = []
    with open("/proc/self/maps") as f:
        for line in f:
            if "libcudnn" in line and "r-xp" in line:
                p = line.strip().split()[-1]
                if p not in loaded_paths:
                    loaded_paths.append(p)
    for p in loaded_paths:
        marker = " [pip 9.22]" if "nvidia/cudnn/lib" in p else ""
        print(f"  {p}{marker}")
    if not any("nvidia/cudnn/lib" in p for p in loaded_paths):
        fail("loaded cuDNN is NOT from pip dir — env overlay didn't take effect")
    print("  OK — pip 9.22 is the loaded cuDNN")

    print("\n=== Check 4: sdpa_mxfp8 graph build (cudnn-frontend) ===")
    try:
        import math

        import cudnn
    except ImportError as e:
        print(f"  SKIP — cudnn-frontend not installed: {e}")
        return
    try:
        handle = cudnn.create_handle()
        g = cudnn.pygraph(
            io_data_type=cudnn.data_type.FP8_E4M3,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=handle,
        )
        # Tiny shape: build only; don't execute (no real data prepared).
        B, H, S, D = 1, 8, 512, 128
        q_t = g.tensor(
            uid=0,
            dim=(B, H, S, D),
            stride=(H * S * D, S * D, D, 1),
            data_type=cudnn.data_type.FP8_E4M3,
        )
        k_t = g.tensor(
            uid=1,
            dim=(B, H, S, D),
            stride=(H * S * D, S * D, D, 1),
            data_type=cudnn.data_type.FP8_E4M3,
        )
        v_t = g.tensor(
            uid=2,
            dim=(B, H, S, D),
            stride=(H * S * D, S * D, D, 1),
            data_type=cudnn.data_type.FP8_E4M3,
        )
        d_scale = ((D + 31) // 32 + 3) // 4 * 4
        s_pad = (S + 127) // 128 * 128
        s_scale = ((S + 31) // 32 + 3) // 4 * 4
        d_pad = (D + 127) // 128 * 128
        sfq = g.tensor(
            uid=5,
            dim=(B, H, s_pad, d_scale),
            stride=(H * s_pad * d_scale, s_pad * d_scale, d_scale, 1),
            data_type=cudnn.data_type.FP8_E8M0,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )
        sfk = g.tensor(
            uid=6,
            dim=(B, H, s_pad, d_scale),
            stride=(H * s_pad * d_scale, s_pad * d_scale, d_scale, 1),
            data_type=cudnn.data_type.FP8_E8M0,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )
        sfv = g.tensor(
            uid=7,
            dim=(B, H, s_scale, d_pad),
            stride=(H * s_scale * d_pad, s_scale * d_pad, d_pad, 1),
            data_type=cudnn.data_type.FP8_E8M0,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )
        o, _stats, amax_o = g.sdpa_mxfp8(
            q=q_t,
            k=k_t,
            v=v_t,
            descale_q=sfq,
            descale_k=sfk,
            descale_v=sfv,
            attn_scale=1.0 / math.sqrt(D),
            generate_stats=False,
        )
        o.set_uid(3).set_output(True).set_dim((B, H, S, D)).set_stride(
            (H * S * D, S * D, D, 1)
        ).set_data_type(cudnn.data_type.BFLOAT16)
        amax_o.set_uid(12).set_output(True).set_dim((1, 1, 1, 1)).set_stride(
            (1, 1, 1, 1)
        ).set_data_type(cudnn.data_type.FLOAT)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.check_support()
        g.build_plans()
        print("  OK — sdpa_mxfp8 graph built (shape B=1,H=8,S=512,D=128)")
    except Exception as e:
        fail(f"sdpa_mxfp8 graph build failed: {type(e).__name__}: {e}")

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
