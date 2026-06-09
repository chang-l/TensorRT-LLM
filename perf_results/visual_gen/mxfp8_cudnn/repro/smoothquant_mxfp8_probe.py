# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Probe: does SmoothQuant on the QK^T matmul improve MXFP8 attention accuracy?

SmoothQuant migrates per-channel magnitude across a matmul: for QK^T (contraction
over D), pick a per-D scale s_d, set Q' = Q / s_d, K' = K * s_d. In full precision
Q'·K'^T == Q·K^T exactly, but the MXFP8-quantized product has different error.
s_d = amax_d(|Q|)^alpha / amax_d(|K|)^(1-alpha):
  alpha -> 0 : K normalized  ("smooth K")
  alpha = 0.5: balanced       ("smooth Q/K")
  alpha -> 1 : Q normalized   ("smooth Q")

MXFP8 already block-scales every 32 elements along D, so SmoothQuant can only help
to the extent there is large dynamic range WITHIN a 32-channel block. We therefore
inject realistic same-channel outliers into Q and K (shared outlier channels, as
real attention activations tend to have) and measure rel-err vs the bf16 reference.

CAVEAT: synthetic data. A real verdict needs Q/K captured from a Wan forward.
"""
import math

import cudnn
import torch
import torch.nn.functional as F
import transformer_engine  # noqa: F401
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

H, D = 40, 128
DEV = "cuda"


def ceil_div(a, b):
    return (a + b - 1) // b


def _te_quant(x_bhsd, columnwise):
    B, H_, S, D_ = x_bhsd.shape
    L = B * H_
    blk = 32
    dsp = ceil_div(ceil_div(D_, blk), 4) * 4
    dp = dsp * blk
    ssp = ceil_div(ceil_div(S, blk), 4) * 4
    sp = ssp * blk
    x = x_bhsd.reshape(L, S, D_)
    if (sp - S) or (dp - D_):
        x = F.pad(x, (0, dp - D_, 0, sp - S))
    x2d = x.reshape(L * sp, dp)
    q = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3, rowwise=not columnwise, columnwise=columnwise
    )
    q.optimize_for_gemm = True
    res = q(x2d)
    data = res._columnwise_data if columnwise else res._rowwise_data
    sf = res._columnwise_scale_inv if columnwise else res._rowwise_scale_inv
    fp8 = data.reshape(L, sp, dp)[:, :S, :D_].contiguous().view(torch.float8_e4m3fn).reshape(
        B, H_, S, D_
    )
    return fp8, sf


def _build_graph(B, S, handle):
    s_pad = ceil_div(S, 128) * 128
    dsp = ceil_div(ceil_div(D, 32), 4) * 4
    ssp = ceil_div(ceil_div(S, 32), 4) * 4
    dp = ceil_div(D, 128) * 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )
    fin = lambda uid: g.tensor(
        uid=uid, dim=(B, H, S, D), stride=(H * S * D, S * D, D, 1), data_type=cudnn.data_type.FP8_E4M3
    )
    q, k, v = fin(0), fin(1), fin(2)
    sfq = g.tensor(uid=5, dim=(B, H, s_pad, dsp),
                   stride=(H * s_pad * dsp, s_pad * dsp, dsp, 1),
                   data_type=cudnn.data_type.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4)
    sfk = g.tensor(uid=6, dim=(B, H, s_pad, dsp),
                   stride=(H * s_pad * dsp, s_pad * dsp, dsp, 1),
                   data_type=cudnn.data_type.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4)
    sfv = g.tensor(uid=7, dim=(B, H, ssp, dp),
                   stride=(H * ssp * dp, ssp * dp, dp, 1),
                   data_type=cudnn.data_type.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4)
    o, _s, amax = g.sdpa_mxfp8(q=q, k=k, v=v, descale_q=sfq, descale_k=sfk, descale_v=sfv,
                              attn_scale=1.0 / math.sqrt(D), generate_stats=False)
    o.set_uid(3).set_output(True).set_dim((B, H, S, D)).set_stride(
        (H * S * D, S * D, D, 1)).set_data_type(cudnn.data_type.BFLOAT16)
    amax.set_uid(12).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(
        cudnn.data_type.FLOAT)
    g.validate(); g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    g.check_support(); g.build_plans()
    return g


def mxfp8_attn(q, k, v, handle, graph):
    B, _, S, _ = q.shape
    qf, sfq = _te_quant(q, False)
    kf, sfk = _te_quant(k, False)
    vf, sfv = _te_quant(v, True)
    out = torch.empty(B, H, S, D, dtype=torch.bfloat16, device=DEV)
    amax = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device=DEV)
    ws = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device=DEV)
    graph.execute({0: qf, 1: kf, 2: vf, 5: sfq, 6: sfk, 7: sfv, 3: out, 12: amax}, ws, handle=handle)
    return out


def make_inputs(B, S, outlier, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=DEV, generator=g) * 0.5
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=DEV, generator=g) * 0.5
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=DEV, generator=g) * 0.5
    if outlier:
        # shared outlier channels (real Q/K share massive-activation channels),
        # spread across the 32-wide MX blocks, asymmetric Q/K magnitude.
        chans = list(range(2, D, 15))  # ~9 channels, ~2-3 per 32-block
        for c in chans:
            q[..., c] *= 30.0
            k[..., c] *= 8.0
    return q, k, v


def smooth(q, k, alpha):
    aq = q.float().abs().amax(dim=(0, 1, 2)).clamp(min=1e-6)  # (D,)
    ak = k.float().abs().amax(dim=(0, 1, 2)).clamp(min=1e-6)
    s = (aq**alpha) / (ak ** (1 - alpha))
    s = s.clamp(min=1e-4, max=1e4).to(q.dtype)
    return q / s, k * s


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def run(B, S, outlier, seeds=(0, 1, 2)):
    handle = cudnn.create_handle()
    graph = _build_graph(B, S, handle)
    rows = {}
    for label, alpha in [("baseline", None), ("smoothK a=0.0", 0.0), ("a=0.25", 0.25),
                         ("Q/K a=0.5", 0.5), ("a=0.75", 0.75), ("smoothQ a=1.0", 1.0)]:
        errs_bf16, errs_fp32 = [], []
        for sd in seeds:
            q, k, v = make_inputs(B, S, outlier, sd)
            ref_bf16 = F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(D))
            ref_fp32 = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), scale=1.0 / math.sqrt(D))
            qq, kk = (q, k) if alpha is None else smooth(q, k, alpha)
            out = mxfp8_attn(qq, kk, v, handle, graph)
            errs_bf16.append(rel_err(out, ref_bf16))
            errs_fp32.append(rel_err(out, ref_fp32))
        rows[label] = (sum(errs_bf16) / len(errs_bf16), sum(errs_fp32) / len(errs_fp32))
    return rows


print("=" * 72)
print("SmoothQuant-for-MXFP8 probe (synthetic). torch", torch.__version__,
      "cudnn", cudnn.backend_version(), "TE", transformer_engine.__version__)
print(f"device {torch.cuda.get_device_name(0)}  H={H} D={D}")
print("=" * 72)
for outlier in (False, True):
    B, S = 1, 14040
    print(f"\n### outliers={'YES (shared Q/K channels, 30x/8x)' if outlier else 'NO (plain gaussian)'} "
          f"B={B} S={S}  (rel-err mean over 3 seeds) ###")
    rows = run(B, S, outlier)
    base = rows["baseline"][0]
    print(f"  {'variant':<16} {'rel_err_vs_bf16':>16} {'rel_err_vs_fp32':>16} {'vs baseline':>14}")
    for label, (e_bf16, e_fp32) in rows.items():
        delta = "" if label == "baseline" else f"{(e_bf16 / base - 1) * 100:+.1f}%"
        print(f"  {label:<16} {e_bf16:>16.4f} {e_fp32:>16.4f} {delta:>14}")
print("\ndone.")
