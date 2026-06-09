# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the MXFP8 cuDNN sdpa_mxfp8 RuntimeError at Wan self-attention shapes.

Goal: settle three questions the colleague raised --
  1. Does TE.MXFP8Quantizer + cuDNN sdpa_mxfp8 THROW at S=75600 (720p/81f) but
     work at S=14040 (480p/33f)?  -> capture the FULL error message.
  2. Is the throw triggered by batch size B=2 (real run, CFG) vs B=1 (perf
     microbench)?  -> sweep B in {1, 2}.
  3. Does torch.ops.trtllm.mxfp8_quantize(..., alignment=32) make graph.execute()
     succeed where TE throws?  -> swap just the Q/K quantizer.

Faithfully replicates the quantize + graph code from
tensorrt_llm/_torch/visual_gen/attention_backend/mxfp8_cudnn.py.
"""

import math
import sys

import torch
import torch.nn.functional as F

H, D = 40, 128  # Wan2.2-T2V-A14B self-attn
SHAPES = [14040, 75600]  # 480p/33f , 720p/81f
BATCHES = [1, 2]  # perf microbench used 1 ; real run (CFG) uses 2


def ceil_div(a, b):
    return (a + b - 1) // b


# ----------------------------------------------------------------------------
print("=" * 72)
print("VERSIONS")
print("=" * 72)
print("torch                 ", torch.__version__)
print("torch.backends.cudnn  ", torch.backends.cudnn.version())
import cudnn  # noqa: E402

print("cudnn-frontend        ", cudnn.__version__)
print("cudnn backend_version ", cudnn.backend_version())
import transformer_engine  # noqa: E402,F401
import transformer_engine_torch as tex  # noqa: E402
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer  # noqa: E402

print("transformer_engine    ", transformer_engine.__version__)
print("device cap            ", torch.cuda.get_device_capability())
print("device                ", torch.cuda.get_device_name(0))
HAS_TRTLLM_OP = hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "mxfp8_quantize")
try:
    import tensorrt_llm  # noqa: F401

    HAS_TRTLLM_OP = hasattr(torch.ops.trtllm, "mxfp8_quantize")
except Exception as e:
    print("trtllm import:", repr(e))
print("has torch.ops.trtllm.mxfp8_quantize:", HAS_TRTLLM_OP)
print()


# ----------------------------------------------------------------------------
# TE quantize helpers (verbatim from mxfp8_cudnn.py)
# ----------------------------------------------------------------------------
def te_quantize_q_or_k_along_d(x_bhsd):
    B, H_, S, D_ = x_bhsd.shape
    L = B * H_
    block = 32
    d_scale = ceil_div(D_, block)
    d_scale_padded = ceil_div(d_scale, 4) * 4
    d_padded = d_scale_padded * block
    s_scale = ceil_div(S, block)
    s_scale_padded = ceil_div(s_scale, 4) * 4
    s_padded = s_scale_padded * block
    x = x_bhsd.reshape(L, S, D_)
    pad_d = d_padded - D_
    pad_s = s_padded - S
    if pad_s or pad_d:
        x = F.pad(x, (0, pad_d, 0, pad_s))
    x2d = x.reshape(L * s_padded, d_padded)
    q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    q.optimize_for_gemm = True
    res = q(x2d)
    fp8 = (
        res._rowwise_data.reshape(L, s_padded, d_padded)[:, :S, :D_]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(B, H_, S, D_)
    )
    return fp8, res._rowwise_scale_inv


def te_quantize_v_along_s(x_bhsd):
    B, H_, S, D_ = x_bhsd.shape
    L = B * H_
    block = 32
    d_scale = ceil_div(D_, block)
    d_scale_padded = ceil_div(d_scale, 4) * 4
    d_padded = d_scale_padded * block
    s_scale = ceil_div(S, block)
    s_scale_padded = ceil_div(s_scale, 4) * 4
    s_padded = s_scale_padded * block
    x = x_bhsd.reshape(L, S, D_)
    pad_d = d_padded - D_
    pad_s = s_padded - S
    if pad_s or pad_d:
        x = F.pad(x, (0, pad_d, 0, pad_s))
    x2d = x.reshape(L * s_padded, d_padded)
    q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=False, columnwise=True)
    q.optimize_for_gemm = True
    res = q(x2d)
    fp8 = (
        res._columnwise_data.reshape(L, s_padded, d_padded)[:, :S, :D_]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(B, H_, S, D_)
    )
    return fp8, res._columnwise_scale_inv


def trtllm_quantize_q_or_k_along_d(x_bhsd):
    """Q/K rowwise (along D) via torch.ops.trtllm.mxfp8_quantize (the colleague's fix).

    Feeds the SAME pre-padded x2d as the TE path so the swizzled scale size matches
    cuDNN's expected B*H*s_pad*d_scale_padded. Quantizes along the last dim (D).
    """
    B, H_, S, D_ = x_bhsd.shape
    L = B * H_
    block = 32
    d_scale_padded = ceil_div(ceil_div(D_, block), 4) * 4
    d_padded = d_scale_padded * block
    s_scale_padded = ceil_div(ceil_div(S, block), 4) * 4
    s_padded = s_scale_padded * block
    x = x_bhsd.reshape(L, S, D_)
    pad_d = d_padded - D_
    pad_s = s_padded - S
    if pad_s or pad_d:
        x = F.pad(x, (0, pad_d, 0, pad_s))
    x2d = x.reshape(L * s_padded, d_padded).contiguous()
    val, sf = torch.ops.trtllm.mxfp8_quantize(x2d, True, 32)  # (swizzled, alignment=32)
    fp8 = val.reshape(L, s_padded, d_padded)[:, :S, :D_].contiguous().reshape(B, H_, S, D_)
    return fp8, sf


_UID_Q, _UID_K, _UID_V = 0, 1, 2
_UID_SFQ, _UID_SFK, _UID_SFV = 5, 6, 7
_UID_O, _UID_AMAX_O = 3, 12


def build_graph(B, H_, S, D_, attn_scale, handle):
    s_pad = ceil_div(S, 128) * 128
    d_scale_padded = ceil_div(ceil_div(D_, 32), 4) * 4
    s_scale_padded = ceil_div(ceil_div(S, 32), 4) * 4
    d_padded = ceil_div(D_, 128) * 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )

    def fp8_in(uid):
        return g.tensor(
            uid=uid,
            dim=(B, H_, S, D_),
            stride=(H_ * S * D_, S * D_, D_, 1),
            data_type=cudnn.data_type.FP8_E4M3,
        )

    q, k, v = fp8_in(_UID_Q), fp8_in(_UID_K), fp8_in(_UID_V)
    sf_q = g.tensor(
        uid=_UID_SFQ,
        dim=(B, H_, s_pad, d_scale_padded),
        stride=(H_ * s_pad * d_scale_padded, s_pad * d_scale_padded, d_scale_padded, 1),
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    sf_k = g.tensor(
        uid=_UID_SFK,
        dim=(B, H_, s_pad, d_scale_padded),
        stride=(H_ * s_pad * d_scale_padded, s_pad * d_scale_padded, d_scale_padded, 1),
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    sf_v = g.tensor(
        uid=_UID_SFV,
        dim=(B, H_, s_scale_padded, d_padded),
        stride=(H_ * s_scale_padded * d_padded, s_scale_padded * d_padded, d_padded, 1),
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
    o.set_uid(_UID_O).set_output(True).set_dim((B, H_, S, D_)).set_stride(
        (H_ * S * D_, S * D_, D_, 1)
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


def expected_sf_numels(B, S):
    s_pad = ceil_div(S, 128) * 128
    d_scale_padded = ceil_div(ceil_div(D, 32), 4) * 4
    s_scale_padded = ceil_div(ceil_div(S, 32), 4) * 4
    d_padded = ceil_div(D, 128) * 128
    qk = B * H * s_pad * d_scale_padded
    v = B * H * s_scale_padded * d_padded
    return qk, v


def run_once(B, S, qk_quant="TE", verbose=False):
    """Run one cuDNN sdpa_mxfp8 forward; qk_quant selects the Q/K quantizer.

    'TE' = TransformerEngine MXFP8Quantizer (the original backend); 'trtllm' =
    torch.ops.trtllm.mxfp8_quantize (colleague's fix). V always uses the TE
    columnwise path (its swizzled layout is hard to match with the last-dim-only
    torch.ops op), so this isolates whether the Q/K quantizer is the trigger.
    """
    torch.manual_seed(42)
    dev = "cuda"
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=dev) * 0.5
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=dev) * 0.5
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=dev) * 0.5
    try:
        if qk_quant == "TE":
            qf, sfq = te_quantize_q_or_k_along_d(q)
            kf, sfk = te_quantize_q_or_k_along_d(k)
        elif qk_quant == "trtllm":
            qf, sfq = trtllm_quantize_q_or_k_along_d(q)
            kf, sfk = trtllm_quantize_q_or_k_along_d(k)
        else:
            raise ValueError(qk_quant)
        vf, sfv = te_quantize_v_along_s(v)  # V always TE
        qk_exp, v_exp = expected_sf_numels(B, S)
        if verbose:
            # Compare TE vs torch.ops scale-tensor size for Q/K against cuDNN's expectation.
            te_qk = te_quantize_q_or_k_along_d(q)[1].numel()
            to_qk = trtllm_quantize_q_or_k_along_d(q)[1].numel() if HAS_TRTLLM_OP else -1
            print(
                f"        Q/K sf numel: TE={te_qk:>12}  torch.ops={to_qk:>12}  "
                f"cuDNN_expects={qk_exp:>12}  (TE {'==' if te_qk == qk_exp else '!='} expected, "
                f"torch.ops {'==' if to_qk == qk_exp else '!='} expected)"
            )
            print(
                f"        sfq.numel={sfq.numel():>12}  expected_qk={qk_exp:>12}  "
                f"{'MATCH' if sfq.numel() == qk_exp else 'MISMATCH'}"
            )
            print(
                f"        sfv.numel={sfv.numel():>12}  expected_v ={v_exp:>12}  "
                f"{'MATCH' if sfv.numel() == v_exp else 'MISMATCH'}"
            )
            print(
                f"        sfq.dtype={sfq.dtype} shape={tuple(sfq.shape)}  S*S={S * S} "
                f"({'>INT32' if S * S > 2**31 - 1 else '<int32'})"
            )
        handle = cudnn.create_handle()
        g = build_graph(B, H, S, D, scale, handle)
        out = torch.empty(B, H, S, D, dtype=torch.bfloat16, device=dev)
        amax_o = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device=dev)
        ws = torch.empty(g.get_workspace_size(), dtype=torch.uint8, device=dev)
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
        g.execute(var_pack, ws, handle=handle)
        torch.cuda.synchronize()
        # also compute a bf16 reference to gauge error magnitude
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)
        rel = (out.float() - ref.float()).norm() / ref.float().norm()
        return "OK", f"out.norm={out.float().norm():.2f} rel_err_vs_bf16={rel:.4f}"
    except Exception as e:
        return "ERR", f"{type(e).__name__}: {e}"


print("=" * 72)
print("EXPERIMENT A : TE.MXFP8Quantizer (the original backend) -- B x S sweep")
print("=" * 72)
for S in SHAPES:
    for B in BATCHES:
        print(f"[B={B} S={S}]  (verbose diagnostics)")
        status, msg = run_once(B, S, qk_quant="TE", verbose=True)
        print(f"    --> {status}: {msg}")
        print()
        sys.stdout.flush()

print("=" * 72)
print("EXPERIMENT B : colleague's fix -- torch.ops.trtllm.mxfp8_quantize for Q/K")
print("(V kept on TE; isolates whether the Q/K scale tensor is what cuDNN rejects)")
print("=" * 72)
if not HAS_TRTLLM_OP:
    print("  torch.ops.trtllm.mxfp8_quantize NOT available -- skipping Experiment B")
else:
    for S in SHAPES:
        for B in BATCHES:
            print(f"[B={B} S={S}  Q/K=torch.ops  V=TE]")
            status, msg = run_once(B, S, qk_quant="trtllm", verbose=False)
            print(f"    --> {status}: {msg}")
            sys.stdout.flush()

print("done.")
