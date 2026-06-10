#!/bin/bash
# Standalone invariance gate: prove Q'.K'^T == Q.K^T (fp32) for SmoothQuant + Hadamard.
# Runs from /tmp (NOT the worktree) so the INSTALLED tensorrt_llm (with C++ bindings)
# is imported, not the bindings-less worktree copy.
export HOME=/tmp
cd /tmp || exit 1
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import os, torch
torch.backends.cuda.matmul.allow_tf32 = False  # true fp32 matmul, so residual reflects the transform math, not TF32
torch.set_float32_matmul_precision("highest")
from tensorrt_llm._torch.visual_gen.attention_backend import mxfp8_cudnn as m
assert hasattr(m, "_smoothquant_qk") and hasattr(m, "_hadamard_qk"), "patched backend not overlaid!"
torch.manual_seed(0)
q = torch.randn(2, 40, 512, 128, dtype=torch.float32, device="cuda")
k = torch.randn(2, 40, 512, 128, dtype=torch.float32, device="cuda")
ref = q @ k.transpose(-1, -2)
relerr = lambda qq, kk: ((qq @ kk.transpose(-1, -2) - ref).norm() / ref.norm()).item()
ok = True
for mode in ["k", "qk", "q"]:
    os.environ["TRTLLM_VISUAL_GEN_MXFP8_SMOOTHQUANT"] = mode; m._SQ_LOGGED = True
    e = relerr(*m._smoothquant_qk(q, k)); ok = ok and e < 1e-4
    print(f"SmoothQuant {mode:>2}: QK^T rel-err = {e:.2e}")
os.environ.pop("TRTLLM_VISUAL_GEN_MXFP8_SMOOTHQUANT", None)
os.environ["TRTLLM_VISUAL_GEN_MXFP8_HADAMARD"] = "1"; m._HAD_LOGGED = True
e = relerr(*m._hadamard_qk(q, k)); ok = ok and e < 1e-4
print(f"Hadamard   : QK^T rel-err = {e:.2e}")
os.environ.pop("TRTLLM_VISUAL_GEN_MXFP8_HADAMARD", None)
print("INVARIANCE_OK" if ok else "INVARIANCE_FAIL")
PY
