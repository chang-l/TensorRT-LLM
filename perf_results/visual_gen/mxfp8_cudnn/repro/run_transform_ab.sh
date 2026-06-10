#!/bin/bash
# A/B: does SmoothQuant (K / QK) or Hadamard rotation on Q/K improve MXFP8 LPIPS vs bf16?
# All transforms are QK^T-invariant + applied only to Q/K (never V); no silent fallback.
# Step 1 GATES on an fp32 invariance check (Q'.K'^T == Q.K^T to ~machine eps) before generating.
# Then 8-GPU sweep: 480p {VANILLA, MXnone, MXsqK, MXsqQK, MXhad} + 720p {VANILLA, MXnone, MXhad}.
set -o pipefail
WT=/home/scratch.liuc_coreai/codes/mxfp8-ctrl-feat
REPRO=/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/repro
MODEL=/home/scratch.liuc_coreai/ckpts/Wan2.2-T2V-A14B-Diffusers
LOG=/tmp/transform_ab.log
P480="busy_street dancer_jump cat_windowsill"
P720="busy_street cat_windowsill"
export HOME=/tmp
SITE=$(python3 -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "tensorrt_llm"))')
sudo cp "$WT/tensorrt_llm/_torch/visual_gen/attention_backend/mxfp8_cudnn.py" \
        "$SITE/_torch/visual_gen/attention_backend/mxfp8_cudnn.py"
cd "$WT" || exit 4
{
  echo "=== $(date) host=$(hostname) — transform A/B (SmoothQuant + Hadamard) ==="
  echo "=================== INVARIANCE GATE (fp32: Q'.K'^T must equal Q.K^T) ==================="
  CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import os, torch
os.environ.setdefault("HOME", "/tmp")
from tensorrt_llm._torch.visual_gen.attention_backend import mxfp8_cudnn as m
torch.manual_seed(0)
# fp32 inputs so the helpers run in fp32 -> isolates analytical invariance from bf16 rounding
q = torch.randn(2, 40, 512, 128, dtype=torch.float32, device="cuda")
k = torch.randn(2, 40, 512, 128, dtype=torch.float32, device="cuda")
ref = q @ k.transpose(-1, -2)
def relerr(qq, kk):
    return ((qq @ kk.transpose(-1, -2) - ref).norm() / ref.norm()).item()
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
  echo "=================== 8-GPU generation ==================="
  RP=perf_bench/visual_gen/mxfp8_cudnn/run_prompt_suite.py
  C480="--model_path $MODEL --height 480 --width 832 --num_frames 33 --steps 40 --prompts $P480"
  C720="--model_path $MODEL --height 720 --width 1280 --num_frames 81 --steps 40 --prompts $P720"
  # 480p (GPUs 0-4)
  CUDA_VISIBLE_DEVICES=0 python3 $RP --backend VANILLA     --backend_tag VANILLA --gpu_id 0 --out_dir /tmp/ab480 $C480 > /tmp/ab_van480.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python3 $RP --backend MXFP8_CUDNN --backend_tag MXnone  --gpu_id 1 --out_dir /tmp/ab480 $C480 > /tmp/ab_mxnone480.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 TRTLLM_VISUAL_GEN_MXFP8_SMOOTHQUANT=k  python3 $RP --backend MXFP8_CUDNN --backend_tag MXsqK  --gpu_id 2 --out_dir /tmp/ab480 $C480 > /tmp/ab_sqk480.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 TRTLLM_VISUAL_GEN_MXFP8_SMOOTHQUANT=qk python3 $RP --backend MXFP8_CUDNN --backend_tag MXsqQK --gpu_id 3 --out_dir /tmp/ab480 $C480 > /tmp/ab_sqqk480.log 2>&1 &
  CUDA_VISIBLE_DEVICES=4 TRTLLM_VISUAL_GEN_MXFP8_HADAMARD=1     python3 $RP --backend MXFP8_CUDNN --backend_tag MXhad  --gpu_id 4 --out_dir /tmp/ab480 $C480 > /tmp/ab_had480.log 2>&1 &
  # 720p (GPUs 5-7)
  CUDA_VISIBLE_DEVICES=5 python3 $RP --backend VANILLA     --backend_tag VANILLA --gpu_id 5 --out_dir /tmp/ab720 $C720 > /tmp/ab_van720.log 2>&1 &
  CUDA_VISIBLE_DEVICES=6 python3 $RP --backend MXFP8_CUDNN --backend_tag MXnone  --gpu_id 6 --out_dir /tmp/ab720 $C720 > /tmp/ab_mxnone720.log 2>&1 &
  CUDA_VISIBLE_DEVICES=7 TRTLLM_VISUAL_GEN_MXFP8_HADAMARD=1     python3 $RP --backend MXFP8_CUDNN --backend_tag MXhad  --gpu_id 7 --out_dir /tmp/ab720 $C720 > /tmp/ab_had720.log 2>&1 &
  wait
  echo "=== all gens finished $(date) ==="
  echo "--- transform ACTIVE confirmations (must see SmoothQuant/Hadamard ACTIVE per variant) ---"
  grep -h "ACTIVE" /tmp/ab_sqk480.log /tmp/ab_sqqk480.log /tmp/ab_had480.log /tmp/ab_had720.log 2>/dev/null | sort -u
  echo "--- per-variant trace path counts (self-attn must be mxfp8, 0 fallback_exception) ---"
  for t in MXnone MXsqK MXsqQK MXhad; do
    echo -n "480 $t: "; awk '{print $3}' /tmp/ab480/prompts/traces/per_call_$t.txt 2>/dev/null | sort | uniq -c | tr '\n' ' '; echo
  done
  for t in MXnone MXhad; do
    echo -n "720 $t: "; awk '{print $3}' /tmp/ab720/prompts/traces/per_call_$t.txt 2>/dev/null | sort | uniq -c | tr '\n' ' '; echo
  done
  echo "=================== LPIPS 480p vs bf16 VANILLA ==================="
  python3 "$REPRO/control_compare.py" --dir /tmp/ab480/prompts --pairs VANILLA:MXnone VANILLA:MXsqK VANILLA:MXsqQK VANILLA:MXhad 2>&1 | grep -vE "Downloading|\|.*MB/s|warnings.warn|Setting up"
  echo "=================== LPIPS 720p vs bf16 VANILLA ==================="
  python3 "$REPRO/control_compare.py" --dir /tmp/ab720/prompts --pairs VANILLA:MXnone VANILLA:MXhad 2>&1 | grep -vE "Downloading|\|.*MB/s|warnings.warn|Setting up"
  echo "=== EXIT=$? done $(date) ==="
} 2>&1 | tee "$LOG"
