#!/bin/bash
# Stage A in the FAITHFUL study container (trtllm-mxfp8-liuc:full = feature-branch
# build, cuDNN 9.22 + TE 2.12 baked in, zero drift). Overlay ONLY my patched
# mxfp8_cudnn.py (chunked-quantize fix + deleted silent fallback), prove chunked
# quantize is bit-exact, then one 480p MXFP8 gen + trace check (must be 100% mxfp8).
set -o pipefail
WT=/home/scratch.liuc_coreai/codes/mxfp8-ctrl-feat
OUT=/tmp/run_stageA
MODEL=/home/liuc/scratch/Wan2.2-T2V-A14B-Diffusers
LOG=/tmp/stageA.log
: > "$LOG"
{
  echo "=== $(date) host=$(hostname) ==="
  echo "--- deps: imageio + lpips (cuDNN 9.22 already active in image) ---"
  pip install -q imageio imageio-ffmpeg lpips 2>&1 | tail -1
  export HOME=/tmp
  SITE=$(python3 -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "tensorrt_llm"))')
  echo "SITE=$SITE"
  DST="$SITE/_torch/visual_gen/attention_backend/mxfp8_cudnn.py"
  # container runs as non-root user liuc; site-packages is root-owned -> sudo cp
  sudo cp "$WT/tensorrt_llm/_torch/visual_gen/attention_backend/mxfp8_cudnn.py" "$DST" && echo "overlaid mxfp8_cudnn.py"
  echo "patched markers (expect >=1 each): chunk=$(grep -c _te_mxfp8_quantize_2d "$DST") nofallback=$(grep -c cross_or_unsupported_bf16 "$DST") fallback_exc_gone=$(grep -c fallback_exception "$DST")"
  python3 -c "from tensorrt_llm import VisualGen; print('IMPORT_OK')" || { echo IMPORT_FAILED; exit 3; }

  echo "=================== CHUNK-EXACTNESS TEST (B=2,S=14040) ==================="
  CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import os, torch
os.environ.setdefault("HOME", "/tmp")
from tensorrt_llm._torch.visual_gen.attention_backend import mxfp8_cudnn as m
assert m._mxfp8_supported(), "mxfp8 not supported on this device/cudnn"
torch.manual_seed(0)
x = torch.randn(2, 40, 14040, 128, dtype=torch.bfloat16, device="cuda") * 0.5
m._MAX_QUANT_ROWS = 10**12          # un-chunked
q0, sq0 = m._quantize_q_or_k_along_d(x); v0, sv0 = m._quantize_v_along_s(x)
m._MAX_QUANT_ROWS = 200_000         # chunked
q1, sq1 = m._quantize_q_or_k_along_d(x); v1, sv1 = m._quantize_v_along_s(x)
ok = (torch.equal(q0.view(torch.uint8), q1.view(torch.uint8)) and torch.equal(sq0, sq1)
      and torch.equal(v0.view(torch.uint8), v1.view(torch.uint8)) and torch.equal(sv0, sv1))
print("CHUNK_BITEXACT:", ok, "| Qd", torch.equal(q0.view(torch.uint8), q1.view(torch.uint8)),
      "Qsf", torch.equal(sq0, sq1), "Vd", torch.equal(v0.view(torch.uint8), v1.view(torch.uint8)),
      "Vsf", torch.equal(sv0, sv1))
PY

  echo "=================== 480p MXFP8 gen (1 prompt, GPU0) ==================="
  cd "$WT" || exit 4
  CUDA_VISIBLE_DEVICES=0 python3 perf_bench/visual_gen/mxfp8_cudnn/run_prompt_suite.py \
    --backend MXFP8_CUDNN --backend_tag MXFP8 --gpu_id 0 --model_path "$MODEL" --out_dir "$OUT" \
    --height 480 --width 832 --num_frames 33 --steps 40 --prompts busy_street 2>&1 \
    | grep -vE "UserWarning|Overriding|operator:|dispatch key|registered at|self.m.impl|previous kernel|new kernel|parakeet|_warnings" | tail -15
  echo "=================== per-call trace (MUST: self-attn=mxfp8, 0 fallback_exception) ==================="
  sort "$OUT/prompts/traces/per_call_MXFP8.txt" 2>/dev/null | awk '{print $3}' | sort | uniq -c
  echo "=== EXIT=$? done $(date) ==="
} 2>&1 | tee -a "$LOG"
