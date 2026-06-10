#!/bin/bash
# flower_blooming 720p A/B: does Hadamard (and Hadamard+SmoothQuant-QK) improve LPIPS vs bf16?
# 4 GPUs: VANILLA | MXFP8(none) | MXFP8+Hadamard | MXFP8+Hadamard+SmoothQuant-QK. No fallback.
set -o pipefail
WT=/home/scratch.liuc_coreai/codes/mxfp8-ctrl-feat
REPRO=/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/repro
SCR=/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/run_real_mxfp8/transform_ab/flower720
MODEL=/home/scratch.liuc_coreai/ckpts/Wan2.2-T2V-A14B-Diffusers
LOG=/tmp/flower_had.log
export HOME=/tmp
sudo pip install --no-deps imageio imageio-ffmpeg lpips >/dev/null 2>&1
SITE=$(python3 -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "tensorrt_llm"))')
sudo cp "$WT/tensorrt_llm/_torch/visual_gen/attention_backend/mxfp8_cudnn.py" \
        "$SITE/_torch/visual_gen/attention_backend/mxfp8_cudnn.py"
# FULL path to the driver so the script's dir (not the worktree root) is on sys.path -> installed bindings
RP="$WT/perf_bench/visual_gen/mxfp8_cudnn/run_prompt_suite.py"
C="--model_path $MODEL --out_dir /tmp/flw720 --height 720 --width 1280 --num_frames 81 --steps 40 --prompts flower_blooming"
{
  echo "=== $(date) host=$(hostname) — flower_blooming 720p A/B (Hadamard) ==="
  CUDA_VISIBLE_DEVICES=0 python3 $RP --backend VANILLA     --backend_tag VANILLA --gpu_id 0 $C > /tmp/fl_van.log   2>&1 &
  CUDA_VISIBLE_DEVICES=1 python3 $RP --backend MXFP8_CUDNN --backend_tag MXnone  --gpu_id 1 $C > /tmp/fl_none.log  2>&1 &
  CUDA_VISIBLE_DEVICES=2 TRTLLM_VISUAL_GEN_MXFP8_HADAMARD=1 \
    python3 $RP --backend MXFP8_CUDNN --backend_tag MXhad --gpu_id 2 $C > /tmp/fl_had.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 TRTLLM_VISUAL_GEN_MXFP8_HADAMARD=1 TRTLLM_VISUAL_GEN_MXFP8_SMOOTHQUANT=qk \
    python3 $RP --backend MXFP8_CUDNN --backend_tag MXhadSQ --gpu_id 3 $C > /tmp/fl_hadsq.log 2>&1 &
  wait
  echo "=== all gens finished $(date) ==="
  echo "--- transform ACTIVE confirmations ---"
  grep -h "ACTIVE" /tmp/fl_had.log /tmp/fl_hadsq.log 2>/dev/null | sort -u
  echo "--- traces (real mxfp8, 0 fallback_exception) ---"
  for t in MXnone MXhad MXhadSQ; do echo -n "$t: "; awk '{print $3}' /tmp/flw720/prompts/traces/per_call_$t.txt 2>/dev/null | sort | uniq -c | tr '\n' ' '; echo; done
  echo "=================== LPIPS flower_blooming 720p vs bf16 VANILLA ==================="
  python3 "$REPRO/control_compare.py" --dir /tmp/flw720/prompts --pairs VANILLA:MXnone VANILLA:MXhad VANILLA:MXhadSQ 2>&1 | grep -vE "Downloading|\|.*MB/s|warnings.warn|Setting up"
  echo "--- copy outputs to scratch for diff videos/persistence ---"
  mkdir -p "$SCR"; cp /tmp/flw720/prompts/*.mp4 /tmp/flw720/prompts/*.npy "$SCR/" 2>/dev/null && echo "copied to $SCR"
  echo "=== EXIT=$? done $(date) ==="
} 2>&1 | tee "$LOG"
