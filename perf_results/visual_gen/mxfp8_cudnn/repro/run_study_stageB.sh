#!/bin/bash
# Stage B: REAL MXFP8 vs bf16 VANILLA, parallel across ALL 8 GPUs.
#   6 prompts split A/B; VANILLA+MXFP8 of the same (res,half) share an out dir so
#   LPIPS compares cleanly. Patched backend = NO silent fallback + chunked quantize.
#   GPU: 0 VAN720A 1 MX720A 2 VAN720B 3 MX720B 4 VAN480A 5 MX480A 6 VAN480B 7 MX480B
set -o pipefail
WT=/home/scratch.liuc_coreai/codes/mxfp8-ctrl-feat
REPRO=/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/repro
MODEL=/home/scratch.liuc_coreai/ckpts/Wan2.2-T2V-A14B-Diffusers
LOG=/tmp/stageB.log
PA="busy_street dancer_jump cat_windowsill"
PB="ocean_sunset drone_city_night flower_blooming"
export HOME=/tmp
SITE=$(python3 -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "tensorrt_llm"))')
sudo cp "$WT/tensorrt_llm/_torch/visual_gen/attention_backend/mxfp8_cudnn.py" \
        "$SITE/_torch/visual_gen/attention_backend/mxfp8_cudnn.py"
cd "$WT" || exit 4

gen() {  # gpu backend tag out_dir H W NF "prompts"
  CUDA_VISIBLE_DEVICES=$1 python3 perf_bench/visual_gen/mxfp8_cudnn/run_prompt_suite.py \
    --backend "$2" --backend_tag "$3" --gpu_id "$1" --model_path "$MODEL" --out_dir "$4" \
    --height "$5" --width "$6" --num_frames "$7" --steps 40 --prompts $8
}

{
  echo "=== $(date) host=$(hostname) — 8-GPU parallel real-MXFP8 vs bf16-VANILLA ==="
  gen 0 VANILLA     VANILLA /tmp/s720A 720 1280 81 "$PA" > /tmp/j0.log 2>&1 &
  gen 1 MXFP8_CUDNN MXFP8   /tmp/s720A 720 1280 81 "$PA" > /tmp/j1.log 2>&1 &
  gen 2 VANILLA     VANILLA /tmp/s720B 720 1280 81 "$PB" > /tmp/j2.log 2>&1 &
  gen 3 MXFP8_CUDNN MXFP8   /tmp/s720B 720 1280 81 "$PB" > /tmp/j3.log 2>&1 &
  gen 4 VANILLA     VANILLA /tmp/s480A 480 832  33 "$PA" > /tmp/j4.log 2>&1 &
  gen 5 MXFP8_CUDNN MXFP8   /tmp/s480A 480 832  33 "$PA" > /tmp/j5.log 2>&1 &
  gen 6 VANILLA     VANILLA /tmp/s480B 480 832  33 "$PB" > /tmp/j6.log 2>&1 &
  gen 7 MXFP8_CUDNN MXFP8   /tmp/s480B 480 832  33 "$PB" > /tmp/j7.log 2>&1 &
  wait
  echo "=== all 8 gens finished $(date) ==="
  for j in 0 1 2 3 4 5 6 7; do
    echo "[j$j] $(grep -E 'DONE|FAILED|Worker died|gen=' /tmp/j$j.log 2>/dev/null | tail -2 | tr '\n' ' ')"
  done
  echo "=================== 720p MXFP8 trace (A+B; MUST: mxfp8 + 0 fallback_exception) ==================="
  cat /tmp/s720A/prompts/traces/per_call_MXFP8.txt /tmp/s720B/prompts/traces/per_call_MXFP8.txt 2>/dev/null | awk '{print $3}' | sort | uniq -c
  echo "=================== 480p MXFP8 trace (A+B) ==================="
  cat /tmp/s480A/prompts/traces/per_call_MXFP8.txt /tmp/s480B/prompts/traces/per_call_MXFP8.txt 2>/dev/null | awk '{print $3}' | sort | uniq -c
  for d in s720A s720B s480A s480B; do
    echo "=================== LPIPS $d : REAL MXFP8 vs bf16 VANILLA ==================="
    python3 "$REPRO/control_compare.py" --dir /tmp/$d/prompts --pairs VANILLA:MXFP8 2>&1 | grep -vE "Downloading|\|.*MB/s|warnings.warn|Setting up"
  done
  echo "=== EXIT=$? done $(date) ==="
} 2>&1 | tee "$LOG"
