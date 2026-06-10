#!/bin/bash
# Fill the 8 sweep combos that OOM'd on GPUs 4-7 (taken by another user). Run on GPUs
# 0-3 only (2 waves of 4), writing into the existing /tmp/sweep_s{42,123,7} dirs, then
# re-aggregate the now-complete 6-variant x 10-prompt x 3-seed matrix.
set -o pipefail
WT=/home/scratch.liuc_coreai/codes/mxfp8-ctrl-feat
REPRO=/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/repro
SCR=/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/run_real_mxfp8/full_sweep
MODEL=/home/scratch.liuc_coreai/ckpts/Wan2.2-T2V-A14B-Diffusers
LOG=/tmp/sweep_fill.log
export HOME=/tmp
SITE=$(python3 -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "tensorrt_llm"))')
sudo cp "$WT/tensorrt_llm/_torch/visual_gen/attention_backend/mxfp8_cudnn.py" \
        "$SITE/_torch/visual_gen/attention_backend/mxfp8_cudnn.py"
RP="$WT/perf_bench/visual_gen/mxfp8_cudnn/run_prompt_suite.py"
PROMPTS="cat_windowsill busy_street ocean_sunset clouds_timelapse dancer_jump flower_blooming drone_city_night text_hello ball_bouncing empty_room_sun"
C="--model_path $MODEL --height 480 --width 832 --num_frames 33 --steps 40 --prompts $PROMPTS"
gen() {  # gpu seed tag backend "env"
  env $5 CUDA_VISIBLE_DEVICES=$1 python3 "$RP" --backend "$4" --backend_tag "$3" --gpu_id "$1" \
    --out_dir /tmp/sweep_s$2 --seed "$2" $C > /tmp/swf_${3}_s${2}.log 2>&1
}
{
  echo "=== $(date) — fill 8 missing combos on GPUs 0-3 (4-7 busy) ==="
  echo "--- wave 1 ---"
  gen 0 42  MXhad   MXFP8_CUDNN "TRTLLM_VISUAL_GEN_MXFP8_HADAMARD=1" &
  gen 1 42  MXhadSQ MXFP8_CUDNN "TRTLLM_VISUAL_GEN_MXFP8_HADAMARD=1 TRTLLM_VISUAL_GEN_MXFP8_SMOOTHQUANT=qk" &
  gen 2 123 VANILLA VANILLA     "" &
  gen 3 123 MXnone  MXFP8_CUDNN "" &
  wait
  echo "--- wave 2 ---"
  gen 0 7 VANILLA VANILLA     "" &
  gen 1 7 MXnone  MXFP8_CUDNN "" &
  gen 2 7 MXsqK   MXFP8_CUDNN "TRTLLM_VISUAL_GEN_MXFP8_SMOOTHQUANT=k" &
  gen 3 7 MXsqQK  MXFP8_CUDNN "TRTLLM_VISUAL_GEN_MXFP8_SMOOTHQUANT=qk" &
  wait
  echo "=== fill done $(date) ==="
  echo "--- fallback_exception across all MXFP8 traces (MUST be 0) ---"; cat /tmp/sweep_s*/prompts/traces/per_call_MX*.txt 2>/dev/null | grep -c fallback_exception
  for s in 42 123 7; do
    echo "=================== seed $s : LPIPS vs bf16 VANILLA ==================="
    python3 "$REPRO/control_compare.py" --dir /tmp/sweep_s$s/prompts \
      --pairs VANILLA:MXnone VANILLA:MXsqK VANILLA:MXsqQK VANILLA:MXhad VANILLA:MXhadSQ 2>&1 | grep -E "VANILLA  vs|MEAN"
  done
  echo "=================== AGGREGATE: per-variant mean LPIPS over 10 prompts x 3 seeds ==================="
  python3 "$REPRO/aggregate_sweep.py" /tmp/sweep_s42 /tmp/sweep_s123 /tmp/sweep_s7 2>&1 \
    | grep -vE "Downloading|\|.*MB/s|warnings.warn|Setting up"
  mkdir -p "$SCR"; for s in 42 123 7; do mkdir -p "$SCR/s$s"; cp /tmp/sweep_s$s/prompts/*.mp4 "$SCR/s$s/" 2>/dev/null; done
  echo "=== EXIT=$? done $(date) ==="
} 2>&1 | tee "$LOG"
