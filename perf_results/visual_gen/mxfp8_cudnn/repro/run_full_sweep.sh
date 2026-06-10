#!/bin/bash
# FULL SWEEP (480p): 6 variants x 10 prompts x 3 seeds, all vs bf16 VANILLA, no fallback.
# Variants: VANILLA, MXnone, MXsqK, MXsqQK, MXhad, MXhadSQ. 18 (variant,seed) processes
# scheduled in waves of 8 across the GPUs. Then per-seed LPIPS + cross-seed aggregate.
set -o pipefail
WT=/home/scratch.liuc_coreai/codes/mxfp8-ctrl-feat
REPRO=/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/repro
SCR=/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/run_real_mxfp8/full_sweep
MODEL=/home/scratch.liuc_coreai/ckpts/Wan2.2-T2V-A14B-Diffusers
LOG=/tmp/full_sweep.log
export HOME=/tmp
sudo pip install --no-deps imageio imageio-ffmpeg lpips >/dev/null 2>&1
SITE=$(python3 -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "tensorrt_llm"))')
sudo cp "$WT/tensorrt_llm/_torch/visual_gen/attention_backend/mxfp8_cudnn.py" \
        "$SITE/_torch/visual_gen/attention_backend/mxfp8_cudnn.py"
RP="$WT/perf_bench/visual_gen/mxfp8_cudnn/run_prompt_suite.py"
PROMPTS="cat_windowsill busy_street ocean_sunset clouds_timelapse dancer_jump flower_blooming drone_city_night text_hello ball_bouncing empty_room_sun"
SEEDS="42 123 7"
SPECS=(
  "VANILLA|VANILLA|"
  "MXnone|MXFP8_CUDNN|"
  "MXsqK|MXFP8_CUDNN|TRTLLM_VISUAL_GEN_MXFP8_SMOOTHQUANT=k"
  "MXsqQK|MXFP8_CUDNN|TRTLLM_VISUAL_GEN_MXFP8_SMOOTHQUANT=qk"
  "MXhad|MXFP8_CUDNN|TRTLLM_VISUAL_GEN_MXFP8_HADAMARD=1"
  "MXhadSQ|MXFP8_CUDNN|TRTLLM_VISUAL_GEN_MXFP8_HADAMARD=1 TRTLLM_VISUAL_GEN_MXFP8_SMOOTHQUANT=qk"
)
{
  echo "=== $(date) host=$(hostname) — FULL SWEEP 480p: 6 variants x 10 prompts x 3 seeds ==="
  i=0
  for s in $SEEDS; do
    for spec in "${SPECS[@]}"; do
      tag="${spec%%|*}"; r="${spec#*|}"; backend="${r%%|*}"; envs="${r#*|}"
      gpu=$((i % 8))
      echo "launch gpu=$gpu seed=$s tag=$tag env='$envs'"
      env $envs CUDA_VISIBLE_DEVICES=$gpu python3 "$RP" --backend "$backend" --backend_tag "$tag" \
        --gpu_id "$gpu" --model_path "$MODEL" --out_dir /tmp/sweep_s$s --height 480 --width 832 \
        --num_frames 33 --steps 40 --seed "$s" --prompts $PROMPTS > /tmp/sw_${tag}_s${s}.log 2>&1 &
      i=$((i + 1))
      if (( i % 8 == 0 )); then echo "--- wave barrier after $i launches $(date) ---"; wait; fi
    done
  done
  wait
  echo "=== all $i gens finished $(date) ==="
  echo "--- per-(variant,seed) prompts completed (of 10) ---"
  for s in $SEEDS; do for spec in "${SPECS[@]}"; do tag="${spec%%|*}"; echo -n "$tag/s$s=$(grep -cE 'gen=' /tmp/sw_${tag}_s${s}.log 2>/dev/null) "; done; echo; done
  echo "--- transforms fired (ACTIVE counts) ---"; grep -h ACTIVE /tmp/sw_*.log 2>/dev/null | sort | uniq -c
  echo "--- fallback_exception across ALL MXFP8 traces (MUST be 0) ---"; cat /tmp/sweep_s*/prompts/traces/per_call_MX*.txt 2>/dev/null | grep -c fallback_exception
  for s in $SEEDS; do
    echo "=================== seed $s : LPIPS vs bf16 VANILLA ==================="
    python3 "$REPRO/control_compare.py" --dir /tmp/sweep_s$s/prompts \
      --pairs VANILLA:MXnone VANILLA:MXsqK VANILLA:MXsqQK VANILLA:MXhad VANILLA:MXhadSQ 2>&1 \
      | grep -E "VANILLA  vs|MEAN"
  done
  echo "=================== AGGREGATE: per-variant mean LPIPS over 10 prompts x 3 seeds ==================="
  python3 "$REPRO/aggregate_sweep.py" /tmp/sweep_s42 /tmp/sweep_s123 /tmp/sweep_s7 2>&1 \
    | grep -vE "Downloading|\|.*MB/s|warnings.warn|Setting up"
  mkdir -p "$SCR"; for s in $SEEDS; do mkdir -p "$SCR/s$s"; cp /tmp/sweep_s$s/prompts/*.mp4 "$SCR/s$s/" 2>/dev/null; done
  echo "copied mp4s -> $SCR"
  echo "=== EXIT=$? done $(date) ==="
} 2>&1 | tee "$LOG"
