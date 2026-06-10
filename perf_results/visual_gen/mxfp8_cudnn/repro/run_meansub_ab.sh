#!/bin/bash
# mean-sub K-centering A/B: MEANSUB=k vs baseline, 3 seeds x 10 prompts (480p), GPUs 0-2.
# Writes tag MXmsK into the existing /tmp/sweep_s{seed} dirs (reuses their VANILLA/MXnone refs).
set -o pipefail
WT=/home/scratch.liuc_coreai/codes/mxfp8-ctrl-feat
REPRO=/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/repro
MODEL=/home/scratch.liuc_coreai/ckpts/Wan2.2-T2V-A14B-Diffusers
export HOME=/tmp
SITE=$(python3 -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "tensorrt_llm"))')
sudo cp "$WT/tensorrt_llm/_torch/visual_gen/attention_backend/mxfp8_cudnn.py" "$SITE/_torch/visual_gen/attention_backend/mxfp8_cudnn.py"
RP="$WT/perf_bench/visual_gen/mxfp8_cudnn/run_prompt_suite.py"
PROMPTS="cat_windowsill busy_street ocean_sunset clouds_timelapse dancer_jump flower_blooming drone_city_night text_hello ball_bouncing empty_room_sun"
C="--model_path $MODEL --height 480 --width 832 --num_frames 33 --steps 40 --prompts $PROMPTS"
gen() { env TRTLLM_VISUAL_GEN_MXFP8_MEANSUB=k CUDA_VISIBLE_DEVICES=$1 python3 "$RP" --backend MXFP8_CUDNN --backend_tag MXmsK --gpu_id "$1" --out_dir /tmp/sweep_s$2 --seed "$2" $C > /tmp/sms_s$2.log 2>&1; }
{
  echo "=== $(date) mean-sub K-centering A/B (3 seeds, gpu 0-2) ==="
  gen 0 42 & gen 1 123 & gen 2 7 & wait
  echo "--- ACTIVE confirmations ---"; grep -h ACTIVE /tmp/sms_s*.log 2>/dev/null | sort -u
  echo "--- fallback_exception across MXmsK traces (MUST be 0) ---"; cat /tmp/sweep_s*/prompts/traces/per_call_MXmsK.txt 2>/dev/null | grep -c fallback_exception
  for s in 42 123 7; do echo "=== seed $s: baseline vs mean-sub-K ==="; python3 "$REPRO/control_compare.py" --dir /tmp/sweep_s$s/prompts --pairs VANILLA:MXnone VANILLA:MXmsK 2>&1 | grep -E "VANILLA  vs|MEAN"; done
  echo "=== EXIT=$? done $(date) ==="
} 2>&1 | tee /tmp/meansub_ab.log
