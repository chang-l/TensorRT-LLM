#!/bin/bash
# VANILLA-vs-VANILLA divergence floor on STOCK rc13 wheel (NO feature-branch overlay).
# Two cold VANILLA runs, identical config + seed -> run-to-run divergence floor.
# (MXFP8 arm needs a feature-branch build; deferred.)
set -o pipefail
WT=/home/scratch.liuc_coreai/codes/mxfp8-ctrl-feat
REPRO=/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/repro
WHEEL=/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/build/tensorrt_llm-1.3.0rc13-cp312-cp312-linux_x86_64.whl
OUT=/tmp/run_van_floor
MODEL=/home/liuc/scratch/Wan2.2-T2V-A14B-Diffusers
LOG=/tmp/vanfloor_run.log
PROMPTS="busy_street dancer_jump"
: > "$LOG"
{
  echo "=== $(date) host=$(hostname) ==="
  echo "--- RESTORE clean rc13 visual_gen (undo overlay) + lpips ---"
  pip install "$WHEEL" --no-deps --force-reinstall 2>&1 | tail -1
  pip install -q lpips 2>&1 | tail -1
  export HOME=/tmp
  python3 -c "from tensorrt_llm import VisualGen; print('IMPORT_OK (clean rc13)')" || { echo IMPORT_FAILED; exit 3; }
  cd "$WT" || exit 4
  COMMON="--model_path $MODEL --out_dir $OUT --height 480 --width 832 --num_frames 33 --steps 40 --gpu_id 0 --prompts $PROMPTS"
  for TAG in VAN_a VAN_b; do
    echo "=================== generate $TAG (VANILLA, cold) ==================="
    CUDA_VISIBLE_DEVICES=0 python3 perf_bench/visual_gen/mxfp8_cudnn/run_prompt_suite.py \
      --backend VANILLA --backend_tag "$TAG" $COMMON 2>&1 \
      | grep -vE "UserWarning|Overriding|operator:|dispatch key|registered at|self.m.impl|previous kernel|new kernel|parakeet|_warnings" | tail -18
  done
  echo "=================== COMPARE VAN_a vs VAN_b (DIVERGENCE FLOOR) ==================="
  CUDA_VISIBLE_DEVICES=0 python3 "$REPRO/control_compare.py" --dir "$OUT/prompts" --pairs VAN_a:VAN_b 2>&1 \
    | grep -vE "Setting up|Downloading|\|.*MB/s|warnings.warn"
  echo "=== EXIT=$? done $(date) ==="
} 2>&1 | tee -a "$LOG"
