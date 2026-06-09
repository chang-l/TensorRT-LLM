#!/bin/bash
# Control experiment: separate FP8 error from diffusion trajectory divergence.
# Runs INSIDE the release container. Generates to /tmp (NFS root-squash blocks
# root writes to scratch); compare runs in-container too. mp4s docker-cp'd out after.
set -o pipefail
WT=/home/scratch.liuc_coreai/codes/mxfp8-ctrl-feat
REPRO=/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/repro
OUT=/tmp/run_control_480p
MODEL=/home/liuc/scratch/Wan2.2-T2V-A14B-Diffusers
LOG=/tmp/control_run.log
PROMPTS="busy_street dancer_jump cat_windowsill"
: > "$LOG"
{
  echo "=== $(date) host=$(hostname) ==="
  echo "--- deps: cuDNN 9.22 overlay (for MXFP8) + lpips ---"
  pip install -q nvidia-cudnn-cu13==9.22.0.52 nvidia-cudnn-frontend==1.23.0 2>&1 | tail -1
  pip install -q lpips 2>&1 | tail -1
  CUDNN_LIB=$(python3 -c 'import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))')
  export LD_LIBRARY_PATH="$CUDNN_LIB:$LD_LIBRARY_PATH" HOME=/tmp

  echo "--- overlay feature-branch visual_gen onto installed trtllm ---"
  SITE=$(python3 -c 'import tensorrt_llm, os; print(os.path.dirname(tensorrt_llm.__file__))')
  echo "SITE=$SITE  trtllm=$(python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)')"
  cp -r "$WT/tensorrt_llm/_torch/visual_gen/." "$SITE/_torch/visual_gen/" && echo "overlaid _torch/visual_gen"
  cp -r "$WT/tensorrt_llm/visual_gen/." "$SITE/visual_gen/" && echo "overlaid visual_gen"

  echo "--- smoke test import ---"
  python3 -c "from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams; print('IMPORT_OK')" \
    || { echo "IMPORT_FAILED -> needs source build"; exit 3; }

  cd "$WT" || exit 4
  COMMON="--model_path $MODEL --out_dir $OUT --height 480 --width 832 --num_frames 33 --steps 40 --gpu_id 0 --prompts $PROMPTS"
  for tag_backend in "VAN_a VANILLA" "VAN_b VANILLA" "MXFP8 MXFP8_CUDNN"; do
    set -- $tag_backend; TAG=$1; BK=$2
    echo "=================== generate $TAG ($BK) ==================="
    CUDA_VISIBLE_DEVICES=0 python3 perf_bench/visual_gen/mxfp8_cudnn/run_prompt_suite.py \
      --backend "$BK" --backend_tag "$TAG" $COMMON 2>&1 | grep -vE "UserWarning|Overriding|operator:|dispatch key|registered at|self.m.impl|previous kernel|new kernel" | tail -20
  done

  echo "=================== per-call trace check (MXFP8 actually ran?) ==================="
  sort "$OUT/prompts/traces/per_call_MXFP8.txt" 2>/dev/null | awk '{print $3}' | sort | uniq -c

  echo "=================== COMPARE (floor vs FP8) ==================="
  CUDA_VISIBLE_DEVICES=0 python3 "$REPRO/control_compare.py" --dir "$OUT/prompts" \
    --pairs VAN_a:VAN_b VAN_a:MXFP8 2>&1 | grep -vE "Setting up|Loading model|/usr/local"
  echo "=== EXIT=$? done $(date) ==="
} 2>&1 | tee -a "$LOG"
