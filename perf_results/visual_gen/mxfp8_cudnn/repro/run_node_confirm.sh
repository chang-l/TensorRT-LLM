#!/bin/bash
# Quick env confirm in a study container: deps (--no-deps, don't touch cuBLAS),
# overlay patched backend, one 480p MXFP8 gen + trace. Used to test whether a node
# can run the diffusion worker (worker-CUDA-init issue seen on 043).
set -o pipefail
WT=/home/scratch.liuc_coreai/codes/mxfp8-ctrl-feat
MODEL=/home/scratch.liuc_coreai/ckpts/Wan2.2-T2V-A14B-Diffusers
LOG=/tmp/confirm.log
: > "$LOG"
{
  echo "=== $(date) host=$(hostname) ==="
  sudo pip install --no-deps imageio imageio-ffmpeg lpips 2>&1 | tail -1
  export HOME=/tmp
  SITE=$(python3 -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "tensorrt_llm"))')
  sudo cp "$WT/tensorrt_llm/_torch/visual_gen/attention_backend/mxfp8_cudnn.py" \
          "$SITE/_torch/visual_gen/attention_backend/mxfp8_cudnn.py" && echo "overlaid patched backend"
  echo "--- 480p MXFP8 gen (1 prompt) ---"
  cd "$WT" || exit 4
  python3 perf_bench/visual_gen/mxfp8_cudnn/run_prompt_suite.py \
    --backend MXFP8_CUDNN --backend_tag MXFP8 --gpu_id 0 --model_path "$MODEL" --out_dir /tmp/confirm480 \
    --height 480 --width 832 --num_frames 33 --steps 40 --prompts busy_street 2>&1 \
    | grep -iE "gen=|DONE|worker died|CUBLAS|CUDNN_STATUS|error|Traceback|busy_street" \
    | grep -ivE "pynvml|Requests|warnings|Skipping|parakeet" | tail -8
  echo "--- trace ---"; sort /tmp/confirm480/prompts/traces/per_call_MXFP8.txt 2>/dev/null | awk '{print $3}' | sort | uniq -c
  echo "--- mp4 ---"; ls -la /tmp/confirm480/prompts/*.mp4 2>/dev/null | awk '{print $5,$NF}'
  echo "=== EXIT=$? done $(date) ==="
} 2>&1 | tee "$LOG"
