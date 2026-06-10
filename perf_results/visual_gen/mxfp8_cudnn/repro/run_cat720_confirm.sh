#!/bin/bash
# Dedicated PROOF: cat_windowsill ONLY, 720p MXFP8, with its own per-call trace.
# Patched backend (no silent fallback + chunked). Self-attn must be 100% mxfp8, 0 exceptions.
set -o pipefail
WT=/home/scratch.liuc_coreai/codes/mxfp8-ctrl-feat
MODEL=/home/scratch.liuc_coreai/ckpts/Wan2.2-T2V-A14B-Diffusers
LOG=/tmp/cat720.log
: > "$LOG"
{
  echo "=== $(date) host=$(hostname) — cat_windowsill 720p MXFP8 dedicated trace ==="
  sudo pip install --no-deps imageio imageio-ffmpeg lpips 2>&1 | tail -1
  export HOME=/tmp
  SITE=$(python3 -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "tensorrt_llm"))')
  sudo cp "$WT/tensorrt_llm/_torch/visual_gen/attention_backend/mxfp8_cudnn.py" \
          "$SITE/_torch/visual_gen/attention_backend/mxfp8_cudnn.py" && echo "overlaid patched backend"
  grep -c fallback_exception "$SITE/_torch/visual_gen/attention_backend/mxfp8_cudnn.py" | sed 's/^/backend fallback_exception count (want 0): /'
  cd "$WT" || exit 4
  CUDA_VISIBLE_DEVICES=0 python3 perf_bench/visual_gen/mxfp8_cudnn/run_prompt_suite.py \
    --backend MXFP8_CUDNN --backend_tag MXFP8 --gpu_id 0 --model_path "$MODEL" --out_dir /tmp/catonly720 \
    --height 720 --width 1280 --num_frames 81 --steps 40 --prompts cat_windowsill 2>&1 \
    | grep -iE "gen=|DONE|worker died|error|Traceback|cat_windowsill" \
    | grep -ivE "pynvml|Requests|warnings|Skipping|parakeet" | tail -6
  echo "=================== cat_windowsill 720p per-call trace (THIS PROMPT ONLY) ==================="
  awk '{print $3}' /tmp/catonly720/prompts/traces/per_call_MXFP8.txt 2>/dev/null | sort | uniq -c
  echo "fallback_exception lines: $(grep -c fallback_exception /tmp/catonly720/prompts/traces/per_call_MXFP8.txt 2>/dev/null)"
  echo "mp4: $(ls -la /tmp/catonly720/prompts/cat_windowsill_MXFP8.mp4 2>/dev/null | awk '{print $5}')"
  echo "=== EXIT=$? done $(date) ==="
} 2>&1 | tee "$LOG"
