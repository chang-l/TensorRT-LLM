#!/bin/bash
# Run inside the release:1.3.0rc6 container. Installs cuDNN 9.22 overlay, then
# runs the MXFP8 fallback repro. Output tee'd to a log on scratch (readable from
# the dev box without re-ssh).
set -o pipefail
REPRO_DIR=/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/repro
LOG=/tmp/live_run.log   # container-local: NFS root-squash blocks root writes to the 46646-owned scratch dir
: > "$LOG"
{
  echo "=== $(date) host=$(hostname) ==="
  echo "--- installing cuDNN 9.22 + frontend 1.23 ---"
  pip install -q nvidia-cudnn-cu13==9.22.0.52 nvidia-cudnn-frontend==1.23.0 2>&1 | tail -2
  CUDNN_LIB=$(python3 -c 'import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))')
  echo "CUDNN_LIB=$CUDNN_LIB"
  export LD_LIBRARY_PATH="$CUDNN_LIB:$LD_LIBRARY_PATH"
  export HOME=/tmp
  export CUDA_VISIBLE_DEVICES=0
  echo "--- running repro ---"
  python3 "$REPRO_DIR/repro_mxfp8_fallback.py"
  echo "=== EXIT=$? done $(date) ==="
} 2>&1 | tee -a "$LOG"
