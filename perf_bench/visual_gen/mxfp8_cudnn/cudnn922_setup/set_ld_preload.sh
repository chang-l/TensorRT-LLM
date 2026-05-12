# shellcheck shell=bash
# Source this script for the more aggressive LD_PRELOAD path. Sets LD_PRELOAD
# to the full list of cuDNN 9.22 sub-libraries. Use only if `activate.sh`
# (which just prepends LD_LIBRARY_PATH) is insufficient — e.g. when a target
# binary's RUNPATH overrides LD_LIBRARY_PATH and you need to force-preload.
#
# Usage:
#     source /path/to/set_ld_preload.sh
#     python3 your_script.py

_cudnn_dir() {
    python3 -c 'import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0] + "/lib")' 2>/dev/null
}

CUDNN_LIB_DIR="$(_cudnn_dir)"
unset -f _cudnn_dir

if [ -z "$CUDNN_LIB_DIR" ] || [ ! -d "$CUDNN_LIB_DIR" ]; then
    echo "[cudnn922 preload] ERROR: pip-installed nvidia-cudnn not found." >&2
    return 1 2>/dev/null || exit 1
fi

# Order matters: the main libcudnn.so.9 is a stub that depends on the rest.
# List the stub first, then sub-libraries in dependency order.
PRELOAD_LIST=""
for so in libcudnn.so.9 \
          libcudnn_graph.so.9 \
          libcudnn_ops.so.9 \
          libcudnn_heuristic.so.9 \
          libcudnn_cnn.so.9 \
          libcudnn_adv.so.9 \
          libcudnn_engines_precompiled.so.9 \
          libcudnn_engines_runtime_compiled.so.9 \
          libcudnn_engines_tensor_ir.so.9 \
          libcudnn_ext.so.9; do
    if [ -e "$CUDNN_LIB_DIR/$so" ]; then
        PRELOAD_LIST="${PRELOAD_LIST:+$PRELOAD_LIST:}$CUDNN_LIB_DIR/$so"
    fi
done

if [ -z "$PRELOAD_LIST" ]; then
    echo "[cudnn922 preload] ERROR: no cuDNN .so files found in $CUDNN_LIB_DIR" >&2
    return 1 2>/dev/null || exit 1
fi

if [ -n "${LD_PRELOAD:-}" ]; then
    export LD_PRELOAD="$PRELOAD_LIST:$LD_PRELOAD"
else
    export LD_PRELOAD="$PRELOAD_LIST"
fi

echo "[cudnn922 preload] LD_PRELOAD set to $(echo "$PRELOAD_LIST" | tr ':' '\n' | wc -l) cuDNN 9.22 libraries."
echo "[cudnn922 preload] To verify: python3 -c \"import torch; print(torch.backends.cudnn.version())\"  # expect 92200"
