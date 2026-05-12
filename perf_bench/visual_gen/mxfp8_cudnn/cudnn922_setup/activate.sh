# shellcheck shell=bash
# Source this script (do not execute) to prepend pip-installed cuDNN 9.22 to
# the dynamic linker search path, overriding the container's bundled cuDNN
# 9.19. Required for cudnn.sdpa_mxfp8 (Blackwell sm_100+, cuDNN >= 9.21).
#
# Usage:
#     source /path/to/activate.sh
#     python3 your_script.py     # torch.backends.cudnn.version() -> 92200
#
# Prereq: pip install nvidia-cudnn-cu13==9.22.0.52

_cudnn_dir() {
    # nvidia.cudnn is a namespace package (no __init__.py), so __file__ is None.
    # Use __path__ to locate the install root, then append /lib.
    python3 -c 'import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0] + "/lib")' 2>/dev/null
}

CUDNN_LIB_DIR="$(_cudnn_dir)"
unset -f _cudnn_dir

if [ -z "$CUDNN_LIB_DIR" ] || [ ! -d "$CUDNN_LIB_DIR" ]; then
    echo "[cudnn922 activate] ERROR: pip-installed nvidia-cudnn not found." >&2
    echo "[cudnn922 activate] Install with: pip install nvidia-cudnn-cu13==9.22.0.52" >&2
    return 1 2>/dev/null || exit 1
fi

if [ ! -e "$CUDNN_LIB_DIR/libcudnn.so.9" ]; then
    echo "[cudnn922 activate] ERROR: $CUDNN_LIB_DIR/libcudnn.so.9 missing." >&2
    return 1 2>/dev/null || exit 1
fi

# Prepend to LD_LIBRARY_PATH so the loader finds 9.22 before /usr/lib/.
case ":${LD_LIBRARY_PATH:-}:" in
    *":$CUDNN_LIB_DIR:"*) ;;  # already present, no-op
    *) export LD_LIBRARY_PATH="$CUDNN_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

echo "[cudnn922 activate] LD_LIBRARY_PATH prepended with $CUDNN_LIB_DIR"
echo "[cudnn922 activate] To verify: python3 -c \"import torch; print(torch.backends.cudnn.version())\"  # expect 92200"
