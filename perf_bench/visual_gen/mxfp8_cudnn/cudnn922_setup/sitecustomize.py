# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""sitecustomize shim to force-load cuDNN 9.22 before PyTorch imports.

When this file lives in any site-packages directory, Python executes it
automatically at interpreter startup (before any user code or torch import).
That gives us a chance to dlopen the pip-installed cuDNN 9.22 libraries with
``RTLD_GLOBAL``, so they end up in the process's global symbol table. By the
time ``import torch`` later does its own ``dlopen("libcudnn.so.9")``, the
loader sees the file is already loaded and reuses it — bypassing the bundled
container cuDNN 9.19 entirely.

Install with:

    cp sitecustomize.py $(python3 -c 'import site; print(site.getsitepackages()[0])')/

Prereq: ``pip install nvidia-cudnn-cu13==9.22.0.52``.

This is an *alternative* to ``activate.sh`` (``LD_LIBRARY_PATH``) and
``set_ld_preload.sh`` (``LD_PRELOAD``). Use it when env vars are awkward —
notebook kernels, CI runners that strip env, etc.
"""

import ctypes
import os
import sys


def _load_cudnn_922():
    try:
        import nvidia.cudnn
    except ImportError:
        return False

    # nvidia.cudnn is a namespace package — __file__ is None. Use __path__.
    cudnn_dir = list(nvidia.cudnn.__path__)[0] + "/lib"
    if not os.path.isdir(cudnn_dir):
        return False

    # Order matters: load stub last so its dependencies are already global.
    sub_libs = [
        "libcudnn_graph.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_heuristic.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_adv.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_engines_tensor_ir.so.9",
        "libcudnn_ext.so.9",
        "libcudnn.so.9",
    ]
    loaded = 0
    for so in sub_libs:
        path = os.path.join(cudnn_dir, so)
        if not os.path.exists(path):
            continue
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            loaded += 1
        except OSError as e:
            # Don't fail Python startup over a missing cuDNN sub-library.
            print(f"[sitecustomize cuDNN 9.22] warning: failed {so}: {e}", file=sys.stderr)
    return loaded > 0


# Auto-load only if explicitly opted in (so this file is safe to leave
# globally installed). To opt in, set CUDNN_922_AUTOLOAD=1 in the environment.
if os.environ.get("CUDNN_922_AUTOLOAD", "0") == "1":
    _load_cudnn_922()
