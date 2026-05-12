# cuDNN 9.22 setup for TRT-LLM visual_gen MXFP8 path

The NGC PyTorch 26.02 container is pinned to cuDNN 9.19, but
`cudnn.sdpa_mxfp8` requires cuDNN ≥ 9.21 (Blackwell sm_100+ feature).
This directory packages a recipe for overlaying cuDNN 9.22 onto an
otherwise-stock 26.02 container *without* modifying any system files.

The previous (fragile) workaround was to replace
`/usr/lib/x86_64-linux-gnu/libcudnn_*.so.9.19.0` with symlinks to pip-installed
9.22 binaries. That works but requires root, persists across container restarts
only because docker layered filesystem keeps the mutations, and is invisible to
anyone reading the Dockerfile or environment.

This recipe replaces the symlinks with **environment-variable-driven library
overlay** (`LD_LIBRARY_PATH` / `LD_PRELOAD`), which is the canonical pattern
used by JAX, ONNXRuntime, and the cuDNN frontend project itself.

## Quick start

```bash
# 1. Install cuDNN 9.22 via pip (one-time, persists in image)
pip install nvidia-cudnn-cu13==9.22.0.52 nvidia-cudnn-frontend==1.23.0

# 2. Source the env helper before running any cudnn-using code
source /code/tensorrt_llm/perf_bench/visual_gen/mxfp8_cudnn/cudnn922_setup/activate.sh

# 3. Run as normal — torch.backends.cudnn.version() will report 92200
python3 your_script.py
```

For containers where you want the override to apply automatically to every
Python process without manual `source`, install the sitecustomize.py shim:

```bash
# Optionally, install sitecustomize so cuDNN 9.22 loads before torch
python3 -m site --user-base   # gets the user-site-packages dir
cp sitecustomize.py $(python3 -c 'import site; print(site.getsitepackages()[0])')/
```

## How it works

`activate.sh` prepends the pip-installed cuDNN 9.22 lib dir to
`LD_LIBRARY_PATH`. The dynamic loader searches `LD_LIBRARY_PATH` *before*
`/usr/lib/x86_64-linux-gnu/` (the directory the bundled 9.19 lives in / the
symlink trick targeted), so any `dlopen("libcudnn.so.9")` — including the one
PyTorch makes from its `libtorch_cuda.so` — resolves to 9.22.

`sitecustomize.py` does the same thing via `ctypes.CDLL(..., RTLD_GLOBAL)` at
Python startup. Useful when you can't set environment variables (e.g., a
notebook kernel or a CI runner that strips env vars).

`set_ld_preload.sh` is the more aggressive variant: it computes
`LD_PRELOAD` as a colon-separated list of every cuDNN 9.22 sub-library. Use
this only if `LD_LIBRARY_PATH` alone is insufficient for your environment
(rare, but happens with some weird RUNPATH configurations).

## Files

| File | Purpose |
|---|---|
| `activate.sh` | `source` this to set `LD_LIBRARY_PATH`. Recommended default. |
| `set_ld_preload.sh` | Aggressive: sets `LD_PRELOAD` to all 10 cuDNN .so files. |
| `sitecustomize.py` | Auto-load shim. Install into a site-packages dir if env vars are problematic. |
| `verify.py` | Sanity script — confirms cuDNN 9.22 is loaded, runs a tiny sdpa_mxfp8 call. |
| `README.md` | This file. |

## Removing the legacy symlinks (optional cleanup)

If you previously applied the symlink trick and want to switch fully to the
env-driven approach:

```bash
# Inside the container, as root:
for f in /usr/lib/x86_64-linux-gnu/libcudnn*.so*; do
  if readlink -f "$f" | grep -q '/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/'; then
    sudo rm "$f"
  fi
done
sudo ldconfig
```

You don't *have* to remove the symlinks — `LD_LIBRARY_PATH` takes precedence
over standard search paths, so the env-driven approach works either way.
Removing them just reduces the number of moving parts you need to remember.

## Why not the symlink trick?

| Aspect | Symlink trick | Env-var overlay |
|---|---|---|
| Requires root | yes (`/usr/lib/` mutation) | no |
| Survives `docker run --rm`? | only if baked into image | yes (env in Dockerfile) |
| Visible in container config | no (filesystem state) | yes (Dockerfile `ENV`) |
| Affects non-Python tools | yes (everyone gets new cuDNN) | yes (same — env vars are global) |
| Affects only target Python | no | yes if scoped to a single shell |
| Risk of breaking system tools | yes (e.g. `apt-get install libcudnn9` re-overwrites) | no |
| Canonical for "ship newer cuDNN in older container" | no | yes (JAX, ORT, cuDNN frontend docs all use it) |

## Long-term: vendor inside the wheel

For a fully-self-contained solution, see
`../WHEEL_VENDORING_PLAN.md` — the plan to ship cuDNN 9.22 inside the
TRT-LLM wheel itself (following the existing UCX/NVSHMEM precedent).
That removes the need for *any* env var or symlink, at the cost of a
~150 MB wheel growth.
