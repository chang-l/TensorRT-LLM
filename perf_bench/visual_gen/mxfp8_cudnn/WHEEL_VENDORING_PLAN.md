# Plan: vendor cuDNN 9.22 inside the TRT-LLM wheel

**Status**: draft / pre-implementation. To be reviewed by a supervisor agent
before code starts landing. See `cudnn922_setup/README.md` for the interim
LD_LIBRARY_PATH overlay used today.

## Why

TRT-LLM's `MXFP8CudnnAttention` backend (`tensorrt_llm/_torch/visual_gen/
attention_backend/mxfp8_cudnn.py`) calls `cudnn.sdpa_mxfp8` (Blackwell-only,
cuDNN ≥ 9.21), but the NGC PyTorch 26.02 container ships cuDNN 9.19. The
overlay recipe (`cudnn922_setup/activate.sh`) papers over this at runtime, but
two real costs remain:

1. **Env-var dependency** — users have to source `activate.sh` (or set
   `LD_PRELOAD`, or install `sitecustomize.py`) before running anything that
   touches MXFP8. Easy to forget. Easy to break in CI / nested shells.
2. **Container coupling** — the recipe assumes the container has cuDNN 9.22
   pip-installed at `/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/`.
   Different container, different cuDNN version, different problem.

Vendoring cuDNN inside the TRT-LLM wheel itself solves both: TRT-LLM ships a
specific cuDNN with itself, no env var needed, no container coupling.

## What "vendor inside the wheel" means concretely

After install, a TRT-LLM wheel produces this layout:

```
<site-packages>/tensorrt_llm/
├── __init__.py
├── ...
└── libs/                              ← already exists today
    ├── libtensorrt_llm.so             ← already vendored
    ├── libth_common.so                ← already vendored
    ├── ...
    ├── ucx/...                        ← already vendored
    ├── nvshmem/...                    ← already vendored
    └── cudnn/                         ← NEW
        ├── libcudnn.so.9 -> libcudnn.so.9.22.0
        ├── libcudnn.so.9.22.0
        ├── libcudnn_graph.so.9 -> libcudnn_graph.so.9.22.0
        ├── libcudnn_graph.so.9.22.0
        ├── libcudnn_ops.so.9 -> libcudnn_ops.so.9.22.0
        ├── libcudnn_ops.so.9.22.0
        ├── libcudnn_heuristic.so.9 -> libcudnn_heuristic.so.9.22.0
        ├── libcudnn_heuristic.so.9.22.0
        ├── libcudnn_engines_precompiled.so.9 -> ...
        ├── libcudnn_engines_runtime_compiled.so.9 -> ...
        ├── libcudnn_engines_tensor_ir.so.9 -> ...
        ├── libcudnn_cnn.so.9 -> ...
        ├── libcudnn_adv.so.9 -> ...
        └── libcudnn_ext.so.9 -> ...
```

Total payload: ~870 MB across 10 sub-libs (cuDNN 9.22.0.52 amd64 today).
Wheel grows accordingly. See "Wheel-size tradeoffs" below.

## Architecture (3 layers)

### Layer 1 — getting the cuDNN .so files into the build tree

Two viable mechanisms:

**Option 1a: `FetchContent` from the upstream cuDNN tarball.** Same pattern as
the UCX/NVSHMEM precedent.

```cmake
# cpp/CMakeLists.txt -- new block, around the existing FetchContent declarations
option(VENDOR_CUDNN "Bundle cuDNN runtime libraries inside the wheel" ON)
set(CUDNN_VENDORED_VERSION "9.22.0.52" CACHE STRING "cuDNN version to vendor")

if(VENDOR_CUDNN)
  include(FetchContent)
  FetchContent_Declare(
    cudnn_vendored
    URL https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VENDORED_VERSION}_cuda13-archive.tar.xz
    URL_HASH SHA256=<TO_FILL>      # pin SHA256 for supply-chain hygiene
  )
  FetchContent_MakeAvailable(cudnn_vendored)
  set(CUDNN_VENDORED_LIB_DIR ${cudnn_vendored_SOURCE_DIR}/lib)
endif()
```

**Option 1b: `pip install nvidia-cudnn-cu13==9.22.0.52` at build time, then
collect the libs out of the wheel.** Simpler and matches what `requirements*.txt`
already does for some deps. Downside: the build-host has to be able to pip
install, and we add a build-time wheel dep.

Recommend **Option 1a** for build hygiene and supply-chain transparency. Pin
the SHA256 of the archive. The downloaded artifact gets cached in
`${CMAKE_BINARY_DIR}/_deps/cudnn_vendored-src/`.

### Layer 2 — copying the libs into the wheel staging dir

```cmake
# Continue in cpp/CMakeLists.txt
if(VENDOR_CUDNN)
  set(CUDNN_SO_FILES
    libcudnn.so.9
    libcudnn_graph.so.9
    libcudnn_ops.so.9
    libcudnn_heuristic.so.9
    libcudnn_engines_precompiled.so.9
    libcudnn_engines_runtime_compiled.so.9
    libcudnn_engines_tensor_ir.so.9
    libcudnn_cnn.so.9
    libcudnn_adv.so.9
    libcudnn_ext.so.9
  )
  set(WHEEL_CUDNN_DIR ${CMAKE_SOURCE_DIR}/../tensorrt_llm/libs/cudnn)
  add_custom_target(stage_cudnn ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${WHEEL_CUDNN_DIR}
    COMMENT "Staging vendored cuDNN libs"
  )
  foreach(so IN LISTS CUDNN_SO_FILES)
    add_custom_command(TARGET stage_cudnn POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
        ${CUDNN_VENDORED_LIB_DIR}/${so}.${CUDNN_VENDORED_VERSION}
        ${WHEEL_CUDNN_DIR}/${so}.${CUDNN_VENDORED_VERSION}
      COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${so}.${CUDNN_VENDORED_VERSION}
        ${WHEEL_CUDNN_DIR}/${so}
    )
  endforeach()
endif()
```

Notes:
- `tensorrt_llm/libs/` is gitignored (built artifact), matching the existing
  pattern for the other vendored libs.
- Symlinks are created in the stage dir so the wheel ships them as-is. cuDNN's
  loader stub `libcudnn.so.9` dlopens the sub-libs by versioned name (e.g.
  `libcudnn_graph.so.9`), so the symlinks must exist for the chain to resolve.

### Layer 3 — making Python find the vendored libs at runtime

The wheel ships the libs at `<site-packages>/tensorrt_llm/libs/cudnn/`. At
runtime we need PyTorch's `libtorch_cuda.so` to load *our* `libcudnn.so.9`
instead of the system one. PyTorch loads cuDNN with a hardcoded NEEDED entry
(`libcudnn.so.9`), so the loader follows the standard search path. We have
three ways to inject ourselves into that path:

**3a — `LD_LIBRARY_PATH` mutation via a `.pth` file.** Drop a
`tensorrt_llm_cudnn_path.pth` into site-packages:

```python
# tensorrt_llm_cudnn_path.pth -- one-liner, executed by Python at startup
import os, sys, sysconfig; \
sys.path = sys.path; \
_d = os.path.join(sysconfig.get_paths()['purelib'], 'tensorrt_llm', 'libs', 'cudnn'); \
os.environ['LD_LIBRARY_PATH'] = _d + (':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
```

**Problem**: by the time Python runs the .pth, the loader may have already
cached `LD_LIBRARY_PATH`. Setting it after-the-fact does NOT retroactively
help with subsequent `dlopen` calls in many libc implementations. PyTorch's
`libtorch_cuda.so` typically loads cuDNN on first cuDNN-using call, so the
window is small but not zero.

**3b — `ctypes.CDLL(..., RTLD_GLOBAL)` preload in `tensorrt_llm/__init__.py`.**
This is the recipe we already use in `cudnn922_setup/sitecustomize.py`. Move
it to TRT-LLM's package init:

```python
# tensorrt_llm/__init__.py (top, before anything else)
def _preload_vendored_cudnn():
    import ctypes, os
    pkg_dir = os.path.dirname(__file__)
    cudnn_dir = os.path.join(pkg_dir, 'libs', 'cudnn')
    if not os.path.isdir(cudnn_dir):
        return  # not a vendored build; rely on system cuDNN
    # Order matters: load sub-libs before the stub.
    for so in ('libcudnn_graph.so.9', 'libcudnn_ops.so.9',
               'libcudnn_heuristic.so.9', 'libcudnn_cnn.so.9',
               'libcudnn_adv.so.9', 'libcudnn_engines_precompiled.so.9',
               'libcudnn_engines_runtime_compiled.so.9',
               'libcudnn_engines_tensor_ir.so.9', 'libcudnn_ext.so.9',
               'libcudnn.so.9'):
        path = os.path.join(cudnn_dir, so)
        if os.path.exists(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

_preload_vendored_cudnn()
```

When `import tensorrt_llm` happens, all 10 cuDNN .so files are loaded with
`RTLD_GLOBAL`. Any subsequent `dlopen("libcudnn.so.9")` (including PyTorch's)
sees the lib is already loaded and reuses it. **This is the correct primary
mechanism**.

Caveat: this only works if `import tensorrt_llm` happens **before** anything
that triggers PyTorch to load cuDNN (cuDNN load is lazy — first SDPA / first
convolution / first BatchNorm). In practice for visual_gen workloads,
`import tensorrt_llm` happens at the top of the script, well before any
forward pass, so this is fine. Document it anyway.

**3c — Hybrid: RPATH baked into the wheel's `libtensorrt_llm.so`.** TRT-LLM
already does this for nccl/ucx etc.:

```cmake
set_target_properties(tensorrt_llm PROPERTIES
  INSTALL_RPATH "$ORIGIN:$ORIGIN/cudnn:$ORIGIN/ucx/lib"
  BUILD_RPATH_USE_ORIGIN TRUE
)
```

This makes TRT-LLM's own C++ code find `libs/cudnn/libcudnn.so.9` directly
when it dlopens. **But** PyTorch's `libtorch_cuda.so` has its own RPATH and
doesn't know about ours, so this doesn't help PyTorch's cuDNN dispatch path.
RPATH is useful for any of TRT-LLM's own C++ tools that link cuDNN, but the
primary mechanism for the MXFP8 path is **3b**.

**Recommended combination**: **3b** (primary, covers all callers via
RTLD_GLOBAL preload) + **3c** (defense-in-depth for TRT-LLM's own C++).

## File-by-file change checklist

| File | Change | Estimated LOC |
|---|---|---|
| `cpp/CMakeLists.txt` | `FetchContent_Declare(cudnn_vendored …)` block; the `stage_cudnn` target with `add_custom_command` foreach loop; optional `INSTALL_RPATH` extension to include `$ORIGIN/cudnn`. | ~40 |
| `setup.py` | Add `'libs/cudnn/libcudnn*.so*'` to the Linux `package_data` glob (line ~130). | ~2 |
| `tensorrt_llm/__init__.py` | The `_preload_vendored_cudnn()` function shown in §3b. | ~25 |
| `tensorrt_llm/_torch/visual_gen/attention_backend/mxfp8_cudnn.py` | Drop the "graceful fallback" code path on cuDNN missing — vendored builds are guaranteed. Keep fallback for source builds with `-DVENDOR_CUDNN=OFF`. | ~5 |
| `docs/source/installation/build-from-source.md` | Update build docs: cuDNN is now vendored unless `-DVENDOR_CUDNN=OFF`. | ~5 |
| `.gitignore` | Add `tensorrt_llm/libs/cudnn/` (built artifact). | ~1 |
| `perf_bench/visual_gen/mxfp8_cudnn/cudnn922_setup/README.md` | Add a "see also: wheel-vendoring superseded by …" note once landed. | ~2 |
| `requirements*.txt` / `pyproject.toml` | **No change** — cuDNN is bundled, not a pip dep. (Optional: add `nvidia-cudnn-frontend>=9.22` if visual_gen should pull frontend automatically; currently it's a soft optional dep.) | 0 or ~3 |
| `nightly_trtllm_build.sh` | No change — `FetchContent` runs inside `cmake`, transparent to the build script. | 0 |

**Total**: ~75 LOC across 6–8 files. Mostly mechanical.

## Wheel-size tradeoffs

| Approach | Wheel size delta | Pros | Cons |
|---|---|---|---|
| Vendor full cuDNN 9.22 (all 10 libs) | +870 MB | Simple. Self-contained. Matches the surface API PyTorch expects. | Wheel is huge. Multi-GPU-arch builds duplicate this. |
| Vendor only the libs that `sdpa_mxfp8` actually touches | +~400 MB | Smaller wheel. | Need to figure out which sub-libs `sdpa_mxfp8` dlopens. Brittle to cuDNN updates. |
| Vendor only kernels needed (cubin extraction) | +~10–50 MB | Tiny. | Have to rebuild cuDNN's plan-dispatch logic ourselves — multi-month engineering project. **Not recommended**. |
| Make vendoring optional (`-DVENDOR_CUDNN=ON` default, `OFF` for users who already have 9.22+) | 0 or +870 MB | Best of both worlds. CI/end-users get the heavy wheel; devs can opt out. | More CMake conditional logic. |

**Recommended**: **vendor everything by default**, allow opt-out with
`-DVENDOR_CUDNN=OFF`. Wheel growth is acceptable for the convenience.

## Validation plan

After implementation, run the following on a **clean container with no cuDNN
9.22 preinstalled** (e.g. a fresh `nvcr.io/nvidia/pytorch:26.02-py3` with
TRT-LLM wheel `pip install`'d):

1. `python3 -c "import torch; print(torch.backends.cudnn.version())"` →
   **expect 92200** (the vendored version, not the container's 91900).
2. Run `perf_bench/visual_gen/mxfp8_cudnn/cudnn922_setup/verify.py` —
   all four checks pass.
3. Run the 4-backend microbench
   (`perf_bench/visual_gen/mxfp8_cudnn/microbench_attn_backends.py`) — numbers
   match the current symlink/LD_LIBRARY_PATH-based result within run noise.
4. Run a Wan2.2 short generation under the MXFP8_CUDNN backend; LPIPS vs bf16
   stays at 0.04x at the production shape.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Wheel size 2× larger than today | Make vendoring optional (`-DVENDOR_CUDNN=ON|OFF`). Most CI/end-users get the bundled version; size-conscious deploys opt out. |
| Vendored cuDNN goes stale | Pin `CUDNN_VENDORED_VERSION` in CMake, bump explicitly. Add a CI lint that warns if pin is > 6 months old. |
| Library mismatch between vendored cuDNN and CUDA toolkit on host | Test against current NGC PyTorch 26.0x's CUDA 13.x. cuDNN 9.22 supports CUDA 12.x and 13.x; the 9.22-`cuda13` variant is the right choice for NGC 26.x containers. |
| RTLD_GLOBAL preload conflicts with other libs that load cuDNN | We're loading the SAME `libcudnn.so.9` SONAME everyone else expects. ABI-stable within 9.x. Tested on NGC 26.02 + PyTorch 2.11. |
| `import tensorrt_llm` happens AFTER cuDNN already loaded by some earlier import | Document the import-order constraint. If a workaround is needed, see "Future work" below. |
| FetchContent download fails in offline build environments | Add `FetchContent_Declare(URL …)` fallback to a local mirror; gate behind a `CUDNN_OFFLINE_DIR` cache variable. |

## Future work

- **TE-only path**: if/when TransformerEngine's MXFP8 attention learns to call
  CuTe DSL directly (or DKG cuts a Blackwell MXFP8 SDPA kernel that doesn't go
  through cuDNN), this whole vendoring exercise becomes unnecessary. Track
  `#dlarch-fastkernels`.
- **Wheel-time fallback for missing cuDNN**: if user runs an old PyTorch that
  has cuDNN already-loaded before `import tensorrt_llm`, we can't override.
  Detect this in `_preload_vendored_cudnn()` (check `/proc/self/maps` for an
  existing libcudnn before we preload) and warn that the vendored version
  was not loaded. A clean re-import is the only fix.

## Effort estimate

| Phase | Effort |
|---|---|
| Implementation (CMake + setup.py + `__init__.py` + tests) | 1–2 days |
| Local validation (microbench + Wan smoke test) | 0.5 day |
| CI smoke-test in nightly build | 0.5 day |
| Documentation updates | 0.5 day |
| **Total** | **2–3 days end-to-end** |

Smaller than the symlink-trick maintenance burden over a year.
