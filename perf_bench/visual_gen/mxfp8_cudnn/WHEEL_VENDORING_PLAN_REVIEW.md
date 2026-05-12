# Review: WHEEL_VENDORING_PLAN.md

Reviewer hat: release-engineering. Tone: blunt. The plan reads like a junior
who learned about `FetchContent` last week and is excited to use it. Let me
strip it back to load-bearing parts.

## 1. Is vendoring even the right call? Mostly **no**.

The plan never honestly weighs cost vs benefit. Let me do it:

- **Wheel-size**: TRT-LLM's Linux wheel is ~1.0 GB today. Adding 870 MB
  *doubles it*. Every end-user `pip install tensorrt-llm` from a slow link
  (anyone outside a hyperscaler datacenter) pays this cost. Even the plan's
  own table calls this out and then waves it away with "acceptable for the
  convenience". It is not acceptable for a feature that, at present, serves
  exactly **one** code path: `MXFP8CudnnAttention` for Wan visual_gen.
- **Lifespan**: NGC PyTorch 26.04 ships cuDNN 9.21+. That release is on the
  order of weeks away. After that, every supported container already has a
  good-enough cuDNN. We would be carrying 870 MB of dead weight for the next
  N years to solve a problem that exists in *one* container tag.
- **"Users source activate.sh" vs "users download a 2x wheel"**: the interim
  recipe is one `source` line. The 870 MB tax is paid forever, by every user,
  on every install. The convenience trade is upside-down.
- **The plan dismisses bumping NGC to 26.04 with a hand-wave.** That is the
  actual right answer for 95% of consumers. A one-line Dockerfile bump in
  whatever image we publish (and a README note for self-hosted users)
  eliminates the entire problem with zero wheel-size impact, zero CMake
  changes, and zero runtime preload hackery.

The interim `cudnn922_setup/activate.sh` recipe is already working and is
canonical (JAX/ORT do the same thing). It is not "fragile"; the plan
overstates how often users forget to source it. The pip-install + LD path
overlay should remain the documented escape hatch for users stuck on
26.02 until NGC 26.04 lands.

## 2. If we do vendor anyway, the plan is overengineered.

Cut, in order:

- **Drop `FetchContent`.** UCX needed FetchContent because we *build* UCX
  from source. We do not build cuDNN; it is a redistributable tarball.
  `pip install nvidia-cudnn-cu13==X` at build time (we already pip-install
  things in our Docker build) and copy the libs out of the resulting
  `site-packages/nvidia/cudnn/lib/`. That is 5 lines in the build script
  vs. ~40 lines of CMake plus an SHA256 we now have to babysit.
- **Drop the `.pth` file mechanism (3a).** The plan itself admits it
  doesn't work reliably. Don't list non-working options.
- **Drop the RPATH layer (3c).** PyTorch's `libtorch_cuda.so` opens cuDNN,
  not ours. RPATH on `libtensorrt_llm.so` does not help the only caller
  we care about. "Defense in depth" with a mechanism that *cannot* defend
  the actual attack surface is just noise.
- **Keep only the RTLD_GLOBAL preload (3b)**, exactly as
  `cudnn922_setup/sitecustomize.py` already does it. That code is committed
  and validated end-to-end. Lift it verbatim into `tensorrt_llm/__init__.py`,
  gated on `os.path.isdir(<vendored cudnn dir>)`. Done.
- **Vendored lib set**: ten `.so` files is probably more than needed.
  `sdpa_mxfp8` is in the `_graph` + `_engines_*` + `_ops` path. `_cnn`,
  `_adv` are convolution/RNN paths. A quick `LD_DEBUG=libs` trace on a
  microbench run gives the actual closure; expect ~5 libs, not 10. That
  alone halves the payload.

## 3. Risks the plan understates or omits.

- **aarch64**: TRT-LLM ships an aarch64 wheel (Grace/GB200). The plan
  pulls `linux-x86_64-...-archive.tar.xz` with no `if(CMAKE_HOST_ARCH ...)`
  branching. Shipping x86_64 binaries inside the aarch64 wheel is a silent
  ImportError on every Grace/GB200 box. This must be conditional, and now
  we are vendoring two architectures and managing two SHA256s.
- **manylinux**: `auditwheel repair` will see vendored .so files with
  glibc symbol versions newer than `manylinux_2_28` and refuse the wheel.
  Either we ship `--plat linux_x86_64` (loses PyPI compatibility) or we
  start excluding our own vendored cuDNN from auditwheel, which is
  documented but adds CI surface.
- **CUDA major-version skew**: cuDNN 9.22 has separate `cuda12` and
  `cuda13` redists. NGC 26.02 is CUDA 13.x; NGC 25.x is CUDA 12.x. If a
  user installs the wheel on a CUDA-12 host, our vendored `cuda13` cuDNN
  is wrong. The interim pip-install recipe sidesteps this because pip
  resolves the right variant; vendoring locks us in.
- **PyTorch minor-version asserts**: PyTorch builds against a specific
  cuDNN minor and asserts `cudnnGetVersion() >= compile-time`. PyTorch
  in NGC 26.02 was built against 9.19. Forcing the process to use 9.22
  via RTLD_GLOBAL is fine in the *interim* (verified on the bench), but
  this is a global mutation that will surprise the next person who
  imports `tensorrt_llm` from a script that also uses PyTorch's own
  conv/BN paths.
- **Pip install time**: doubling wheel size doubles the
  download-and-unpack tail on slow links. For a wheel users install
  exactly once this is annoying; for CI that nukes site-packages each
  run it is real wall time.

## 4. Concrete simplified checklist

If we *must* land this, here is the minimum diff:

| File | Change |
|---|---|
| `tensorrt_llm/__init__.py` | Copy the `_load_cudnn_922()` body out of `cudnn922_setup/sitecustomize.py`. Point it at `<pkg>/libs/cudnn/lib`. Gate on directory existence. ~20 LOC. |
| `setup.py` | Add `'libs/cudnn/**/*'` to the Linux `package_data` list. Add an `if platform.machine() == 'aarch64': raise` guard or simply skip including cuDNN on aarch64. ~3 LOC. |
| `nightly_trtllm_build.sh` (or the docker build) | One `pip install --target <stage>/libs/cudnn nvidia-cudnn-cuXX==9.22.0.52` and a `cp` into the wheel staging dir. ~5 LOC. No CMake changes at all. |
| `.gitignore` | `tensorrt_llm/libs/cudnn/`. |
| `mxfp8_cudnn.py` | **No change.** Existing fallback already handles missing-cuDNN gracefully. |

Total: ~30 LOC, one shell change, zero new CMake. Compare to the plan's
~75 LOC across CMake/setup/init.

## 5. Recommendation

**Skip the vendoring exercise. Bump the base NGC image to 26.04 in our
Dockerfile/CI when it ships, and keep `cudnn922_setup/activate.sh` as the
documented one-liner for anyone stuck on 26.02 until then.** If somebody
insists on vendoring before 26.04 lands, do the 30-LOC version above —
not the FetchContent + RPATH + .pth + RTLD_GLOBAL tower the plan
proposes. Almost nothing in the plan's "Architecture (3 layers)" is
load-bearing.

VERDICT: scrap-and-go-with-different-approach
