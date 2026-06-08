# How to reproduce the MXFP8 cuDNN SDPA accuracy study (Wan2.2-T2V-A14B, B200)

This guide reproduces the LPIPS/PSNR/SSIM numbers in
[REPORT.html](https://sc.talos.nvidia.com/view/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/REPORT.html)
and
[REPORT_PROMPTS.html](https://sc.talos.nvidia.com/view/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/REPORT_PROMPTS.html).

## ⚠️ Read this first — the two most common repro failures

**1. The "MXFP8_CUDNN" backend IS cuDNN's `sdpa_mxfp8`. If you're not using
cuDNN, you are not running the same thing I measured.**

- This study's MXFP8 path is **cuDNN ≥ 9.21's `cudnn.sdpa_mxfp8`** (a
  Blackwell sm_100+ kernel). It is *not* TransformerEngine MXFP8 GEMM, not a
  Triton/CUTLASS kernel, not `torch` anything. The backend lives in
  `tensorrt_llm/_torch/visual_gen/attention_backend/mxfp8_cudnn.py`
  (on the `feature/mxfp8-sage-accuracy-study` branch — **not on upstream
  main**).
- If your container has cuDNN < 9.21 (NGC PyTorch 26.02 ships **9.19**), the
  backend **silently falls back to bf16 SDPA** — so you'd measure LPIPS ≈ 0
  (identical to reference) and conclude "no effect", OR you'd be measuring
  some *other* MXFP8 implementation and get different numbers. Both look like
  "can't reproduce."
- **Verify which cuDNN is loaded** before trusting any number (see Step 2).

**2. The backend + benchmark scripts are on a feature branch, not main.**

- Branch: `feature/mxfp8-sage-accuracy-study` on the local worktree
  `/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2`, mirrored to fork
  `git@github.com:chang-l/TensorRT-LLM.git`.
- If your colleague checked out upstream `main`, neither the `MXFP8_CUDNN`
  visual-gen backend nor `perf_bench/visual_gen/mxfp8_cudnn/` exists.

---

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA B200 (sm_100 / cc 10.0). MXFP8 SDPA is **Blackwell-only**; will not run on Hopper/Ampere. |
| Base container | NGC PyTorch 26.02 (`nvcr.io/nvidia/pytorch:26.02-py3`-derived TRT-LLM jenkins image) — ships cuDNN **9.19** |
| cuDNN needed | **≥ 9.21** for `sdpa_mxfp8`; this study used **9.22.0.52** |
| cuDNN frontend | `nvidia-cudnn-frontend==1.23.0` |
| TransformerEngine | 2.12.0 (used only to emit the F8_128x4 swizzled MXFP8 scale layout) |
| Model | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (40 layers, H=40, D=128) |
| Code branch | `feature/mxfp8-sage-accuracy-study` |

---

## Step 1 — Get the code + container

```bash
# On a B200 host (e.g. umbriel-b200-043), from the worktree that has the branch:
cd /home/scratch.liuc_coreai/codes/trtllm-v3-wt-2
git checkout feature/mxfp8-sage-accuracy-study

# Start a fresh jenkins container built from this repo (matches torch/TRT ABI):
make -C docker jenkins_run LOCAL_USER=1 \
  DOCKER_RUN_ARGS="--cap-add=SYS_PTRACE --cap-add=SYS_ADMIN \
    -v /home/scratch.trt_llm_data_ci:/home/scratch.trt_llm_data_ci:ro \
    -v /home/liuc/scratch/:/home/liuc/scratch \
    -p 8000:8000 --network=host"
```

Inside the container the repo is mounted at `/code/tensorrt_llm`.

---

## Step 2 — Install + activate cuDNN 9.22 (THE critical step)

The base container has cuDNN 9.19. `sdpa_mxfp8` needs ≥ 9.21. Overlay 9.22
via pip + `LD_LIBRARY_PATH` (canonical, no root, no system-file mutation):

```bash
# Inside the container:
pip install nvidia-cudnn-cu13==9.22.0.52 nvidia-cudnn-frontend==1.23.0

# Prepend the pip cuDNN 9.22 lib dir so the loader finds it before the
# container's bundled 9.19. (This is the recommended of three modes — see
# perf_bench/visual_gen/mxfp8_cudnn/cudnn922_setup/README.md for LD_PRELOAD
# and sitecustomize.py alternatives.)
source /code/tensorrt_llm/perf_bench/visual_gen/mxfp8_cudnn/cudnn922_setup/activate.sh
```

**Verify the overlay actually took effect** — this is the step that catches
"I'm secretly still on 9.19" and "I'm secretly falling back to bf16":

```bash
HOME=/tmp CUDA_VISIBLE_DEVICES=0 \
  python3 /code/tensorrt_llm/perf_bench/visual_gen/mxfp8_cudnn/cudnn922_setup/verify.py
```

Expected: all 4 checks pass —
1. `torch.backends.cudnn.version()` → **92200** (NOT 91900)
2. forced cuDNN SDPA runs
3. `/proc/self/maps` shows the loaded `libcudnn.so.9` is from the **pip dir**
   (`.../nvidia/cudnn/lib/`), not `/usr/lib/x86_64-linux-gnu/`
4. a tiny `sdpa_mxfp8` graph builds

If check 1 shows 91900 → the overlay didn't take; re-`source activate.sh`.
If check 4 fails → cudnn-frontend missing or HW isn't Blackwell.

---

## Step 3 — Generate videos for each backend

Driver: `perf_bench/visual_gen/mxfp8_cudnn/run_prompt_suite.py`. It loads
`VisualGen` once per backend, iterates the 10-prompt suite, and writes
`<out_dir>/prompts/{prompt}_{tag}.{mp4,npy,json}`. The `.npy` (raw uint8
frames) is what the metrics script consumes.

Set a common output dir and a model path:

```bash
MODEL=/home/liuc/scratch/Wan2.2-T2V-A14B-Diffusers   # or HF id
OUT=/home/liuc/scratch/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/run_720p_81f
cd /code/tensorrt_llm
```

**Reference (bf16 VANILLA):**
```bash
HOME=/tmp CUDA_VISIBLE_DEVICES=0 python3 perf_bench/visual_gen/mxfp8_cudnn/run_prompt_suite.py \
  --backend VANILLA --backend_tag VANILLA --gpu_id 0 \
  --model_path $MODEL --out_dir $OUT \
  --height 720 --width 1280 --num_frames 81 --steps 40
```

**MXFP8_CUDNN (the path under study):**
```bash
HOME=/tmp CUDA_VISIBLE_DEVICES=0 python3 perf_bench/visual_gen/mxfp8_cudnn/run_prompt_suite.py \
  --backend MXFP8_CUDNN --backend_tag MXFP8 --gpu_id 0 \
  --model_path $MODEL --out_dir $OUT \
  --height 720 --width 1280 --num_frames 81 --steps 40
```

**Sage (1, 16, 1) and (1, 4, 1) — the comparison backends (TRTLLM + sage):**
```bash
# (1,16,1)
HOME=/tmp CUDA_VISIBLE_DEVICES=0 python3 perf_bench/visual_gen/mxfp8_cudnn/run_prompt_suite.py \
  --backend TRTLLM --backend_tag sage_blk16 --sage_blk_k 16 --gpu_id 0 \
  --model_path $MODEL --out_dir $OUT --height 720 --width 1280 --num_frames 81 --steps 40
# (1,4,1)
HOME=/tmp CUDA_VISIBLE_DEVICES=0 python3 perf_bench/visual_gen/mxfp8_cudnn/run_prompt_suite.py \
  --backend TRTLLM --backend_tag sage_blk4 --sage_blk_k 4 --gpu_id 0 \
  --model_path $MODEL --out_dir $OUT --height 720 --width 1280 --num_frames 81 --steps 40
```

For the 480p config in REPORT_PROMPTS, repeat all four with
`--height 480 --width 832 --num_frames 33 --steps 40` and
`OUT=.../run_480p_33f`.

Key invariants that must match for numbers to line up:
- **Same seed (42)** — it's the default; don't change it.
- **Same steps (40)** — the production setting. (20-step runs give very
  different, worse LPIPS — that was the invalidated early study.)
- **torch.compile + autotune ON** (the default; the report's opts-on rows).
- One backend per process; CFG size 1, Ulysses size 1, parallel VAE off,
  cuda_graph off — all set inside the driver.

---

## Step 3.5 — Prove the MXFP8 path actually fired (no silent fallback)

`run_prompt_suite.py` sets `TRTLLM_VISUAL_GEN_MXFP8_PER_CALL_TRACE` to
`<out_dir>/prompts/traces/per_call_MXFP8.txt`. After the MXFP8 run, inspect it:

```bash
sort $OUT/prompts/traces/per_call_MXFP8.txt | awk '{print $3}' | sort | uniq -c
# Expect, per prompt at S=75600:
#   80 path=mxfp8              <- self-attn took the cuDNN MXFP8 path
#   80 path=fallback_dispatch  <- cross-attn intentionally on bf16
#    0 path=fallback_exception <- ZERO cuDNN failures
```

If you see `path=fallback_dispatch` or `fallback_exception` on the
**self-attention** calls, your cuDNN overlay isn't working and you're
measuring bf16 — which is exactly the "can't reproduce / no difference"
symptom.

---

## Step 4 — Compute metrics (LPIPS / PSNR / SSIM / corr)

Script: `perf_bench/visual_gen/mxfp8_cudnn/compute_prompt_metrics_v2.py`.
It loads each `{prompt}_VANILLA.npy` as reference and scores
`{prompt}_{MXFP8,sage_blk16,sage_blk4}.npy` against it.

```bash
HOME=/tmp CUDA_VISIBLE_DEVICES=0 python3 \
  perf_bench/visual_gen/mxfp8_cudnn/compute_prompt_metrics_v2.py \
  --run_dir $OUT --tag 720p_81f
# writes $OUT/prompts/metrics_summary.json
```

LPIPS backbone is AlexNet (`lpips.LPIPS(net="alex")`), input normalized to
[-1, 1], scored at **native resolution** (no 256×256 resize), **all frames**
(not subsampled). These three choices matter — a different LPIPS backbone,
a resize, or frame subsampling will shift the absolute numbers.

Expected headline (720p/81f/40-step, opts-on, vs bf16):
- MXFP8 mean LPIPS ≈ **0.044** (imperceptible)
- Sage (1,16,1) ≈ 0.109, Sage (1,4,1) ≈ 0.116

---

## Step 5 — (optional) regenerate the HTML reports

```bash
python3 perf_bench/visual_gen/mxfp8_cudnn/_render_prompt_report_v2.py   # REPORT_PROMPTS.html
```

---

## Per-call kernel microbench (separate, no generation)

To reproduce REPORT.html §3b (per-call latency + TFLOPS across backends),
no Wan model needed — just synthetic Q/K/V at Wan shapes:

```bash
source perf_bench/visual_gen/mxfp8_cudnn/cudnn922_setup/activate.sh
HOME=/tmp CUDA_VISIBLE_DEVICES=0 python3 \
  perf_bench/visual_gen/mxfp8_cudnn/microbench_attn_backends.py \
  --warmup 10 --iters 50 --out /tmp/microbench.json
```

---

## File map (all under `perf_bench/visual_gen/mxfp8_cudnn/`)

| File | Role |
|---|---|
| `cudnn922_setup/` | cuDNN 9.22 overlay recipe (activate.sh / set_ld_preload.sh / sitecustomize.py / verify.py) |
| `run_prompt_suite.py` | 10-prompt generator, one backend/process — produces the `.npy` frames |
| `compute_prompt_metrics_v2.py` | LPIPS/PSNR/SSIM/corr vs VANILLA → `metrics_summary.json` |
| `microbench_attn_backends.py` | per-call kernel latency + TFLOPS, all 4 backends, no generation |
| `microbench_with_quant.py` | MXFP8 full-path (3× TE quant + kernel) microbench |
| `_render_prompt_report_v2.py` | renders REPORT_PROMPTS.html from `metrics_summary.json` |
| `run_wan_mxfp8_eval.py` | the single-prompt (panda) end-to-end driver used for §4/§5 |

The MXFP8 backend itself: `tensorrt_llm/_torch/visual_gen/attention_backend/mxfp8_cudnn.py`.

---

## If they still can't reproduce — checklist

1. `verify.py` shows cuDNN **92200** and pip-dir libcudnn? If not → overlay
   broken, fix Step 2 first.
2. Per-call trace shows `path=mxfp8` (not fallback) on self-attn? If not →
   still on bf16.
3. On a **B200** (sm_100)? MXFP8 SDPA won't run on H100.
4. Using the **`feature/mxfp8-sage-accuracy-study`** branch (has the backend
   + scripts)? Upstream main doesn't.
5. **40 steps, seed 42, opts-on**? 20-step or different seed → different LPIPS.
6. LPIPS computed **native-res, all-frames, AlexNet**? A resize / frame
   subsample / vgg backbone shifts absolute values.
7. Comparing against a **bf16 VANILLA reference generated in the same
   container/run**, not a golden from elsewhere?
