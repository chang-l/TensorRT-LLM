# MXFP8 cuDNN study — silent-fallback bug: the colleague is right

**TL;DR.** The colleague who "couldn't reproduce" actually *did* reproduce — they
correctly discovered a bug I missed: at the **720p/81f production shape
(B=2, S=75600)**, the cuDNN `sdpa_mxfp8` path **throws a `RuntimeError` on every
self-attention call and silently falls back to bf16 SDPA**. So the headline
720p MXFP8 accuracy numbers in `REPORT*.html` measured **bf16-vs-bf16**, not FP8.
And **yes — the perf POC and the accuracy run executed different kernels**: the
microbench ran at **B=1** (where MXFP8 works), the real Wan run at **B=2** (where
it throws). The 480p/33f (S=14040) run is the one config where MXFP8 genuinely
ran, and there it *does* perturb quality.

Date: 2026-06-09. Evidence is from committed artifacts on branch
`feature/mxfp8-sage-accuracy-study` (no live GPU needed — the B200 allocation
expired at 10:01 mid-investigation; live error-message capture is still pending).

---

## 1. What the committed per-call traces prove

`run_prompt_suite.py` writes `TRTLLM_VISUAL_GEN_MXFP8_PER_CALL_TRACE`. The MXFP8
backend logs one line per attention call with the path it took:
`path=mxfp8` (kernel fired), `path=fallback_dispatch` (cross-attn, intentional
bf16), `path=fallback_exception:<Err>` (kernel threw → bf16).

Decompressed from `run_720p_81f/prompts/traces/per_call_MXFP8.txt.gz` and
`run_480p_33f/...`:

| Accuracy run | shape (self-attn) | `path=mxfp8` | `fallback_exception:RuntimeError` | verdict |
|---|---|---:|---:|---|
| **720p/81f** | **S=75600** | **0** | **16,160** | MXFP8 NEVER ran; 100% bf16 fallback |
| 720p/81f | small warmup shapes (S=14040/32400/32760) | 480 | 0 | only warmups fired |
| **480p/33f** | **S=14040** | **16,480** | 0 | MXFP8 genuinely ran |
| 480p/33f | large transient shapes | 0 | 160 | threw at big S |

Exactly matches the colleague's trace line:
`... path=fallback_exception:RuntimeError B=2 H=40 S=75600 D=128 dtype=torch.bfloat16`
and their observation "only warm-ups had 50% MXFP8" (the small warmup shapes
fire; 50% = self-attn half, the other half is cross-attn `fallback_dispatch`).

## 2. Why the bogus 720p number was 0.10 LPIPS and not 0

The MXFP8 fallback path calls
`F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=self.scale)`
(`mxfp8_cudnn.py:345/355`). **`VanillaAttention` (the VANILLA reference backend)
calls the *identical* function with the *identical* HND layout**
(`vanilla.py:68`, both `preferred_layout = HND`). So 720p "MXFP8" was literally
the same bf16 SDPA call as the reference.

The residual LPIPS ≈ 0.10 is therefore a **bf16-vs-bf16 noise floor**, not FP8
error. Most likely source: `MXFP8CudnnAttention._mxfp8_forward` is
`@torch.compiler.disable` and wrapped in try/except, so under `torch.compile` the
attention is a graph break with a different kernel-selection/fusion context than
VANILLA's inlined SDPA → tiny per-step bf16 differences compound over 40 denoise
steps. (Run-to-run cuDNN/flash nondeterminism would do the same.) Either way it
says nothing about MXFP8.

**Consequence:** the 720p study cannot resolve any perturbation below ≈ 0.10
LPIPS. Sage at 720p (0.21) is clearly above that floor (real); 720p "MXFP8"
(0.10) sits *at* the floor — consistent with "it contributed nothing because it
never ran."

## 3. Perf POC vs accuracy run — different kernels (the core question)

From committed `microbench_attn_backends.json` (`B=1`, H=40, D=128, 50 iters):

| shape | backend | median_ms | TFLOPS | err |
|---|---|---:|---:|---|
| S=75600 | VANILLA bf16 | 87.42 | 1339 | None |
| **S=75600** | **MXFP8_CUDNN** | **69.58** | **1682** | **None ← ran fine** |
| S=75600 | sage_blk16 | 75.80 | 1544 | None |
| S=75600 | sage_blk4 | 82.63 | 1416 | None |

`bench_mxfp8()` has **no fallback path** — it either executes the cuDNN graph or
raises (caught in `main()` as an `error` row). `err=None` + 1.26× speedup
(87.42/69.58) ⇒ at **B=1, S=75600 the MXFP8 kernel genuinely executed.**

But the real Wan run uses **B=2** (CFG concatenates conditional+unconditional;
the driver sets `dit_cfg_size=1`, so cond+uncond batch into B=2). At B=2 the same
shape throws.

> **So the perf number (1.26× at S=75600) is real but was measured at B=1 — a
> batch size the production pipeline never uses. The accuracy run at B=2 fell
> back to bf16. They ran different kernels.** This is precisely the
> "CC keeps running different kernels for perf vs acc" failure the colleague
> suspected.

## 4. Root cause hypothesis (B-triggered, not S² alone)

The B=1 success **falsifies** a pure-S² int32 overflow (75600² = 5.7e9 > INT32),
because S² is batch-independent and would break B=1 too. The trigger is
**B-dependent**. Leading candidate: the **bf16 output tensor's byte count** —
`B·H·S·D·2` bytes = **1.55 GB at B=1** (< INT32_MAX 2.15e9) but **3.10 GB at
B=2** (> INT32_MAX). A 32-bit byte-size/stride computation inside the cuDNN
`sdpa_mxfp8` plan would overflow at B=2 only. (Unconfirmed — needs the live
error text; see §6.)

The colleague's fix — swapping `TE.MXFP8Quantizer` for
`torch.ops.trtllm.mxfp8_quantize(x2d, True, alignment=32)` — making the error go
away suggests the throw is sensitive to the scale/data tensor allocation
(layout, contiguity, or total byte size) the quantizer emits, consistent with a
size/stride overflow rather than a math error. Their finding that this then
shows a **real quality regression** (their "video 2") is the *true* 720p MXFP8
behavior, and aligns with their point that softmax (a normalized exponential) is
more sensitive to mantissa than to the extra exponent range MX buys.

## 5. What is valid vs invalid in REPORT*.html

| Claim | Status |
|---|---|
| 720p/81f MXFP8 LPIPS ("imperceptible") | **INVALID** — kernel never ran; bf16 noise floor |
| 720p/81f MXFP8 perf 1.26× speedup | **Real at B=1 only**; not realizable at production B=2 (throws) |
| 480p/33f MXFP8 LPIPS ≈ 0.1375 | **VALID** — kernel genuinely fired (16,480 mxfp8 calls); real FP8 |
| Sage 720p / 480p LPIPS | Valid (Sage path doesn't use cuDNN sdpa_mxfp8) |
| "Sage K-block depends on S" headline | Unaffected (Sage-only) |
| REPRO_GUIDE Step 3.5 "expect 0 fallback_exception" | **Was aspirational/wrong for 720p** — the real 720p trace is 100% fallback_exception |

## 6. Pending live verification (B200 expired mid-run)

Repro script ready: `perf_results/visual_gen/mxfp8_cudnn/repro/repro_mxfp8_fallback.py`.
Sweeps B∈{1,2} × S∈{14040,75600} through TE-quantize + cuDNN `sdpa_mxfp8`,
printing scale-tensor sizes and the **full** exception text. To run on a fresh
B200 (release:1.3.0rc6 image has cuDNN 9.17 → overlay 9.22):

```bash
docker run -d --name mxfp8repro --gpus all --ipc=host --ulimit memlock=-1 \
  -e HOME=/tmp -v /home/scratch.liuc_coreai:/home/scratch.liuc_coreai \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6 sleep infinity
docker exec mxfp8repro bash -lc '
  pip install -q nvidia-cudnn-cu13==9.22.0.52 nvidia-cudnn-frontend==1.23.0
  L=$(python3 -c "import nvidia.cudnn,os;print(os.path.join(list(nvidia.cudnn.__path__)[0],\"lib\"))")
  LD_LIBRARY_PATH=$L:$LD_LIBRARY_PATH python3 \
    /home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/repro/repro_mxfp8_fallback.py'
```

Expected (from the on-disk evidence): B=1 S=75600 → OK; B=2 S=75600 → ERR
RuntimeError; both S=14040 → OK. The printed message should name the exact cuDNN
status / overflow.

Still to do after the message is captured: confirm the colleague's
`torch.ops.trtllm.mxfp8_quantize` swap makes B=2/S=75600 execute, and re-measure
720p MXFP8 LPIPS with the kernel *actually running*.
