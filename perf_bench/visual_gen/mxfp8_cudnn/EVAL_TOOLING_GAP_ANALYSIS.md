# Gap analysis: `compute_prompt_metrics_v2.py` vs PR #13567

PR: [NVIDIA/TensorRT-LLM #13567 — "Use LPIPS for visual gen model regression test"](https://github.com/NVIDIA/TensorRT-LLM/pull/13567/changes)

My script (this study): `perf_bench/visual_gen/mxfp8_cudnn/compute_prompt_metrics_v2.py`

The two tools were built independently for different audiences (research vs.
CI regression), so they overlap on LPIPS but diverge significantly on
everything else. This doc inventories the differences and recommends a path
to a single unified eval methodology for all VisualGen model development.

## TL;DR

- **Different problems, both solved correctly.** PR #13567 is an
  end-to-end "generate + LPIPS-score + threshold-gate" CI tool. My script
  is a post-hoc "compute every metric against cached frames" research tool.
- **Neither one alone is the right canonical tool.** PR's structural shape
  (JSON dataset, YAML config, model aliasing, threshold gate, used by
  pytest) is right for CI. My script's metric breadth (PSNR, SSIM, LPIPS,
  Pearson corr, abs_diff + per-frame arrays + min/max + all-frame coverage)
  is right for research.
- **Recommendation: unify by composing.** Take the PR's skeleton; bolt on
  the metric layer from my script behind flags. ~2–3 day project.
  Single eval interface across FLUX / LTX-2 / Wan-2.1 / Wan-2.2 onward.

## 1. Side-by-side feature matrix

| Capability | `compute_prompt_metrics_v2.py` (110 LOC) | `visual_gen_lpips_score_eval.py` (670 + 113 LOC) |
|---|---|---|
| **End-to-end** (generate + score) | ❌ score-only (consumes cached `.npy`) | ✅ both (loads VisualGen, generates, scores) |
| **Score-only** (re-score cached outputs) | ✅ primary mode | ✅ secondary (`generated_image_path` in dataset) |
| Input format — image | ❌ | ✅ PIL via `PIL.Image.open` |
| Input format — video | ✅ raw `.npy` (BHWC uint8) | ✅ mp4 via ffmpeg decode + tensor via permute |
| Input format — tensor | ❌ | ✅ `torch.Tensor`, `np.ndarray`, `PIL.Image` |
| **Metrics — PSNR** | ✅ full video + per-frame | ❌ |
| **Metrics — SSIM** (skimage) | ✅ per-frame + mean/min/max | ❌ |
| **Metrics — LPIPS** | ✅ per-frame + mean/min/max (AlexNet only) | ✅ mean only (alex/vgg/squeeze configurable) |
| **Metrics — Pearson correlation** | ✅ | ❌ |
| **Metrics — abs_diff** (mean, max) | ✅ | ❌ |
| LPIPS frame coverage (video) | all frames | 8 frames sampled `linspace` |
| LPIPS resolution handling | native (no resize) | bicubic-resize to `(W, H)`, default 256×256 |
| **N-way comparison** (multiple backends vs one ref in one run) | ✅ iterates `BACKENDS = [...]` vs `REF` | ❌ one pair per sample |
| **Per-frame metric arrays** in output | ✅ `[psnr, ssim, lpips]` arrays length `n_frames` | ❌ single mean per sample |
| **Min/max** stats per metric | ✅ | ❌ |
| **Skip-on-error** (continue past missing files / shape mismatch) | ✅ | ❌ raises |
| Dataset config | hardcoded `PROMPTS` and `BACKENDS` lists | JSON dataset (samples + threshold + per-sample params) |
| Model alias resolution (flux1, wan21, …) | ❌ N/A (offline) | ✅ `MODEL_ALIASES` + `$LLM_MODELS_ROOT` lookup + HF fallback |
| YAML config integration (`VisualGenArgs.from_yaml`) | ❌ N/A | ✅ same format as `trtllm-serve --extra_visual_gen_options` |
| Threshold pass/fail gate (CI) | ❌ | ✅ raises `RuntimeError` on fail with non-zero exit |
| Stdout-only JSON output for tool chaining | ❌ | ✅ `--json` flag |
| Hooked into pytest regression tests | ❌ standalone | ✅ used by `test_flux_pipeline.py`, `test_ltx2_pipeline.py`, `test_wan21_t2v_pipeline.py` |
| Golden references in-tree | ❌ external `.npy` artifacts | ✅ `tests/unittest/_torch/visual_gen/golden/*.png|.mp4|.json` committed |

## 2. What my script has that PR is missing (gaps in PR)

These are the things you'd lose if you migrated wholesale to the PR's tool today:

### 2a. PSNR / SSIM / corr / abs_diff
LPIPS alone is great for "does it look the same?" but bad for "is this a
1-bit perturbation or a 50-bit one?". For quant-vs-bf16 studies, the four
metrics together tell a story LPIPS alone can't:
- **PSNR** distinguishes bit-identical (∞ dB) from sub-uint8 quant noise
  (~30–40 dB) from visible degradation (<25 dB)
- **SSIM** picks up structural change LPIPS missed (skimage SSIM is a
  cheap classic baseline)
- **corr** flags channel/colour drift that LPIPS' perceptual features
  invariantize away
- **abs_diff** is the literal "how many uint8 levels are different" — the
  one number a debugger actually wants

### 2b. Per-frame arrays
PR returns one mean per sample. My script returns `per_frame_psnr`,
`per_frame_ssim`, `per_frame_lpips` length-`n_frames` arrays. Per-frame is
required to:
- spot frame-level regressions (e.g. only frame 27 collapses → temporal
  attention bug; mean would dilute this into "0.05 LPIPS, looks fine")
- generate the sparklines in our REPORT.html § 5a (visual time-series
  per-metric across the video)
- compute confidence intervals (need the distribution, not just the mean)

### 2c. All-frame coverage by default
PR samples 8 frames evenly via `_sample_frame_indices`. For a 81-frame Wan
video that's a 10× subsampling. Three downsides:
- subsamples might miss the worst-frame region (sampling alias)
- the "mean" then conflates per-frame degradation with frame selection
- changing `max_frames` between runs perturbs the score

For CI gating, 8 frames is fine — speed matters and the threshold absorbs
noise. For research, default `all`.

### 2d. Native-resolution mode
PR resizes everything to 256×256 by default. LPIPS values **change** with
resolution: a perturbation that's 0.05 LPIPS at 1280×720 may be 0.02 LPIPS
at 256×256 because the perceptual network sees fewer details to disagree
on. Our REPORT_PROMPTS.html S-sensitivity analysis (Sage K-block collapse
at small S) is unreliable if LPIPS were computed at a fixed 256×256.

### 2e. Multi-backend in one run
My script iterates `BACKENDS = ["MXFP8", "sage_blk16", "sage_blk4"]` ×
`REF = "VANILLA"` in one invocation. The PR does one ref vs one cmp per
sample, so an N-backend comparison study would need N separate runs and
manual aggregation. For CI (one model, one threshold), the PR's pattern
is right. For research (4 backends × 10 prompts × 2 resolutions = 80
cells), the iteration model is right.

### 2f. Skip semantics
My script logs and continues past missing files / shape mismatches. PR
raises immediately. CI wants raises; research wants skip-and-keep-going
so a single broken cell doesn't lose the other 79.

## 3. What PR has that my script is missing (gaps in mine)

These are the things you'd gain by adopting the PR's shape:

### 3a. End-to-end generation
PR loads `VisualGen` from a YAML config + model alias, generates the
images / videos itself, and scores in one pass. My script assumes
someone already ran the workload and saved `.npy` tensors. This is a real
productivity gap — every research study currently re-runs a custom
generator script (`run_prompt_suite.py`, `run_wan_mxfp8_eval.py`, …)
followed by `compute_prompt_metrics_v2.py`. The PR collapses that into
one command.

### 3b. JSON dataset format
Prompts + reference paths + per-sample generation params (height, width,
steps, seed, …) all in one declarative file. My script hardcodes the
prompt list and assumes a fixed file layout.

### 3c. YAML config integration with `VisualGenArgs`
Same YAML file format as `trtllm-serve --extra_visual_gen_options`. Means
the same config tested for serving is tested for accuracy — no drift.

### 3d. Model alias resolution + `LLM_MODELS_ROOT` lookup
`flux1` → checks `$LLM_MODELS_ROOT/FLUX.1-dev` → falls back to HF ID.
Stops everyone hardcoding `/home/scratch.trt_llm_data_ci/...` paths.

### 3e. LPIPS backbone selection
`alex` is fast but noisy. `vgg` is more correlated with human judgement
but 3× slower. `squeeze` is the smallest. CI may want one, research may
want another.

### 3f. CI hookup
PR's tool is wired into three pytest files. My script has zero. For a
canonical eval tool, CI integration is the highest-leverage feature.

### 3g. Threshold pass/fail
`--threshold 0.05`: process exits non-zero if mean LPIPS > 0.05. CI gates
need this. Research doesn't, but it's free to include.

## 4. Recommendation: unify by composing

Build **one** tool with the PR's structural shape and the metric breadth
of my script. Concrete recipe:

### 4.1 Base: take PR #13567's `visual_gen_lpips_score_eval.py` as the skeleton

Keep:
- JSON dataset format + YAML `VisualGenArgs` config
- Model alias resolution + `LLM_MODELS_ROOT`
- End-to-end generate-and-score loop with `--output-dir`
- `--threshold` pass/fail gate
- `--json` stdout mode for tool chaining

### 4.2 Add a metric-selection layer

```
--metrics psnr,ssim,lpips,corr,abs_diff    # default: lpips only (CI parity)
                                            # research mode: --metrics all
```

Internally route each metric through a small `compute_<metric>` function;
collect results into the per-sample dict.

### 4.3 Add per-frame mode for videos

```
--frame-sampling all|N                      # default: 8 (CI parity)
                                            # research: all
--per-frame-output                          # default off; on → emit
                                            #  per_frame_<metric> arrays
```

### 4.4 Add native-resolution mode

```
--resize WxH|none                           # default: 256x256 (CI parity)
                                            # research: none
```

LPIPS values change with input size, so this flag must be reported in the
output JSON so cross-run comparisons can detect mismatched configs.

### 4.5 Add multi-comparison in one run

Extend the dataset schema:

```json
{
  "id": "cat_windowsill",
  "prompt": "...",
  "params": {"height": 480, "width": 832, ...},
  "reference_image_path": "vanilla/cat_windowsill.npy",
  "compare_to": [
    {"backend_tag": "MXFP8",      "generated_image_path": "mxfp8/cat_windowsill.npy"},
    {"backend_tag": "sage_blk16", "generated_image_path": "sage16/cat_windowsill.npy"},
    {"backend_tag": "sage_blk4",  "generated_image_path": "sage4/cat_windowsill.npy"}
  ]
}
```

`compare_to` is optional; if absent, fall back to a single
`generated_image_path` / generate-and-score (the PR's current behaviour).

### 4.6 Add `--skip-on-error` flag

Default off (CI behaviour). On = log and continue (research behaviour).

### 4.7 Migrate consumers

- `perf_bench/visual_gen/mxfp8_cudnn/compute_prompt_metrics_v2.py` →
  delete; call the unified tool from `_render_prompt_report_v2.py`'s
  pipeline (or have it consume the JSON output of the unified tool).
- `run_prompt_suite.py` → optionally fold into the unified tool by adding
  a `--generate-only` mode that runs N backends sequentially writing
  `.npy` outputs (so the prompt-suite workflow stays the same shape).
- The three pytest files in PR #13567 keep working — they use the
  `lpips_video_utils.py` helper, which is unchanged.

### 4.8 File layout after the unification

```
scripts/visualgen_eval/
├── visual_gen_eval.py              # renamed from visual_gen_lpips_score_eval.py
├── lpips_video_utils.py            # already in PR
├── metrics.py                      # NEW: psnr, ssim, corr, abs_diff helpers
└── README.md                       # how to drive it from CI + research
```

## 5. Pros / cons of unifying vs. keeping two

### Pros
- **One eval methodology** across image (FLUX.1/2), video (LTX-2, Wan-2.1,
  Wan-2.2, future Cosmos) and across CI + research.
- **Single source of truth for LPIPS numbers** — different eval scripts
  computing LPIPS differently (frame subsampling, resize, backbone) is
  the most common source of "why don't your numbers match mine" friction.
- **CI gets richer metrics for free** — if the metric layer exists, CI
  can opt in to PSNR/SSIM checks that catch class-bugs LPIPS missed
  (e.g. colour-channel swap → LPIPS unaffected, corr drops to ~0).
- **Research gets CI integration for free** — adding a new model into
  the test suite is just dropping a JSON dataset + golden references.
- **Lower maintenance** — one tool to keep in sync with VisualGen API
  changes vs. two.

### Cons
- One-time migration cost (~2–3 days engineering): adding flags,
  refactoring my script's call sites, validating against existing
  REPORT_PROMPTS numbers.
- The unified tool is necessarily larger than either piece alone
  (~800 LOC vs 110 / 670). Trade simplicity-per-file for unification.
- Risk of regression on the existing CI tests if defaults change —
  mitigate by keeping the PR's defaults (256×256 resize, 8-frame
  sampling, LPIPS-only) and gating new behaviour behind explicit flags.

### Bottom line

Worth doing. The PR's tool is already going to be the canonical regression
gate for FLUX / LTX / Wan; making it also serve the research-study workflow
(at the cost of ~30 LOC of flag plumbing + a `metrics.py` helper) prevents
two tools diverging and gives every future model-onboarding effort a
clear path to "drop in a dataset, drop in a golden, you're done."

If you'd like, the next step is a small RFC PR layering the metric +
flags onto a branch off of yibinl-nvidia's PR #13567 before it lands,
so the two efforts converge instead of competing.
