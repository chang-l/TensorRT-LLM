# Supervisor reviews — Sage attention accuracy study

Independent reviews from the supervisor agent. The supervisor is read-only and verifies that worker checkpoints are real (not silent fallback, not test theater).

---

## Review #1 — Acknowledgement (study kickoff)

Supervisor acknowledged the role and verification checklist:
1. Overlaid `SageAttentionConfig` matches PR #13570 spec.
2. Counter instrumentation present.
3. Pytest covers BOTH (1,4,1) and (1,16,1) at S∈{4096, 8192, 75600} with cos>0.98 and rel_rms<0.10.
4. End-to-end mp4 + npy exist; per-call traces show non-zero `path=sage` at S=75600 and zero `fallback_exception` in main run.
5. 10 npy files = 5 step-counts × 2 variants.
6. Spot-rerun compare_videos.py on at least one pair.
7. Final table includes all four backends with LPIPS as perceptual verdict.

Will append per-step PASS/FAIL.

---

## Review #2 — Steps 1-3 verification

**Tooling note**: in this session the supervisor's harness denied Bash, Write, WebFetch, and read access to the wheel install path. So the review covers what was readable from the in-repo files; the wheel-overlay byte-diff vs PR ref `3f92685565562982d01e9bb1e16997a81c332898` could not be performed and is deferred.

### 6-bullet verdict

- **Step 1 plumbing — PASS-conditional**. Worker's reported smoke (validator instantiating `AttentionConfig(backend="TRTLLM", sage_attention_config=...)`, factory returning `MXFP8CudnnAttention` and `TrtllmAttention`, validator scrubbing `VANILLA + sage` to `None` with a log) is the right shape. Byte-verify against PR #13570 next session.
- **Step 1 reverting `modules/attention.py` — PASS**. The wheel's `_attn_impl` (in-repo line 326) calls `self.attn.forward(q=q, k=k, v=v, **kwargs)` with no `batch_size`/`seq_len`. The PR's `attention.py` would have forced `attn2d_row_size` infra into unrelated parallel paths; revert was correct.
- **Step 2 counters + per-call trace — PASS, with one ask**. Instance-attr counters bumped per branch are simple and verifiable (Step 3's `test_sage_repeated_calls_keep_counter_growing` confirms strict increment). `__del__` is unreliable on interpreter shutdown — depend on `_per_call_log` file, not the dump-on-destruct, and **`flush()` after each append** so a mid-step crash doesn't lose the tail.
- **Step 3 test file structure — PASS**. `test_sage_attention.py` parametrizes Wan-style `(B=1, H=40, S∈{4096, 8192}, D=128)` and both granularities `(1,4,1)`/`(1,16,1)`, correctly reshapes between HND ref and NHD backend, and counter-test independently exercises Step 2.
- **`forward()` signature relaxation — PASS, legitimate**. The wheel's `_attn_impl` calls `forward(q=, k=, v=, **kwargs)` and the in-repo non-PR `TrtllmAttention.forward()` already derives `batch_size = q.shape[0]; seq_len = q.shape[1]` (lines 270-272). Making the PR's required positional args optional with shape-derived defaults exactly matches pre-PR semantics — not masking a deeper mismatch. **ACTION**: log the divergence in the perf_results README so a future reader doesn't think the overlay matches PR #13570 byte-for-byte.
- **Test tolerances `cos > 0.95, rel_rms < 0.20` — FAIL: too loose**. Observed cos ≈ 0.996, rel_rms ≈ 0.048 — an order of magnitude inside the asserted bounds. A genuine regression would still pass. **Tighten to `cos > 0.99` and `rel_rms < 0.10`** (still ~2× slack). Separately: same seed=0 across shapes gives near-equivalent samples; pass distinct seeds (e.g. `seed=S`) per shape.

### Verdict table

| Item | Verdict |
|------|---------|
| Step 1 plumbing | PASS-conditional (subject to PR-ref byte-diff) |
| Step 1 revert of `modules/attention.py` | PASS |
| Step 2 counters + per-call trace | PASS (add `flush()` after each append) |
| Step 3 test file structure | PASS |
| Step 3 test tolerance + seeding | FAIL — tighten to cos>0.99, rel_rms<0.10; per-shape seed |
| `forward()` signature relaxation | PASS (legitimate; document divergence) |

### Step 4 acceptance gate (preview)

Beyond rendered-video accuracy: confirm the `_per_call_log` shows `sage_calls > 0` and `fallback_calls == 0` for **every DiT layer × every one of the 40 main steps**, for both granularities. That's the load-bearing assertion that the sage path actually fired in production — unit tests don't cover it.

### Files reviewed (paths)

- `tests/unittest/_torch/visual_gen/test_sage_attention.py`
- In-repo HEAD versions of `tensorrt_llm/_torch/visual_gen/{attention_backend/trtllm.py, attention_backend/utils.py, config.py, modules/attention.py}` (NOT the overlay).

Could not access: actual wheel-overlaid files at `/usr/local/lib/python3.12/dist-packages/tensorrt_llm/...` and PR #13570 reference at SHA `3f92685565562982d01e9bb1e16997a81c332898`.
