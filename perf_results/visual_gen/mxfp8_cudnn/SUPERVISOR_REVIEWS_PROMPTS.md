# SUPERVISOR REVIEWS — Multi-Prompt MXFP8 + Sage Accuracy Study

The autonomous supervisor agent for this run had its Bash/Write/Read tools
denied at agent-spawn time, so the worker is logging its own checkpoint
verifications inline below. Each checkpoint cites the exact verification
command/output so a future supervisor can re-audit deterministically.

Study scope: 10 prompts × 3 backends (VANILLA bf16 / MXFP8_CUDNN /
Sage(1,16,1) qk_int8) at 480×832 / 9 frames / 20 steps / seed=42 on
umbriel-b200-027. One backend per GPU (0/1/2).

## Checklist (from the supervisor's pre-run brief)

1. **Container setup on 027** — cuDNN 9.22, cudnn-frontend 1.23, TE 2.12,
   Sage overlay; env parity with the 043 container.
2. **Driver script** — `TRTLLM_VISUAL_GEN_MXFP8_PER_CALL_TRACE` and
   `..._SAGE_PER_CALL_TRACE` both set; seed=42 fixed; one backend per GPU;
   predictable output layout.
3. **Run quality gate** per cell — mp4 + npy exist; per-call trace shows
   non-zero `path=mxfp8` (or `path=sage`) at S=4680; zero
   `path=fallback_exception` in main run; VANILLA shows no MXFP8/Sage firings.
4. **Metrics sanity** — LPIPS / PSNR / SSIM populated for all 30 cells;
   flag LPIPS > 0.2; identify worst 3 prompts per backend.
5. **Final report** — apples-to-apples statement, color-coded heatmap,
   worst-prompt analysis, back-links to prior REPORT/REPORT_SAGE/SUPERVISOR
   reviews.

## Review log

Appended as each checkpoint is verified.
