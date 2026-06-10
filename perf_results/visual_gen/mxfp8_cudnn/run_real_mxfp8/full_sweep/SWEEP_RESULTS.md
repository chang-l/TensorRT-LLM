# Full sweep: 6 variants x 10 prompts x 3 seeds (480p) — statistically-firm LPIPS vs bf16

umbriel-b200-027, patched backend (no fallback; transforms QK^T-invariant verified 1e-7).
All real MXFP8 (0 fallback_exception). LPIPS-alex vs bf16 VANILLA, paired per (prompt,seed).
(Note: GPUs 4-7 were taken by another user mid-run -> those jobs OOM'd; refilled on GPUs 0-3.)

## AGGREGATE (n = 30 = 10 prompts x 3 seeds), mean LPIPS +/- std, vs MXnone baseline
| variant | mean | std | vs baseline |
|---|---|---|---|
| MXnone (baseline)     | 0.1307 | 0.073 | — |
| + SmoothQuant-K       | 0.1199 | 0.070 | -8.2% |
| + SmoothQuant-QK      | 0.1296 | 0.091 | -0.8% (FLAT) |
| + Hadamard            | 0.1169 | 0.082 | -10.6% |
| + Hadamard+SmoothQuant-QK | 0.1132 | 0.082 | **-13.4% (best)** |

## Key corrections vs earlier small-sample numbers
- SmoothQuant-QK looked like -15% (3 prompts) / -20% (seed42, 10 prompts) but is ~FLAT (-0.8%)
  over 30 points. Wildly seed-dependent: seed42 -20%, seed123 +23% (WORSE), seed7 -6%. Do not
  use standalone.
- Hadamard: -10.6% over 30 (earlier -19% on 3 prompts overstated). Robust, best single transform.
- Combined Hadamard+SmoothQuant-QK: -13.4%, best overall.
- SmoothQuant-K: -8.2%, modest but consistent.

## Per-prompt mean LPIPS (rows=prompt, cols=variant)
prompt            MXnone  MXsqK  MXsqQK  MXhad  MXhadSQ
ball_bouncing      0.061  0.052  0.034   0.032  0.054
busy_street        0.257  0.166  0.222   0.156  0.196
cat_windowsill     0.123  0.130  0.129   0.146  0.123
clouds_timelapse   0.118  0.088  0.114   0.114  0.075
dancer_jump        0.180  0.219  0.229   0.217  0.186
drone_city_night   0.134  0.109  0.128   0.125  0.110
empty_room_sun     0.129  0.142  0.151   0.077  0.109
flower_blooming    0.136  0.131  0.123   0.101  0.086
ocean_sunset       0.094  0.085  0.112   0.073  0.094
text_hello         0.075  0.077  0.054   0.126  0.098

Hadamard/combined help ~7/10 prompts but HURT cat_windowsill, text_hello, dancer_jump.
Caveat: 480p, std large (~0.08); paired comparison. Mp4s in full_sweep/s{42,123,7}/.

## Follow-up: mean-subtraction K-centering (MEANSUB=k), 3 seeds x 10 prompts
Exactly QK^T-invariant (vK=K-mean_S(K), Q unchanged); real MXFP8, 0 fallback_exception.
| seed | MXnone | MXmsK (mean-sub-K) | delta |
|---|---|---|---|
| 42  | 0.1350 | 0.1142 | -15% |
| 123 | 0.1338 | 0.1625 | +21% (worse) |
| 7   | 0.1233 | 0.1470 | +19% (worse) |
| mean| 0.1307 | 0.1412 | +8% (NET WORSE) |
VERDICT: does NOT help -- net +8% worse, seed-dependent (great on s42, bad on s123/s7).
Likely the QK-RMSNorm already removes K's DC offset, so centering the residual just
shrinks signal and worsens relative E4M3 error. Clean negative result.
