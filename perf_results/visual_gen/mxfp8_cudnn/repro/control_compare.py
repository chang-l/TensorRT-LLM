# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Compare control-experiment .npy outputs to separate FP8 error from trajectory
divergence.

Given a run dir with {prompt}_{tag}.npy (uint8 frames), computes, per pair:
  byte-identical, mean|Δ|, max|Δ|, %pixels-differ, PSNR(dB), LPIPS(alex, native, all frames).

Pairs of interest:
  VAN_a vs VAN_b      -> pure run-to-run divergence FLOOR (same backend, same seed)
  VAN_a vs MX_a       -> FP8 + divergence  (gap above the floor ≈ true FP8 effect)
"""
import argparse
import glob
import os

import numpy as np

try:
    import lpips
    import torch
    _LPIPS = lpips.LPIPS(net="alex")
    if torch.cuda.is_available():
        _LPIPS = _LPIPS.cuda()
    _HAVE_LPIPS = True
except Exception as e:  # noqa: BLE001
    print(f"[warn] lpips unavailable ({e!r}); reporting byte/PSNR only")
    _HAVE_LPIPS = False


def psnr(a, b):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


def lpips_score(a, b):
    if not _HAVE_LPIPS:
        return float("nan")
    import torch
    # a,b: (F,H,W,3) uint8 -> (F,3,H,W) in [-1,1]; native res, all frames
    def t(x):
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        return x.cuda() if torch.cuda.is_available() else x
    with torch.no_grad():
        vals = []
        ta, tb = t(a), t(b)
        for i in range(0, ta.shape[0], 8):  # chunk frames to bound memory
            vals.append(_LPIPS(ta[i:i + 8], tb[i:i + 8]).flatten())
        return float(torch.cat(vals).mean().item())


def cmp(a, b):
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "byte_ident": bool(np.array_equal(a, b)),
        "mean_abs": float(diff.mean()),
        "max_abs": int(diff.max()),
        "pct_diff": 100.0 * float((diff > 0).mean()),
        "psnr_db": psnr(a, b),
        "lpips": lpips_score(a, b),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="refTag:testTag entries, e.g. VAN_a:VAN_b VAN_a:MX_a")
    args = ap.parse_args()
    for pair in args.pairs:
        ref_tag, test_tag = pair.split(":")
        refs = sorted(glob.glob(f"{args.dir}/*_{ref_tag}.npy"))
        print(f"\n===== {ref_tag}  vs  {test_tag} =====")
        print(f"{'prompt':<18}{'ident':>6}{'mean|Δ|':>9}{'max|Δ|':>8}{'%diff':>7}{'PSNR':>8}{'LPIPS':>8}")
        agg = {"mean_abs": [], "pct_diff": [], "psnr_db": [], "lpips": []}
        for rp in refs:
            pid = os.path.basename(rp)[: -(len(ref_tag) + 5)]
            tp = f"{args.dir}/{pid}_{test_tag}.npy"
            if not os.path.exists(tp):
                continue
            a, b = np.load(rp), np.load(tp)
            if a.shape != b.shape:
                print(f"{pid:<18} SHAPE {a.shape} vs {b.shape}")
                continue
            m = cmp(a, b)
            for k in agg:
                agg[k].append(m[k])
            print(f"{pid:<18}{str(m['byte_ident']):>6}{m['mean_abs']:>9.3f}{m['max_abs']:>8}"
                  f"{m['pct_diff']:>6.1f}%{m['psnr_db']:>8.2f}{m['lpips']:>8.4f}")
            del a, b
        if agg["mean_abs"]:
            n = len(agg["mean_abs"])
            print(f"{'MEAN ('+str(n)+')':<18}{'':>6}{np.mean(agg['mean_abs']):>9.3f}{'':>8}"
                  f"{np.mean(agg['pct_diff']):>6.1f}%{np.mean(agg['psnr_db']):>8.2f}"
                  f"{np.nanmean(agg['lpips']):>8.4f}")


if __name__ == "__main__":
    main()
