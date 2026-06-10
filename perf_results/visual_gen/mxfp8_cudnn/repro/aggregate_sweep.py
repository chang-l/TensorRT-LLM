# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate LPIPS across seed dirs: per-variant mean +/- std over all (prompt, seed).

Usage: aggregate_sweep.py <seed_dir1> <seed_dir2> ...   (each has prompts/{prompt}_{tag}.npy)
"""
import glob
import os
import sys

import numpy as np
import torch
import lpips

VARIANTS = ["MXnone", "MXsqK", "MXsqQK", "MXhad", "MXhadSQ"]
_L = lpips.LPIPS(net="alex")
if torch.cuda.is_available():
    _L = _L.cuda()


def lp(a, b):
    def t(x):
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        return x.cuda() if torch.cuda.is_available() else x
    ta, tb = t(a), t(b)
    with torch.no_grad():
        vals = [ _L(ta[i:i + 8], tb[i:i + 8]).flatten() for i in range(0, ta.shape[0], 8) ]
    return float(torch.cat(vals).mean())


def main():
    dirs = sys.argv[1:]
    res = {v: [] for v in VARIANTS}
    per_prompt = {v: {} for v in VARIANTS}
    for d in dirs:
        for vp in sorted(glob.glob(f"{d}/prompts/*_VANILLA.npy")):
            pid = os.path.basename(vp)[: -len("_VANILLA.npy")]
            van = np.load(vp)
            for v in VARIANTS:
                tp = f"{d}/prompts/{pid}_{v}.npy"
                if not os.path.exists(tp):
                    continue
                e = lp(van, np.load(tp))
                res[v].append(e)
                per_prompt[v].setdefault(pid, []).append(e)
    base = np.array(res["MXnone"]) if res["MXnone"] else np.array([np.nan])
    bmean = base.mean()
    print(f"\n{'variant':<10}{'n':>4}{'meanLPIPS':>11}{'std':>8}{'vs MXnone':>11}")
    for v in VARIANTS:
        a = np.array(res[v]) if res[v] else np.array([np.nan])
        delta = "" if v == "MXnone" else f"{(a.mean() / bmean - 1) * 100:+.1f}%"
        print(f"{v:<10}{len(a):>4}{a.mean():>11.4f}{a.std():>8.4f}{delta:>11}")
    print("\nper-prompt mean LPIPS (rows=prompt, cols=variant):")
    prompts = sorted({p for v in VARIANTS for p in per_prompt[v]})
    print(f"{'prompt':<18}" + "".join(f"{v:>9}" for v in VARIANTS))
    for p in prompts:
        row = "".join(
            f"{np.mean(per_prompt[v][p]):>9.3f}" if p in per_prompt[v] else f"{'-':>9}"
            for v in VARIANTS
        )
        print(f"{p:<18}{row}")


if __name__ == "__main__":
    main()
