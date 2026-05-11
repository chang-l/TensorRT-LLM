"""Compute PSNR/correlation for each step count in the sweep, MXFP8 vs VANILLA."""

import argparse
import json
import math
import os

import numpy as np


def psnr(a, b, max_val=255.0):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 10 * math.log10(max_val * max_val / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step_counts", nargs="+", type=int, default=[2, 5, 10, 20, 40])
    args = ap.parse_args()

    header = (
        f"{'steps':>6} | {'PSNR (dB)':>10} | {'corr':>10} | "
        f"{'mean abs':>10} | {'max abs':>8} | "
        f"{'gen_v (s)':>10} | {'gen_m (s)':>10}"
    )
    print(header)
    print("-" * 90)
    rows = []
    # Read timings from json
    sv = {
        x["steps"]: x["gen_s"]
        for x in json.load(open(os.path.join(args.dir, "sweep_VANILLA.json")))
        if x["seed"] == args.seed
    }
    sm = {
        x["steps"]: x["gen_s"]
        for x in json.load(open(os.path.join(args.dir, "sweep_MXFP8_CUDNN.json")))
        if x["seed"] == args.seed
    }
    for steps in args.step_counts:
        a = np.load(os.path.join(args.dir, f"VANILLA_seed{args.seed}_steps{steps}.npy"))
        b = np.load(os.path.join(args.dir, f"MXFP8_CUDNN_seed{args.seed}_steps{steps}.npy"))
        diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
        p = psnr(a, b)
        c = float(np.corrcoef(a.flatten(), b.flatten())[0, 1])
        ma = diff.mean()
        mx = diff.max()
        gv = sv.get(steps, float("nan"))
        gm = sm.get(steps, float("nan"))
        ps = "inf" if p == float("inf") else f"{p:.2f}"
        print(
            f"{steps:>6} | {ps:>10} | {c:>10.6f} | {ma:>10.3f} | {mx:>8} | {gv:>10.2f} | {gm:>10.2f}"
        )
        rows.append(
            {
                "steps": steps,
                "psnr_db": (None if p == float("inf") else p),
                "corr": c,
                "mean_abs": float(ma),
                "max_abs": int(mx),
                "gen_vanilla_s": gv,
                "gen_mxfp8_s": gm,
            }
        )
    out_json = os.path.join(args.dir, f"sweep_psnr_summary_seed{args.seed}.json")
    json.dump(rows, open(out_json, "w"), indent=2)
    print(f"\nsaved {out_json}")


if __name__ == "__main__":
    main()
