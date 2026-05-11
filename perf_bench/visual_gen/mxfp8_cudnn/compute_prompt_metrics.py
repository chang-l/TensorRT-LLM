# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate PSNR/SSIM/LPIPS for the 10-prompt × 3-backend suite.

Pairs each non-VANILLA backend's .npy against the matching VANILLA .npy and
runs compare_videos.py-equivalent metrics on each pair. Writes a single
JSON summary plus a CSV for quick spreadsheet inspection.
"""

import json
import math
from pathlib import Path

import lpips as _lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_sk

PROMPTS = [
    "cat_windowsill",
    "busy_street",
    "ocean_sunset",
    "clouds_timelapse",
    "dancer_jump",
    "flower_blooming",
    "drone_city_night",
    "text_hello",
    "ball_bouncing",
    "empty_room_sun",
]
BACKENDS = ["MXFP8", "sage_blk16"]
REF = "VANILLA"


def psnr(a, b, max_val=255.0):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 10 * math.log10(max_val * max_val / mse)


def corr(a, b):
    return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])


def ssim_video(a, b):
    vals = [float(ssim_sk(a[i], b[i], channel_axis=-1, data_range=255)) for i in range(a.shape[0])]
    return vals


def lpips_video(a, b, model, device="cuda"):
    def to(x):
        t = torch.from_numpy(x).to(device).float() / 127.5 - 1.0
        return t.permute(0, 3, 1, 2).contiguous()

    vals = []
    with torch.no_grad():
        for i in range(a.shape[0]):
            r = to(a[i : i + 1])
            c = to(b[i : i + 1])
            vals.append(float(model(r, c).item()))
    return vals


def main():
    base = Path(
        "/home/liuc/scratch/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/prompts"
    )
    print("loading lpips alex on cuda…", flush=True)
    lpips_model = _lpips.LPIPS(net="alex").to("cuda").eval()
    rows = []
    for prompt in PROMPTS:
        ref_path = base / f"{prompt}_{REF}.npy"
        if not ref_path.exists():
            print(f"  [skip] {prompt}: no VANILLA ref")
            continue
        ref = np.load(ref_path)
        for backend in BACKENDS:
            cmp_path = base / f"{prompt}_{backend}.npy"
            if not cmp_path.exists():
                print(f"  [skip] {prompt}/{backend}: missing")
                continue
            cmp = np.load(cmp_path)
            if ref.shape != cmp.shape:
                print(f"  [skip] shape mismatch {prompt}/{backend}: {ref.shape} vs {cmp.shape}")
                continue
            diff = np.abs(ref.astype(np.int32) - cmp.astype(np.int32))
            p_full = psnr(ref, cmp)
            c_full = corr(ref, cmp)
            ssim_vals = ssim_video(ref, cmp)
            lpips_vals = lpips_video(ref, cmp, lpips_model)
            row = {
                "prompt": prompt,
                "backend": backend,
                "psnr_db": (None if p_full == float("inf") else p_full),
                "corr": c_full,
                "ssim_mean": float(np.mean(ssim_vals)),
                "ssim_min": float(np.min(ssim_vals)),
                "ssim_max": float(np.max(ssim_vals)),
                "lpips_mean": float(np.mean(lpips_vals)),
                "lpips_min": float(np.min(lpips_vals)),
                "lpips_max": float(np.max(lpips_vals)),
                "abs_diff_mean": float(diff.mean()),
                "abs_diff_max": int(diff.max()),
                "per_frame_psnr": [psnr(ref[i], cmp[i]) for i in range(ref.shape[0])],
                "per_frame_ssim": ssim_vals,
                "per_frame_lpips": lpips_vals,
                "n_frames": int(ref.shape[0]),
            }
            rows.append(row)
            p_str = "inf" if row["psnr_db"] is None else f"{row['psnr_db']:.2f}"
            print(
                f"  {prompt:>20s} {backend:>10s}: "
                f"PSNR={p_str:>6s}dB SSIM={row['ssim_mean']:.3f} "
                f"LPIPS={row['lpips_mean']:.4f}",
                flush=True,
            )

    out_json = base / "metrics_summary.json"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved {out_json}")

    # CSV
    out_csv = base / "metrics_summary.csv"
    with open(out_csv, "w") as f:
        f.write(
            "prompt,backend,psnr_db,corr,ssim_mean,lpips_mean,lpips_max,abs_diff_mean,abs_diff_max\n"
        )
        for r in rows:
            p = "" if r["psnr_db"] is None else f"{r['psnr_db']:.3f}"
            f.write(
                f"{r['prompt']},{r['backend']},{p},{r['corr']:.4f},"
                f"{r['ssim_mean']:.4f},{r['lpips_mean']:.4f},{r['lpips_max']:.4f},"
                f"{r['abs_diff_mean']:.3f},{r['abs_diff_max']}\n"
            )
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
