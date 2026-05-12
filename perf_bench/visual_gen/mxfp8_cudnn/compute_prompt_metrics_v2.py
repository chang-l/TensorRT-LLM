# ruff: noqa: E501
"""Compute PSNR/SSIM/LPIPS for a parameterised run dir (used by both configs)."""

import argparse
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
BACKENDS = ["MXFP8", "sage_blk16", "sage_blk4"]
REF = "VANILLA"


def psnr(a, b, max_val=255.0):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 10 * math.log10(max_val * max_val / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. .../mxfp8_cudnn/run_720p_81f")
    ap.add_argument("--tag", required=True, help="label for this config; e.g. 720p_81f")
    args = ap.parse_args()

    base = Path(args.run_dir) / "prompts"
    print(f"loading lpips alex on cuda for {args.tag}…", flush=True)
    model = _lpips.LPIPS(net="alex").to("cuda").eval()

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
            ssim_vals = [
                float(ssim_sk(ref[i], cmp[i], channel_axis=-1, data_range=255))
                for i in range(ref.shape[0])
            ]
            with torch.no_grad():
                lpips_vals = []
                for i in range(ref.shape[0]):
                    r = torch.from_numpy(ref[i : i + 1]).to("cuda").float() / 127.5 - 1.0
                    c = torch.from_numpy(cmp[i : i + 1]).to("cuda").float() / 127.5 - 1.0
                    r = r.permute(0, 3, 1, 2).contiguous()
                    c = c.permute(0, 3, 1, 2).contiguous()
                    lpips_vals.append(float(model(r, c).item()))
            p_full = psnr(ref, cmp)
            row = {
                "prompt": prompt,
                "backend": backend,
                "config_tag": args.tag,
                "psnr_db": (None if p_full == float("inf") else p_full),
                "corr": float(np.corrcoef(ref.flatten(), cmp.flatten())[0, 1]),
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
            ps = "inf" if row["psnr_db"] is None else f"{row['psnr_db']:.2f}"
            print(
                f"  {args.tag} {prompt:>18s} {backend:>10s}: PSNR={ps:>6s} SSIM={row['ssim_mean']:.3f} LPIPS={row['lpips_mean']:.4f}",
                flush=True,
            )

    out_json = base / "metrics_summary.json"
    json.dump(rows, open(out_json, "w"), indent=2)
    print(f"\nsaved {out_json}")


if __name__ == "__main__":
    main()
