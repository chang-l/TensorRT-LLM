"""Compare two video .npy frame stacks: PSNR / SSIM / LPIPS / Pearson corr / abs-diff.

Metrics:
- PSNR (dB): peak signal-to-noise ratio on uint8 pixels.
- SSIM:      structural similarity index (skimage), measures local luminance/contrast/structure.
- LPIPS:     Learned Perceptual Image Patch Similarity (AlexNet, lpips==0.1).
             Lower = more perceptually similar. Typical: <0.1 = imperceptible.
- corr:      flattened-tensor Pearson correlation; coarse similarity proxy.
- abs-diff:  per-pixel uint8 difference statistics.

LPIPS / SSIM imports are optional — skipped with a note if not installed.
"""

import argparse
import math

import numpy as np


def psnr(a, b, max_val=255.0):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = ((a - b) ** 2).mean()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(max_val * max_val / mse)


def pearson_corr(a, b):
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])


def ssim_per_frame(ref, cmp):
    """SSIM per frame using skimage; ref/cmp shape (N, H, W, 3) uint8."""
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception as e:
        print(f"[skimage SSIM unavailable: {e}]")
        return None
    vals = []
    for i in range(ref.shape[0]):
        s = ssim(ref[i], cmp[i], channel_axis=-1, data_range=255)
        vals.append(float(s))
    return vals


def lpips_per_frame(ref, cmp, net="alex", device="cuda"):
    """LPIPS per frame; lower = more perceptually similar."""
    try:
        import lpips as _lpips
        import torch
    except Exception as e:
        print(f"[LPIPS unavailable: {e}]")
        return None
    model = _lpips.LPIPS(net=net).to(device).eval()

    # LPIPS wants float tensors in [-1, 1], shape (N, 3, H, W)
    def to_lpips(x_np):
        t = torch.from_numpy(x_np).to(device).float() / 127.5 - 1.0  # [-1,1]
        return t.permute(0, 3, 1, 2).contiguous()

    vals = []
    with torch.no_grad():
        # Process per-frame to bound VRAM
        for i in range(ref.shape[0]):
            r = to_lpips(ref[i : i + 1])
            c = to_lpips(cmp[i : i + 1])
            d = model(r, c).item()
            vals.append(float(d))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference .npy (e.g. VANILLA)")
    ap.add_argument("--cmp", required=True, help="compared .npy (e.g. MXFP8_CUDNN)")
    ap.add_argument("--no_lpips", action="store_true", help="skip LPIPS (slow on CPU)")
    ap.add_argument("--no_ssim", action="store_true", help="skip SSIM")
    ap.add_argument(
        "--per_frame", action="store_true", help="print per-frame numbers (default: summary only)"
    )
    args = ap.parse_args()

    ref = np.load(args.ref)
    cmp = np.load(args.cmp)
    print(f"ref shape={ref.shape} dtype={ref.dtype}")
    print(f"cmp shape={cmp.shape} dtype={cmp.dtype}")
    if ref.shape != cmp.shape:
        raise SystemExit("shapes differ; cannot compare")
    diff = np.abs(ref.astype(np.int32) - cmp.astype(np.int32))
    print(f"abs-diff:  mean={diff.mean():.3f}  max={diff.max()}  p99={np.percentile(diff, 99):.3f}")
    print(f"PSNR:      {psnr(ref, cmp):.2f} dB")
    print(f"corr:      {pearson_corr(ref, cmp):.6f}  (Pearson, full-video flatten)")

    ssim_vals = None if args.no_ssim else ssim_per_frame(ref, cmp)
    if ssim_vals is not None:
        s = np.asarray(ssim_vals)
        print(f"SSIM:      mean={s.mean():.4f}  min={s.min():.4f}  max={s.max():.4f}")

    lpips_vals = None if args.no_lpips else lpips_per_frame(ref, cmp)
    if lpips_vals is not None:
        lp = np.asarray(lpips_vals)
        print(
            f"LPIPS:     mean={lp.mean():.4f}  min={lp.min():.4f}  "
            f"max={lp.max():.4f}  (AlexNet, lower=better)"
        )

    if args.per_frame:
        print()
        print(f"{'frame':>5} | {'PSNR':>7} | {'corr':>7} | {'SSIM':>7} | {'LPIPS':>7}")
        for i in range(ref.shape[0]):
            p = psnr(ref[i], cmp[i])
            c = pearson_corr(ref[i], cmp[i])
            s = ssim_vals[i] if ssim_vals else float("nan")
            lp_i = lpips_vals[i] if lpips_vals else float("nan")
            ps = "inf" if p == float("inf") else f"{p:.2f}"
            print(f"{i:>5} | {ps:>7} | {c:>7.4f} | {s:>7.4f} | {lp_i:>7.4f}")


if __name__ == "__main__":
    main()
