# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Render a side-by-side diff video: [VANILLA | MXFP8 | amplified diff-heatmap].

Pure numpy frame compositing piped to the ffmpeg binary (no imageio/cv2/matplotlib).
The right panel is a 'hot' colormap of per-pixel max-channel |Δ|, amplified by --gain
so sub-perceptual differences are visible. Where the panels' content visibly differs
but neither looks degraded, that is trajectory divergence (esp. the 720p bf16-fallback
case); a uniform low-level wash is quantization-style error.
"""
import argparse
import subprocess
import sys

import numpy as np

SEP = 4  # px separator (even, keeps total width even for yuv420p)


def hot_colormap(d01):
    """MATLAB-'hot' colormap: black->red->yellow->white. d01 in [0,1] -> (H,W,3) uint8."""
    r = np.clip(3.0 * d01, 0, 1)
    g = np.clip(3.0 * d01 - 1.0, 0, 1)
    b = np.clip(3.0 * d01 - 2.0, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--van", required=True)
    ap.add_argument("--mxfp8", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--gain", type=float, default=4.0, help="diff amplification for the heatmap")
    args = ap.parse_args()

    a = np.load(args.van)    # (F,H,W,3) uint8  VANILLA
    b = np.load(args.mxfp8)  # (F,H,W,3) uint8  MXFP8
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    F, H, W, _ = a.shape

    diffmag = np.abs(a.astype(np.float32) - b.astype(np.float32)).max(axis=-1)  # (F,H,W) 0..255
    heat = hot_colormap(np.clip(diffmag / 255.0 * args.gain, 0, 1))             # (F,H,W,3)

    sep = np.zeros((H, SEP, 3), dtype=np.uint8)
    total_w = W * 3 + SEP * 2
    print(f"{args.out}: {F} frames, panel {W}x{H}, composite {total_w}x{H}, "
          f"mean|Δ|={diffmag.mean():.2f} maxframe|Δ|={diffmag.max():.0f} gain={args.gain}", flush=True)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{total_w}x{H}", "-r", str(args.fps),
        "-i", "-", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18", args.out,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(F):
        frame = np.concatenate([a[i], sep, b[i], sep, heat[i]], axis=1)  # (H, total_w, 3)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    rc = proc.wait()
    print(f"  ffmpeg rc={rc}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
