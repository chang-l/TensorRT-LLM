"""Render inline SVG sparklines for per-frame PSNR/SSIM/LPIPS metrics.

Reads logs/per_frame_metrics.json (produced by compare_videos.py --per_frame
parsed into a JSON dict of {backend: [{frame, psnr, ssim, lpips}, ...]}) and
emits a single HTML snippet with one <svg> per (metric, backend) pair.

Used at HTML-report generation time only; no runtime dependency.
"""

import json
import os

W = 360
H = 70
PAD = 4
LABEL_W = 38

BACKENDS = [
    ("MXFP8_CUDNN_opts", "MXFP8", "#76b900"),  # NVIDIA green
    ("TRTLLM_sage_blk4", "Sage (1,4,1)", "#1976d2"),
    ("TRTLLM_sage_blk16", "Sage (1,16,1)", "#ef6c00"),
]

METRICS = [
    ("psnr", "PSNR (dB)", False),  # higher is better
    ("ssim", "SSIM", False),
    ("lpips", "LPIPS", True),  # lower is better
]


def render_one_metric(data, metric_key, label, lower_better):
    # global min/max across all backends for this metric to keep y-axis comparable
    all_vals = []
    for b, _, _ in BACKENDS:
        if b in data:
            all_vals.extend(d[metric_key] for d in data[b])
    if not all_vals:
        return f"<div>no data for {metric_key}</div>"
    vmin, vmax = min(all_vals), max(all_vals)
    # pad y range slightly
    span = max(vmax - vmin, 1e-6)
    vmin -= span * 0.05
    vmax += span * 0.05
    n_frames = max(len(data[b]) for b, _, _ in BACKENDS if b in data)
    inner_w = W - LABEL_W - PAD
    inner_h = H - 2 * PAD

    def pt(i, v):
        x = LABEL_W + i / max(n_frames - 1, 1) * inner_w
        y = PAD + (1 - (v - vmin) / (vmax - vmin)) * inner_h
        return f"{x:.1f},{y:.1f}"

    svg = [
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="{label} per frame">'
    ]
    # background
    svg.append(
        f'<rect x="{LABEL_W}" y="{PAD}" width="{inner_w}" height="{inner_h}" '
        f'fill="#fafafa" stroke="#e0e0e0" stroke-width="1"/>'
    )
    # gridlines (3 horizontal)
    for frac in (0.25, 0.5, 0.75):
        y = PAD + frac * inner_h
        svg.append(
            f'<line x1="{LABEL_W}" y1="{y:.1f}" x2="{W - PAD}" y2="{y:.1f}" '
            f'stroke="#eaeaea" stroke-width="1"/>'
        )
    # axis labels (min/max)
    svg.append(
        f'<text x="{LABEL_W - 4}" y="{PAD + 8}" text-anchor="end" '
        f'font-size="9" font-family="monospace" fill="#666">{vmax:.3g}</text>'
    )
    svg.append(
        f'<text x="{LABEL_W - 4}" y="{H - PAD - 1}" text-anchor="end" '
        f'font-size="9" font-family="monospace" fill="#666">{vmin:.3g}</text>'
    )
    # axis title (rotated)
    svg.append(
        f'<text x="10" y="{H / 2}" text-anchor="middle" font-size="9" '
        f'font-family="sans-serif" fill="#444" '
        f'transform="rotate(-90 10 {H / 2})">{label}</text>'
    )
    # one polyline per backend
    for b, name, color in BACKENDS:
        if b not in data:
            continue
        pts = " ".join(pt(i, d[metric_key]) for i, d in enumerate(data[b]))
        svg.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round">'
            f"<title>{name}</title></polyline>"
        )
    svg.append("</svg>")
    return "".join(svg)


def render_legend():
    items = " &middot; ".join(
        f'<span style="color:{color};font-weight:600">■</span> {name}'
        for _, name, color in BACKENDS
    )
    return (
        '<div style="font-size:11px;color:#666;margin:4px 0 8px">'
        f"Per-frame (n=81 frames at 720×1280 / 81f / 40 steps / opts on): "
        f"{items} &nbsp; vs VANILLA bf16 reference."
        "</div>"
    )


def main():
    base = "/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn"
    data = json.load(open(os.path.join(base, "logs", "per_frame_metrics.json")))
    grid_style = (
        "display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:8px;margin:8px 0"
    )
    out = [f'<div class="sparklines" style="{grid_style}">']
    for key, label, lower_better in METRICS:
        out.append(
            '<figure style="margin:0;background:white;border:1px solid #e0e0e0;border-radius:6px;padding:6px 8px">'
        )
        out.append(render_one_metric(data, key, label, lower_better))
        arrow = "↓ lower=better" if lower_better else "↑ higher=better"
        out.append(
            f'<figcaption style="font-size:10px;color:#888;text-align:right;margin-top:-2px">{arrow}</figcaption>'
        )
        out.append("</figure>")
    out.append("</div>")
    out.append(render_legend())
    html = "\n".join(out)
    open(os.path.join(base, "logs", "sparklines.html"), "w").write(html)
    print("wrote sparklines.html, length =", len(html))


if __name__ == "__main__":
    main()
