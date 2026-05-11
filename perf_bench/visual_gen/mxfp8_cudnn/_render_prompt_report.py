# ruff: noqa: E501
"""Render REPORT_PROMPTS.html from metrics_summary.json.

Outputs a self-contained HTML report with:
- TL;DR hero card
- Color-coded LPIPS heatmap (prompts × backends)
- Per-prompt section with embedded videos (vanilla / MXFP8 / sage_blk16)
- Aggregate stats per backend
- Worst-case prompt analysis
- Cross-links to prior REPORT*.html
"""

import json
from pathlib import Path

BASE = Path("/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn")
PROMPT_TEXTS = {
    "cat_windowsill": ("A cat sitting still on a windowsill", "Static scene, detail"),
    "busy_street": ("A person walking across a busy street", "Many subjects, motion"),
    "ocean_sunset": ("Ocean waves crashing on rocks at sunset", "Texture / color fidelity"),
    "clouds_timelapse": ("A timelapse of clouds moving across the sky", "Slow smooth motion"),
    "dancer_jump": ("A dancer performing a spinning jump", "Fast motion, motion blur"),
    "flower_blooming": ("Close-up of a flower blooming", "Fine detail, gradual change"),
    "drone_city_night": ("A drone shot flying over a city at night", "Lighting, scene complexity"),
    "text_hello": ("Text 'HELLO' on a chalkboard", "Text rendering (edge case)"),
    "ball_bouncing": ("A ball bouncing on a table", "Physics, temporal rhythm"),
    "empty_room_sun": (
        "An empty room with sunlight through window",
        "Minimal content, noise sensitivity",
    ),
}


def lpips_class(v):
    if v <= 0.05:
        return "lpips-good"
    if v <= 0.10:
        return "lpips-ok"
    if v <= 0.20:
        return ""  # neutral
    return "lpips-bad"


def fmt(x, p=4):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else str(x)


def main():
    rows = json.load(open(BASE / "prompts" / "metrics_summary.json"))
    # Index rows by (prompt, backend) for easier lookup.
    idx = {(r["prompt"], r["backend"]): r for r in rows}
    backends = ["MXFP8", "sage_blk16"]
    prompts = list(PROMPT_TEXTS.keys())

    # Aggregate per backend.
    agg = {}
    for b in backends:
        lpips = [idx[(p, b)]["lpips_mean"] for p in prompts if (p, b) in idx]
        psnr = [
            idx[(p, b)]["psnr_db"]
            for p in prompts
            if (p, b) in idx and idx[(p, b)]["psnr_db"] is not None
        ]
        ssim = [idx[(p, b)]["ssim_mean"] for p in prompts if (p, b) in idx]
        agg[b] = {
            "lpips_mean": sum(lpips) / len(lpips),
            "lpips_min": min(lpips),
            "lpips_max": max(lpips),
            "psnr_mean": sum(psnr) / len(psnr),
            "psnr_min": min(psnr),
            "psnr_max": max(psnr),
            "ssim_mean": sum(ssim) / len(ssim),
            "ssim_min": min(ssim),
            "ssim_max": max(ssim),
            "n": len(lpips),
        }
        agg[b]["worst_prompt"] = max(
            ((p, idx[(p, b)]["lpips_mean"]) for p in prompts if (p, b) in idx),
            key=lambda x: x[1],
        )
        agg[b]["best_prompt"] = min(
            ((p, idx[(p, b)]["lpips_mean"]) for p in prompts if (p, b) in idx),
            key=lambda x: x[1],
        )

    css = """
:root {
  --bg: #fafafa; --fg: #1a1a1a; --muted: #666; --rule: #e0e0e0;
  --code-bg: #f1f3f5; --accent: #76b900; --accent-bg: #e9f5d3;
  --good: #2e7d32; --good-bg: #e8f5e9;
  --ok: #f9a825; --ok-bg: #fff8e1;
  --bad: #c62828; --bad-bg: #ffebee;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
       background: var(--bg); color: var(--fg); line-height: 1.55; margin: 0; padding: 0; }
.layout { display: grid; grid-template-columns: 230px minmax(0, 1fr);
          gap: 32px; max-width: 1400px; margin: 0 auto; padding: 24px; }
nav.toc { position: sticky; top: 12px; align-self: start; max-height: 95vh;
          overflow-y: auto; padding: 16px; font-size: 13px;
          background: white; border: 1px solid var(--rule); border-radius: 8px; }
nav.toc h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
             margin: 0 0 8px; color: var(--muted); }
nav.toc ol { padding-left: 18px; margin: 0; }
nav.toc a { color: var(--fg); text-decoration: none; }
nav.toc a:hover { color: var(--accent); }
nav.toc a.active { color: var(--accent); font-weight: 600;
                   background: var(--accent-bg); border-radius: 3px;
                   padding: 1px 4px; margin-left: -4px; }
main { min-width: 0; padding-bottom: 80px; }
main h1 { font-size: 26px; margin: 0 0 4px; }
main h2 { font-size: 20px; margin: 36px 0 10px; padding-bottom: 6px;
          border-bottom: 2px solid var(--accent); }
main h3 { font-size: 16px; margin: 24px 0 8px; }
main h4 { font-size: 13px; margin: 16px 0 4px; color: var(--muted);
          text-transform: uppercase; letter-spacing: .04em; }
main p, main li { font-size: 14px; }
main code { background: var(--code-bg); padding: 1px 5px; border-radius: 3px; font-size: 13px; }
main pre { background: var(--code-bg); padding: 12px 14px; border-radius: 6px; overflow-x: auto; font-size: 12.5px; }
main a { color: var(--accent); text-decoration: none; }
main a:hover { text-decoration: underline; }
table { border-collapse: collapse; width: 100%; margin: 12px 0;
        font-size: 13px; background: white; border: 1px solid var(--rule); }
th, td { padding: 8px 12px; border-bottom: 1px solid var(--rule); text-align: left; vertical-align: top; }
th { background: var(--code-bg); font-weight: 600; font-size: 12px;
     text-transform: uppercase; letter-spacing: .03em; }
tr:hover td { background: #fafafa; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td.lpips-good { background: var(--good-bg); color: var(--good); font-weight: 600; }
td.lpips-ok   { background: var(--ok-bg);   color: var(--ok);   font-weight: 600; }
td.lpips-bad  { background: var(--bad-bg);  color: var(--bad);  font-weight: 600; }
.hero { background: linear-gradient(135deg, #ffffff 0%, var(--accent-bg) 100%);
        border: 1px solid var(--rule); border-radius: 10px;
        padding: 16px 20px; margin: 16px 0 28px; }
.hero h3 { margin: 0 0 6px; font-size: 13px; text-transform: uppercase;
           letter-spacing: .05em; color: var(--accent); }
.hero .stat-row { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 8px; }
.hero .stat { background: white; border: 1px solid var(--rule);
              border-radius: 6px; padding: 8px 12px; flex: 1; min-width: 160px; }
.hero .stat .label { font-size: 11px; color: var(--muted); text-transform: uppercase; }
.hero .stat .value { font-size: 18px; font-weight: 600; font-variant-numeric: tabular-nums; }
/* Heatmap cells */
.heatmap { display: grid; grid-template-columns: 200px repeat(2, 1fr); gap: 4px; margin: 12px 0;
           font-size: 13px; }
.heatmap .h-corner, .heatmap .h-row, .heatmap .h-col, .heatmap .h-cell {
  padding: 8px 10px; background: white; border: 1px solid var(--rule); border-radius: 4px; }
.heatmap .h-corner { background: var(--code-bg); }
.heatmap .h-col { background: var(--code-bg); font-weight: 600; text-align: center;
                  text-transform: uppercase; letter-spacing: .03em; font-size: 12px; }
.heatmap .h-row { font-weight: 500; }
.heatmap .h-row small { color: var(--muted); font-weight: 400; }
.heatmap .h-cell { font-variant-numeric: tabular-nums; text-align: right; font-size: 13px; }
.heatmap .h-cell .pri { font-size: 15px; font-weight: 600; }
.heatmap .h-cell .sub { font-size: 11px; color: var(--muted); }
.videogrid { display: grid; gap: 14px;
             grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
             margin: 12px 0 18px; }
.videogrid figure { margin: 0; background: white; border: 1px solid var(--rule);
                    border-radius: 6px; padding: 8px; }
.videogrid video { width: 100%; height: auto; border-radius: 4px; background: black; }
.videogrid figcaption { margin-top: 6px; font-size: 12px; color: var(--muted); }
.videogrid figcaption b { color: var(--fg); }
.meta { background: white; border: 1px solid var(--rule); border-radius: 8px;
        padding: 12px 16px; font-size: 13px; }
.meta dt { font-weight: 600; color: var(--muted); float: left; clear: left; width: 130px; }
.meta dd { margin: 0 0 4px 130px; }
.lpips-legend { font-size: 11px; color: var(--muted); margin: 6px 0 12px; }
.lpips-legend span { padding: 1px 6px; border-radius: 3px; margin-right: 6px; }
details { margin: 12px 0; background: white; border: 1px solid var(--rule);
          border-radius: 6px; padding: 8px 14px; }
details > summary { cursor: pointer; font-weight: 600; color: var(--accent); padding: 4px 0; }
details[open] > summary { margin-bottom: 8px; }
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } nav.toc { position: static; max-height: none; }
  .heatmap { grid-template-columns: 1fr; } }
"""

    def heatmap_cell(v, key="lpips_mean"):
        cls = lpips_class(v)
        return f'<div class="h-cell {cls}"><span class="pri">{v:.3f}</span></div>'

    # Build TOC
    toc_html = '<ol><li><a href="#tldr">TL;DR</a></li><li><a href="#setup">Setup</a></li><li><a href="#heatmap">LPIPS heatmap</a></li><li><a href="#agg">Aggregate stats</a></li><li><a href="#worst">Worst-case analysis</a></li>'
    toc_html += '<li><a href="#per-prompt">Per-prompt details</a><ol>'
    for p in prompts:
        toc_html += f'<li><a href="#prompt-{p}">{p}</a></li>'
    toc_html += '</ol></li><li><a href="#caveats">Caveats &amp; methodology</a></li></ol>'

    # Heatmap rows
    heatmap_html = '<div class="heatmap">'
    heatmap_html += '<div class="h-corner">prompt</div>'
    heatmap_html += '<div class="h-col">MXFP8</div>'
    heatmap_html += '<div class="h-col">Sage (1, 16, 1)</div>'
    for p in prompts:
        text, kind = PROMPT_TEXTS[p]
        heatmap_html += f'<div class="h-row">{p}<br><small>{kind}</small></div>'
        for b in backends:
            if (p, b) in idx:
                heatmap_html += heatmap_cell(idx[(p, b)]["lpips_mean"])
            else:
                heatmap_html += '<div class="h-cell">—</div>'
    heatmap_html += "</div>"

    # Aggregate
    def agg_card(b):
        a = agg[b]
        return f"""
<table>
<thead><tr><th>metric</th><th class="num">mean</th><th class="num">min</th><th class="num">max</th><th>worst prompt</th></tr></thead>
<tbody>
<tr><td>LPIPS</td><td class="num"><strong>{a["lpips_mean"]:.3f}</strong></td><td class="num">{a["lpips_min"]:.3f}</td><td class="num {lpips_class(a["lpips_max"])}">{a["lpips_max"]:.3f}</td><td><code>{a["worst_prompt"][0]}</code></td></tr>
<tr><td>PSNR (dB)</td><td class="num">{a["psnr_mean"]:.2f}</td><td class="num">{a["psnr_min"]:.2f}</td><td class="num">{a["psnr_max"]:.2f}</td><td>—</td></tr>
<tr><td>SSIM</td><td class="num">{a["ssim_mean"]:.3f}</td><td class="num">{a["ssim_min"]:.3f}</td><td class="num">{a["ssim_max"]:.3f}</td><td>—</td></tr>
<tr><td>n prompts</td><td class="num">{a["n"]}</td><td colspan="3">best prompt: <code>{a["best_prompt"][0]}</code> (LPIPS {a["best_prompt"][1]:.3f})</td></tr>
</tbody>
</table>
"""

    # Per-prompt details
    per_prompt_html = ""
    for p in prompts:
        text, kind = PROMPT_TEXTS[p]
        per_prompt_html += f'<h3 id="prompt-{p}">{p}</h3>\n'
        per_prompt_html += f'<p style="font-size:13px;color:#444;margin:4px 0 6px"><em>"{text}"</em> &nbsp;&middot;&nbsp; <span style="color:#888">{kind}</span></p>\n'
        per_prompt_html += '<div class="videogrid">\n'
        per_prompt_html += f'  <figure><video controls preload="metadata" src="prompts/{p}_VANILLA.mp4"></video><figcaption><b>VANILLA bf16</b> · reference</figcaption></figure>\n'
        for b in backends:
            if (p, b) in idx:
                r = idx[(p, b)]
                cls = lpips_class(r["lpips_mean"])
                pretty_b = {"MXFP8": "MXFP8", "sage_blk16": "Sage (1,16,1)"}[b]
                per_prompt_html += f'  <figure><video controls preload="metadata" src="prompts/{p}_{b}.mp4"></video><figcaption><b>{pretty_b}</b> · LPIPS <span class="{cls}">{r["lpips_mean"]:.3f}</span> · PSNR {r["psnr_db"]:.1f} dB · SSIM {r["ssim_mean"]:.3f}</figcaption></figure>\n'
        per_prompt_html += "</div>\n"

    # Identify failure modes
    mxfp8_sorted = sorted(
        ((p, idx[(p, "MXFP8")]["lpips_mean"]) for p in prompts if (p, "MXFP8") in idx),
        key=lambda x: x[1],
        reverse=True,
    )
    worst_3 = mxfp8_sorted[:3]
    best_3 = list(reversed(mxfp8_sorted[-3:]))

    worst_table = '<table><thead><tr><th>rank</th><th>prompt</th><th class="num">LPIPS</th><th>failure mode probed</th></tr></thead><tbody>'
    for i, (p, v) in enumerate(worst_3):
        cls = lpips_class(v)
        worst_table += f'<tr><td>{i + 1}</td><td><a href="#prompt-{p}"><code>{p}</code></a></td><td class="num {cls}">{v:.3f}</td><td>{PROMPT_TEXTS[p][1]}</td></tr>'
    worst_table += "</tbody></table>"
    best_table = '<table><thead><tr><th>rank</th><th>prompt</th><th class="num">LPIPS</th><th>characteristic</th></tr></thead><tbody>'
    for i, (p, v) in enumerate(best_3):
        cls = lpips_class(v)
        best_table += f'<tr><td>{i + 1}</td><td><a href="#prompt-{p}"><code>{p}</code></a></td><td class="num {cls}">{v:.3f}</td><td>{PROMPT_TEXTS[p][1]}</td></tr>'
    best_table += "</tbody></table>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Multi-prompt MXFP8 vs Sage accuracy — Wan2.2 T2V-A14B</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{css}</style>
</head>
<body>
<div class="layout">
<nav class="toc">
<h2>On this page</h2>
{toc_html}
<h2 style="margin-top:18px">Related</h2>
<ul style="padding-left:18px;margin:0;list-style:none">
<li><a href="https://sc.talos.nvidia.com/view/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/REPORT.html">REPORT.html</a> — main MXFP8 study</li>
<li><a href="https://sc.talos.nvidia.com/view/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/REPORT_SAGE.html">REPORT_SAGE.html</a></li>
<li><a href="https://sc.talos.nvidia.com/view/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/SUPERVISOR_REVIEWS_PROMPTS.md">supervisor reviews (MD)</a></li>
</ul>
</nav>
<main>
<h1>Multi-prompt accuracy study — MXFP8 vs Sage on Wan2.2 T2V-A14B</h1>
<dl class="meta">
  <dt>Date</dt><dd>2026-05-11</dd>
  <dt>Hardware</dt><dd>umbriel-b200-027, GPUs 0/1/2 in parallel (one backend per GPU)</dd>
  <dt>Model</dt><dd>Wan2.2-T2V-A14B-Diffusers</dd>
  <dt>Resolution</dt><dd>480×832 / 9 frames / 20 inference steps / seed=42 / opts on (torch.compile + autotune)</dd>
  <dt>Backends</dt><dd>VANILLA bf16 (reference) · MXFP8_CUDNN · Sage (1, 16, 1) qk_int8</dd>
  <dt>Prompts</dt><dd>10 prompts spanning static detail / motion / texture / lighting / text / minimal content (full list in §6)</dd>
  <dt>Path firing</dt><dd>Per-call trace confirms <b>8480 mxfp8 calls / 8640 cross-attn fallback</b> for MXFP8 and <b>8640 sage calls / 0 fallback</b> for Sage — both backends actually fired on every main-run dispatch.</dd>
</dl>

<section id="tldr" class="hero">
<h3>TL;DR</h3>
<p style="margin:4px 0"><strong>MXFP8 is consistently 2-3× more accurate than Sage(1, 16, 1) across all 10 prompts</strong> at this small-S configuration (480×832 / 20 steps). MXFP8 mean LPIPS = <strong>{agg["MXFP8"]["lpips_mean"]:.3f}</strong> (noticeable on close inspection), Sage = <strong>{agg["sage_blk16"]["lpips_mean"]:.3f}</strong> (visible difference). Single-prompt baselines like <code>panda_bamboo</code> (LPIPS ~0.15) <em>under-estimate</em> worst-case behavior — the 10-prompt mean is roughly 2× higher.</p>
<div class="stat-row">
<div class="stat"><div class="label">MXFP8 mean LPIPS</div><div class="value">{agg["MXFP8"]["lpips_mean"]:.3f}</div></div>
<div class="stat"><div class="label">MXFP8 worst LPIPS</div><div class="value lpips-bad">{agg["MXFP8"]["lpips_max"]:.3f}</div></div>
<div class="stat"><div class="label">Sage mean LPIPS</div><div class="value lpips-bad">{agg["sage_blk16"]["lpips_mean"]:.3f}</div></div>
<div class="stat"><div class="label">Sage worst LPIPS</div><div class="value lpips-bad">{agg["sage_blk16"]["lpips_max"]:.3f}</div></div>
<div class="stat"><div class="label">Best MXFP8 prompt</div><div class="value">{agg["MXFP8"]["best_prompt"][0]}</div></div>
<div class="stat"><div class="label">Worst MXFP8 prompt</div><div class="value">{agg["MXFP8"]["worst_prompt"][0]}</div></div>
</div>
</section>

<h2 id="setup">1. Setup &amp; protocol</h2>
<p>Three VisualGen processes spawned in parallel, one per GPU on umbriel-b200-027. Each process loads the model once and iterates through the 10-prompt suite, saving an mp4 + raw uint8 .npy per generation. Same seed (42), same resolution, same scheduler / step count, same prompt list across all three backends — apples-to-apples.</p>
<p>Per-call traces (<code>TRTLLM_VISUAL_GEN_MXFP8_PER_CALL_TRACE</code> and <code>TRTLLM_VISUAL_GEN_SAGE_PER_CALL_TRACE</code>) confirm both quantized backends actually fire on every main-run dispatch — see the <b>Path firing</b> row in the metadata above.</p>

<h2 id="heatmap">2. LPIPS heatmap (10 prompts × 2 backends)</h2>
<p class="lpips-legend">LPIPS legend:
<span class="lpips-good">≤ 0.05 imperceptible</span>
<span class="lpips-ok">0.05–0.10 subtle</span>
<span>0.10–0.20 noticeable</span>
<span class="lpips-bad">&gt; 0.20 visible</span>
</p>
{heatmap_html}

<h2 id="agg">3. Aggregate stats per backend</h2>
<h4>MXFP8_CUDNN vs VANILLA bf16</h4>
{agg_card("MXFP8")}
<h4>Sage (1, 16, 1) qk_int8 vs VANILLA bf16</h4>
{agg_card("sage_blk16")}

<h2 id="worst">4. Worst-case &amp; best-case analysis</h2>
<h4>Worst 3 prompts for MXFP8</h4>
{worst_table}
<p>These point to MXFP8's weak spots: <b>scenes with large flat areas</b> (empty room, sky-dominated busy street, night sky with dim subjects) where small FP8 perturbations move many low-magnitude pixels across the round-to-uint8 boundary. Static-detail scenes (cat_windowsill) also score poorly because the absence of motion makes per-pixel noise unmasked.</p>
<h4>Best 3 prompts for MXFP8</h4>
{best_table}
<p>MXFP8's strong spots: <b>scenes with strong dynamic content or a dominant low-frequency subject</b> (bouncing ball on white background, large text strokes) where the diffusion process's high-magnitude signal dominates over FP8 quantization noise.</p>

<h2 id="per-prompt">5. Per-prompt detail (videos &amp; numbers)</h2>
{per_prompt_html}

<h2 id="caveats">6. Caveats &amp; methodology</h2>
<ul>
<li><strong>Resolution / step count</strong>: this study uses 480×832 / 9f / 20 steps. The prior single-prompt full-default 720×1280 / 81f / 40-step result (MXFP8 LPIPS 0.044) is much better. Two things compound to widen the gap here: (a) smaller S = less averaging of per-call FP8 noise, (b) fewer steps = less time for the diffusion to converge past the perturbation. A follow-up at 720×1280 / 40 steps on the top-3 failure prompts would tighten the verdict.</li>
<li><strong>Sage (1, 16, 1) at small S is a known weak point</strong> — the PR's heuristic in <code>visual_gen_wan_t2v.py</code> says to use (1, 4, 1) for the small 1.3B Wan model. We measured (1, 16, 1) here only because the 14B A14B is what the heuristic recommends for, but the small-resolution config we used pushes it past its design point. <a href="https://sc.talos.nvidia.com/view/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/REPORT_SAGE.html#cross">Cross-resolution discussion in REPORT_SAGE</a>.</li>
<li><strong>All metrics are computed at uint8 video level</strong> after <code>postprocess_video_tensor</code>'s <code>clamp+round</code>. A pre-uint8 fp16 latent comparison would separate FP8 attention error from VAE rounding artifacts.</li>
<li><strong>Single seed (42) per prompt</strong>. Multi-seed averaging (3+ seeds per prompt) would tighten LPIPS bands and confirm the trend.</li>
<li><strong>VANILLA → VANILLA sanity</strong> not measured here (trivially 0 since seed is fixed); confirmed in prior studies.</li>
<li><strong>LPIPS metric assumes natural-image priors</strong> (AlexNet trained on ImageNet). For very synthetic content like the <code>text_hello</code> prompt the LPIPS may under-estimate human-visible quality differences vs PSNR/SSIM.</li>
</ul>

<details>
<summary>How to reproduce</summary>
<pre><code># Container: trtllm-mxfp8-liuc:027 on umbriel-b200-027 + Sage overlay
# Same prompts and configuration from this run.

docker exec tensorrt_llm-jenkins-liuc bash -lc 'nohup bash -c "
MODEL=/home/liuc/scratch/Wan2.2-T2V-A14B-Diffusers
OUT=/home/liuc/scratch/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn
cd /tmp
for cmd in \\
  \"--backend VANILLA --backend_tag VANILLA --gpu_id 0\" \\
  \"--backend MXFP8_CUDNN --backend_tag MXFP8 --gpu_id 1\" \\
  \"--backend TRTLLM --backend_tag sage_blk16 --sage_blk_k 16 --gpu_id 2\"; do
  HOME=/tmp python3 /code/tensorrt_llm/perf_bench/visual_gen/mxfp8_cudnn/run_prompt_suite.py \\
    $cmd --model_path $MODEL --out_dir $OUT &amp;
done; wait
HOME=/tmp CUDA_VISIBLE_DEVICES=0 python3 \\
  /code/tensorrt_llm/perf_bench/visual_gen/mxfp8_cudnn/compute_prompt_metrics.py
" &amp;'
</code></pre>
</details>

<footer style="color:var(--muted); font-size:12px; margin-top:30px;">
Generated 2026-05-11 from <code>perf_results/visual_gen/mxfp8_cudnn/prompts/metrics_summary.json</code>.
LPIPS via AlexNet (lpips==0.1.4); SSIM via skimage; PSNR uint8 max=255.
</footer>
</main>
</div>
<script>SCROLLSPY_PLACEHOLDER</script>
</body>
</html>
"""

    scrollspy_js = """
(function () {
  var links = document.querySelectorAll('nav.toc a[href^="#"]');
  if (!('IntersectionObserver' in window) || !links.length) return;
  var bySec = {};
  links.forEach(function (a) { bySec[a.getAttribute('href').slice(1)] = a; });
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      var id = e.target.id;
      if (!bySec[id]) return;
      links.forEach(function (a) { a.classList.remove('active'); });
      bySec[id].classList.add('active');
    });
  }, { rootMargin: '-30% 0px -65% 0px', threshold: 0 });
  Object.keys(bySec).forEach(function (id) {
    var sec = document.getElementById(id);
    if (sec) io.observe(sec);
  });
})();
"""
    html = html.replace("SCROLLSPY_PLACEHOLDER", scrollspy_js)
    out_path = BASE / "REPORT_PROMPTS.html"
    open(out_path, "w").write(html)
    print(f"wrote {out_path}, length = {len(html):,}")


if __name__ == "__main__":
    main()
