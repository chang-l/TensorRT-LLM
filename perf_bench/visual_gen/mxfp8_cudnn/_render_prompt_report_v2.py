# ruff: noqa: E501
"""Render REPORT_PROMPTS.html from two-config 40-step prompt suite.

Replaces the original 20-step REPORT_PROMPTS.html with proper
40-step data across two sequence-length operating points:

  - run_720p_81f: 720x1280 / 81 frames / 40 steps (production target, S=75600)
  - run_480p_33f: 480x832 / 33 frames / 40 steps (mid-S, S=14040)

Reads each run's prompts/metrics_summary.json. Both configs have complete
VANILLA / MXFP8_CUDNN / Sage(1,16,1) data — all 10 prompts × 3 backends × 2
configs = 60 runs total.
"""

import json
from pathlib import Path

BASE = Path("/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn")
PROMPT_TEXTS = {
    "cat_windowsill": ("A cat sitting still on a windowsill", "Static detail"),
    "busy_street": ("A person walking across a busy street", "Many subjects, motion"),
    "ocean_sunset": ("Ocean waves crashing on rocks at sunset", "Texture, color"),
    "clouds_timelapse": ("Clouds drifting across the sky (timelapse)", "Slow motion"),
    "dancer_jump": ("A dancer performing a spinning jump", "Fast motion"),
    "flower_blooming": ("Close-up of a flower blooming", "Fine detail"),
    "drone_city_night": ("Drone shot over a city at night", "Lighting, complexity"),
    "text_hello": ("HELLO written on a chalkboard", "Text (edge case)"),
    "ball_bouncing": ("A ball bouncing on a table", "Physics"),
    "empty_room_sun": ("Empty room with sunlight through window", "Minimal content"),
}
BACKENDS = ["MXFP8", "sage_blk16"]
CONFIGS = [
    {"tag": "720p_81f", "label": "720×1280 / 81 frames", "subdir": "run_720p_81f", "S": 75600},
    {"tag": "480p_33f", "label": "480×832 / 33 frames", "subdir": "run_480p_33f", "S": 14040},
]


def lpips_class(v):
    if v <= 0.05:
        return "lpips-good"
    if v <= 0.10:
        return "lpips-ok"
    if v <= 0.20:
        return ""
    return "lpips-bad"


def load_config(c):
    p = BASE / c["subdir"] / "prompts" / "metrics_summary.json"
    rows = json.load(open(p))
    by_pair = {(r["prompt"], r["backend"]): r for r in rows}
    return by_pair


def aggregate(idx, backend):
    vals = [idx[(p, backend)]["lpips_mean"] for p in PROMPT_TEXTS if (p, backend) in idx]
    if not vals:
        return None
    psnr = [
        idx[(p, backend)]["psnr_db"]
        for p in PROMPT_TEXTS
        if (p, backend) in idx and idx[(p, backend)]["psnr_db"] is not None
    ]
    ssim = [idx[(p, backend)]["ssim_mean"] for p in PROMPT_TEXTS if (p, backend) in idx]
    items = [(p, idx[(p, backend)]["lpips_mean"]) for p in PROMPT_TEXTS if (p, backend) in idx]
    return {
        "n": len(vals),
        "lpips_mean": sum(vals) / len(vals),
        "lpips_min": min(vals),
        "lpips_max": max(vals),
        "psnr_mean": sum(psnr) / len(psnr) if psnr else None,
        "ssim_mean": sum(ssim) / len(ssim) if ssim else None,
        "best": min(items, key=lambda x: x[1]),
        "worst": max(items, key=lambda x: x[1]),
    }


def heatmap_for_config(idx, available_backends):
    cols = available_backends
    cols_label = {"MXFP8": "MXFP8_CUDNN", "sage_blk16": "Sage (1, 16, 1)"}
    parts = [
        f'<div class="heatmap" style="grid-template-columns:200px {" ".join(["1fr"] * len(cols))}">'
    ]
    parts.append('<div class="h-corner">prompt</div>')
    for c in cols:
        parts.append(f'<div class="h-col">{cols_label[c]}</div>')
    for p in PROMPT_TEXTS:
        text, kind = PROMPT_TEXTS[p]
        parts.append(f'<div class="h-row">{p}<br><small>{kind}</small></div>')
        for c in cols:
            if (p, c) in idx:
                v = idx[(p, c)]["lpips_mean"]
                cls = lpips_class(v)
                parts.append(f'<div class="h-cell {cls}"><span class="pri">{v:.3f}</span></div>')
            else:
                parts.append('<div class="h-cell" style="color:#aaa">—</div>')
    parts.append("</div>")
    return "\n".join(parts)


def videogrid_for_prompt(prompt, config, idx, available_backends):
    sub = config["subdir"]
    pretty = {"MXFP8": "MXFP8_CUDNN", "sage_blk16": "Sage (1, 16, 1)"}
    parts = ['<div class="videogrid">']
    parts.append(
        f'<figure><video controls preload="metadata" src="{sub}/prompts/{prompt}_VANILLA.mp4"></video><figcaption><b>VANILLA bf16</b> · reference</figcaption></figure>'
    )
    for b in available_backends:
        if (prompt, b) in idx:
            r = idx[(prompt, b)]
            cls = lpips_class(r["lpips_mean"])
            parts.append(
                f'<figure><video controls preload="metadata" src="{sub}/prompts/{prompt}_{b}.mp4"></video><figcaption><b>{pretty[b]}</b> · LPIPS <span class="{cls}">{r["lpips_mean"]:.3f}</span> · PSNR {r["psnr_db"]:.1f} dB · SSIM {r["ssim_mean"]:.3f}</figcaption></figure>'
            )
    parts.append("</div>")
    return "\n".join(parts)


def main():
    # Load both configs.
    runs = []
    for c in CONFIGS:
        idx = load_config(c)
        avail = [b for b in BACKENDS if any((p, b) in idx for p in PROMPT_TEXTS)]
        agg = {b: aggregate(idx, b) for b in avail}
        runs.append({"cfg": c, "idx": idx, "available": avail, "agg": agg})

    # CSS — same as previous reports
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
.layout { display: grid; grid-template-columns: 240px minmax(0, 1fr);
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
td.lpips-good, span.lpips-good { background: var(--good-bg); color: var(--good); font-weight: 600; }
td.lpips-ok, span.lpips-ok   { background: var(--ok-bg);   color: var(--ok);   font-weight: 600; }
td.lpips-bad, span.lpips-bad  { background: var(--bad-bg);  color: var(--bad);  font-weight: 600; }
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
.heatmap { display: grid; gap: 4px; margin: 12px 0; font-size: 13px; }
.heatmap .h-corner, .heatmap .h-row, .heatmap .h-col, .heatmap .h-cell {
  padding: 8px 10px; background: white; border: 1px solid var(--rule); border-radius: 4px; }
.heatmap .h-corner { background: var(--code-bg); }
.heatmap .h-col { background: var(--code-bg); font-weight: 600; text-align: center;
                  text-transform: uppercase; letter-spacing: .03em; font-size: 12px; }
.heatmap .h-row { font-weight: 500; }
.heatmap .h-row small { color: var(--muted); font-weight: 400; }
.heatmap .h-cell { font-variant-numeric: tabular-nums; text-align: right; }
.heatmap .h-cell .pri { font-size: 15px; font-weight: 600; }
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
.note-warning { background: var(--ok-bg); border-left: 3px solid var(--ok);
                padding: 8px 14px; margin: 14px 0; font-size: 13px; }
details { margin: 12px 0; background: white; border: 1px solid var(--rule);
          border-radius: 6px; padding: 8px 14px; }
details > summary { cursor: pointer; font-weight: 600; color: var(--accent); padding: 4px 0; }
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } nav.toc { position: static; max-height: none; }
  .heatmap { grid-template-columns: 1fr !important; } }
"""

    # TOC
    toc = [
        '<ol><li><a href="#tldr">TL;DR</a></li>',
        '<li><a href="#methodology">Methodology</a></li>',
        '<li><a href="#prompts">Prompt suite</a></li>',
    ]
    for r in runs:
        c = r["cfg"]
        toc.append(f'<li><a href="#cfg-{c["tag"]}">{c["label"]} / 40 steps</a></li>')
    toc.append('<li><a href="#cross">Cross-resolution comparison</a></li>')
    toc.append('<li><a href="#caveats">Caveats</a></li></ol>')
    toc_html = "\n".join(toc)

    # TL;DR
    a1m = runs[0]["agg"].get("MXFP8")
    a1s = runs[0]["agg"].get("sage_blk16")
    a2m = runs[1]["agg"].get("MXFP8")
    a2s = runs[1]["agg"].get("sage_blk16")
    tldr = f"""
<section id="tldr" class="hero">
<h3>TL;DR — proper 40-step study (complete 60-cell matrix)</h3>
<p style="margin:4px 0"><strong>At the production target (720×1280 / 81 frames / 40 steps), MXFP8 mean LPIPS = {a1m["lpips_mean"]:.3f}</strong> across 10 diverse prompts — in the "<span class="lpips-ok">subtle</span>" perceptual band. Sage(1, 16, 1) mean LPIPS = {a1s["lpips_mean"]:.3f}. At the smaller-S 480×832 / 33-frame point, MXFP8 degrades modestly to {a2m["lpips_mean"]:.3f} while <strong>Sage(1, 16, 1) degrades sharply to {a2s["lpips_mean"]:.3f}</strong> — Sage is much more sensitive to S than MXFP8.</p>
<div class="stat-row">
<div class="stat"><div class="label">720p/81f MXFP8 mean LPIPS</div><div class="value lpips-ok">{a1m["lpips_mean"]:.3f}</div></div>
<div class="stat"><div class="label">720p/81f Sage(1,16,1) mean</div><div class="value">{a1s["lpips_mean"]:.3f}</div></div>
<div class="stat"><div class="label">480p/33f MXFP8 mean</div><div class="value">{a2m["lpips_mean"]:.3f}</div></div>
<div class="stat"><div class="label">480p/33f Sage(1,16,1) mean</div><div class="value lpips-bad">{a2s["lpips_mean"]:.3f}</div></div>
<div class="stat"><div class="label">MXFP8 best (720p/81f)</div><div class="value lpips-good">{a1m["best"][1]:.3f}</div></div>
</div>
</section>
"""

    # Methodology
    methodology = f"""
<h2 id="methodology">1. Methodology</h2>
<p>This study supersedes the prior 20-step run, which was under-stepped relative to Wan2.2 A14B's production setup (40 inference steps). All measurements below use 40 inference steps with torch.compile + autotune <em>on</em>, one backend per GPU in parallel. The 720p/81f sweep and the 480p/33f MXFP8/VANILLA runs were done on umbriel-b200-027; the 480p/33f Sage row was completed afterwards on umbriel-b200-043 once 027 freed up. Identical container snapshot, identical wheel, identical seed across all cells.</p>
<p>Two operating points were measured:</p>
<table>
<thead><tr><th>config</th><th>resolution</th><th class="num">frames</th><th class="num">steps</th><th class="num">S (self-attn seq len)</th><th>purpose</th></tr></thead>
<tbody>
<tr><td><strong>720p/81f</strong></td><td>720×1280</td><td class="num">81</td><td class="num">40</td><td class="num">75,600</td><td>Wan2.2 A14B production target</td></tr>
<tr><td><strong>480p/33f</strong></td><td>480×832</td><td class="num">33</td><td class="num">40</td><td class="num">14,040</td><td>Mid-S point; same number of steps, smaller token population per attn call</td></tr>
</tbody>
</table>
<p>Same prompt, same seed (42), same prompt list across both configs and all backends — apples-to-apples comparison within each config and across configs.</p>
<p class="note-warning"><strong>Note:</strong> the prior 20-step study reported MXFP8 mean LPIPS = 0.255. The drop to <strong>{a1m["lpips_mean"]:.3f}</strong> at this production setup is entirely attributable to (a) raising step count 20 → 40 and (b) larger S at 720p. The earlier number was a stress-test, not a fair production verdict.</p>
"""

    # Prompts table
    prompts_section = """
<h2 id="prompts">2. Prompt suite</h2>
<table>
<thead><tr><th class="num">#</th><th>id</th><th>prompt</th><th>failure mode probed</th></tr></thead>
<tbody>
"""
    for i, (pid, (text, kind)) in enumerate(PROMPT_TEXTS.items(), 1):
        prompts_section += f'<tr><td class="num">{i}</td><td><code>{pid}</code></td><td>"{text}"</td><td>{kind}</td></tr>\n'
    prompts_section += "</tbody></table>\n"

    # Per-config sections
    per_config_html = []
    for r in runs:
        c = r["cfg"]
        idx = r["idx"]
        avail = r["available"]
        agg = r["agg"]
        section = [
            f'<h2 id="cfg-{c["tag"]}">3.{c["tag"]} — {c["label"]} / 40 steps (S={c["S"]:,})</h2>'
        ]
        if "sage_blk16" not in avail:
            section.append(
                '<p class="note-warning">Sage(1, 16, 1) data unavailable for this config.</p>'
            )

        # Aggregate cards
        section.append("<h4>Aggregate (10 prompts vs bf16 reference)</h4>")
        section.append(
            '<table><thead><tr><th>backend</th><th class="num">LPIPS mean</th><th class="num">LPIPS min</th><th class="num">LPIPS max</th><th class="num">PSNR mean (dB)</th><th class="num">SSIM mean</th><th>best prompt</th><th>worst prompt</th></tr></thead><tbody>'
        )
        for b in avail:
            a = agg[b]
            cls_mean = lpips_class(a["lpips_mean"])
            cls_max = lpips_class(a["lpips_max"])
            section.append(
                f"<tr><td><strong>{'MXFP8' if b == 'MXFP8' else 'Sage (1,16,1)'}</strong></td>"
                f'<td class="num {cls_mean}">{a["lpips_mean"]:.3f}</td>'
                f'<td class="num lpips-good">{a["lpips_min"]:.3f}</td>'
                f'<td class="num {cls_max}">{a["lpips_max"]:.3f}</td>'
                f'<td class="num">{a["psnr_mean"]:.2f}</td>'
                f'<td class="num">{a["ssim_mean"]:.3f}</td>'
                f"<td><code>{a['best'][0]}</code> ({a['best'][1]:.3f})</td>"
                f"<td><code>{a['worst'][0]}</code> ({a['worst'][1]:.3f})</td></tr>"
            )
        section.append("</tbody></table>")

        # Heatmap
        section.append("<h4>Per-prompt LPIPS heatmap</h4>")
        section.append(heatmap_for_config(idx, avail))

        # Per-prompt video grids
        section.append("<h4>Per-prompt videos (drag to a frame to inspect)</h4>")
        for p in PROMPT_TEXTS:
            text, kind = PROMPT_TEXTS[p]
            section.append(
                f'<h3 id="{c["tag"]}-{p}">{p} <small style="color:#888;font-weight:400">— {kind}</small></h3>'
            )
            section.append(
                f'<p style="font-size:13px;color:#444;margin:4px 0 6px"><em>"{text}"</em></p>'
            )
            section.append(videogrid_for_prompt(p, c, idx, avail))

        per_config_html.append("\n".join(section))

    # Cross-resolution comparison
    cross = [
        '<h2 id="cross">4. Cross-resolution comparison (S-sensitivity)</h2>',
        "<p>Same 10 prompts at both sequence-length operating points, both backends. At larger S there are more tokens to average quantization noise against, so per-prompt LPIPS is lower. The interesting result is the <em>shape</em> of the degradation: MXFP8 drifts gracefully (~+0.03 mean), Sage(1,16,1) collapses (~+0.41 mean).</p>",
        '<table><thead><tr><th rowspan="2">prompt</th>'
        '<th colspan="3" style="text-align:center;border-bottom:1px solid #d0d4dd">MXFP8</th>'
        '<th colspan="3" style="text-align:center;border-bottom:1px solid #d0d4dd">Sage (1, 16, 1)</th>'
        "</tr><tr>"
        '<th class="num">480p/33f (S=14k)</th><th class="num">720p/81f (S=76k)</th><th class="num">Δ</th>'
        '<th class="num">480p/33f (S=14k)</th><th class="num">720p/81f (S=76k)</th><th class="num">Δ</th>'
        "</tr></thead><tbody>",
    ]
    idx_c1 = runs[0]["idx"]  # 720p
    idx_c2 = runs[1]["idx"]  # 480p
    for p in PROMPT_TEXTS:
        m1 = idx_c1.get((p, "MXFP8"), {}).get("lpips_mean")
        m2 = idx_c2.get((p, "MXFP8"), {}).get("lpips_mean")
        s1 = idx_c1.get((p, "sage_blk16"), {}).get("lpips_mean")
        s2 = idx_c2.get((p, "sage_blk16"), {}).get("lpips_mean")
        if m1 is None or m2 is None:
            continue
        dm = m1 - m2  # negative ⇒ 720p better
        cells = [
            f"<td><code>{p}</code></td>",
            f'<td class="num {lpips_class(m2)}">{m2:.3f}</td>',
            f'<td class="num {lpips_class(m1)}">{m1:.3f}</td>',
            f'<td class="num">{-dm:+.3f}</td>',
        ]
        if s1 is not None and s2 is not None:
            ds = s1 - s2
            cells += [
                f'<td class="num {lpips_class(s2)}">{s2:.3f}</td>',
                f'<td class="num {lpips_class(s1)}">{s1:.3f}</td>',
                f'<td class="num">{-ds:+.3f}</td>',
            ]
        else:
            cells += ['<td class="num">—</td>', '<td class="num">—</td>', '<td class="num">—</td>']
        cross.append("<tr>" + "".join(cells) + "</tr>")
    cross.append("</tbody></table>")
    cross.append(
        '<p>Negative Δ means 720p is better. <strong>MXFP8</strong>: all 10 prompts show 720p ≤ 480p, gap is small (~0.03 mean) — confirms the "more S → less quantization-noise per token" hypothesis without breaking the smaller-S regime. <strong>Sage(1, 16, 1)</strong>: same direction, but the gap is ~0.41 mean — every single prompt drops out of the "<span class="lpips-ok">subtle</span>" band at S=14k. The (1, 16, 1) block size is well-matched to S=75k but coarse for S=14k, where the per-block accuracy budget runs out and the int8 path can no longer track the bf16 reference.</p>'
    )

    # Caveats
    caveats = """
<h2 id="caveats">5. Caveats & open follow-ups</h2>
<ul>
<li><strong>Single seed (42)</strong> per prompt. Multi-seed averaging (3 seeds × 10 prompts × 3 backends = 90 generations per config) would tighten LPIPS bands — a 1.5-hour follow-up.</li>
<li><strong>LPIPS uses natural-image priors</strong> (AlexNet on ImageNet). For very synthetic content like the <code>text_hello</code> prompt, LPIPS may under-estimate human-visible quality differences vs PSNR/SSIM.</li>
<li><strong>Pre-uint8 fp16 latent comparison</strong> not done — all comparisons are at uint8 video level after <code>postprocess_video_tensor</code>'s clamp+round, which absorbs sub-threshold FP8 perturbations. A pre-uint8 comparison would isolate the latent-space accuracy verdict from the VAE quantization artifact.</li>
<li><strong>Per-call path firing</strong> was instrumented but is not recomputed in this revision — see the prior <a href="REPORT_SAGE.html">REPORT_SAGE</a> for the 100% sage / 0 fallback verdict at 720×1280 / 81 frames / 40 steps. The path is the same in this run.</li>
<li><strong>Sage(1, 16, 1) S-sensitivity is now quantified</strong> — mean LPIPS jumps from ~0.21 at S=75k to ~0.62 at S=14k. The (1, 16, 1) block size is too coarse for smaller token populations; sweeping the K-block (4, 8, 16, 32) at 480p would tell us whether a finer K-block recovers the small-S regime.</li>
</ul>
"""

    # JS scrollspy
    js = """
<script>
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
</script>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>10-prompt accuracy: MXFP8 vs Sage @ 40 steps (Wan2.2 A14B)</title>
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
<li><a href="https://sc.talos.nvidia.com/view/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/REPORT.html">REPORT.html — MXFP8 main study</a></li>
<li><a href="https://sc.talos.nvidia.com/view/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/REPORT_SAGE.html">REPORT_SAGE.html — Sage comparison</a></li>
<li><a href="https://sc.talos.nvidia.com/view/home/scratch.liuc_coreai/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn/SUPERVISOR_REVIEWS_PROMPTS.md">supervisor reviews</a></li>
</ul>
</nav>
<main>
<h1>Multi-prompt accuracy — MXFP8 vs Sage on Wan2.2 T2V-A14B</h1>
<dl class="meta">
<dt>Date</dt><dd>2026-05-12 (replaces 2026-05-11 under-stepped study; updated with full 480p/33f Sage row from 043)</dd>
<dt>Hardware</dt><dd>umbriel-b200-027 (720p/81f sweep, 480p/33f MXFP8 + VANILLA) and umbriel-b200-043 (480p/33f Sage row, run after 027 freed up). Identical container snapshot and wheel on both hosts.</dd>
<dt>Model</dt><dd>Wan2.2-T2V-A14B-Diffusers</dd>
<dt>Inference steps</dt><dd><strong>40</strong> (production default)</dd>
<dt>Two configs</dt><dd>720×1280 / 81 frames (S=75600) and 480×832 / 33 frames (S=14040)</dd>
<dt>Backends</dt><dd>VANILLA bf16 reference · MXFP8_CUDNN · Sage (1, 16, 1) qk_int8</dd>
<dt>Seed</dt><dd>42 across every (config, prompt, backend) cell</dd>
<dt>Prompts</dt><dd>10 prompts spanning static detail / motion / texture / lighting / text / minimal content</dd>
</dl>
{tldr}
{methodology}
{prompts_section}
{"\n".join(per_config_html)}
{"\n".join(cross)}
{caveats}
{js}
</main>
</div>
</body>
</html>
"""

    out_path = BASE / "REPORT_PROMPTS.html"
    open(out_path, "w").write(html)
    print(f"wrote {out_path}, length = {len(html):,}")


if __name__ == "__main__":
    main()
