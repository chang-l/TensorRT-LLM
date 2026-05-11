# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-prompt accuracy study driver.

Loads VisualGen ONCE per backend (model load amortized) and iterates through
a fixed set of prompts that exercise different failure modes (static detail,
fast motion, text rendering, etc.). One backend per process — wire the
launcher to fan out 3 backends to 3 GPUs in parallel.

Outputs per (prompt, backend) pair:
  <out_dir>/prompts/{prompt_id}_{backend_tag}.mp4
  <out_dir>/prompts/{prompt_id}_{backend_tag}.npy   # raw uint8 frames
  <out_dir>/prompts/{prompt_id}_{backend_tag}.json  # timing
"""

import argparse
import json
import os
import time
from pathlib import Path

# Prompt suite — focused on different failure modes.
PROMPTS = [
    ("cat_windowsill", "A cat sitting still on a windowsill, soft daylight"),
    ("busy_street", "A person walking across a busy street downtown, midday"),
    ("ocean_sunset", "Ocean waves crashing on rocks at sunset, dramatic lighting"),
    ("clouds_timelapse", "A timelapse of clouds moving slowly across a clear sky"),
    ("dancer_jump", "A dancer performing a fast spinning jump in a studio"),
    ("flower_blooming", "Close-up of a red rose blooming, fine petal detail"),
    ("drone_city_night", "A drone shot flying over a city skyline at night, neon lights"),
    ("text_hello", "The word 'HELLO' written in chalk on a green chalkboard"),
    ("ball_bouncing", "A red ball bouncing on a wooden table, white background"),
    ("empty_room_sun", "An empty white room with sunlight coming through a window"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["VANILLA", "TRTLLM", "MXFP8_CUDNN"])
    p.add_argument(
        "--backend_tag",
        required=True,
        help="Suffix for output files; e.g. VANILLA / MXFP8 / sage_blk16",
    )
    p.add_argument(
        "--sage_blk_k",
        type=int,
        default=0,
        help="If >0 + backend=TRTLLM, enable Sage with this K-block",
    )
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--model_path", required=True)
    p.add_argument("--out_dir", required=True, help="Parent dir; outputs go to <out_dir>/prompts/")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=9)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument(
        "--prompts", nargs="+", default=None, help="Restrict to these prompt IDs (default: all 10)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("HOME", "/tmp")
    # Per-call trace files (so we can verify path fired for each prompt).
    trace_dir = Path(args.out_dir) / "prompts" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "TRTLLM_VISUAL_GEN_MXFP8_PER_CALL_TRACE",
        str(trace_dir / f"per_call_{args.backend_tag}.txt"),
    )
    os.environ.setdefault(
        "TRTLLM_VISUAL_GEN_SAGE_PER_CALL_TRACE",
        str(trace_dir / f"per_call_{args.backend_tag}.txt"),
    )

    import imageio  # noqa: E402
    import numpy as np  # noqa: E402

    from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams, logger

    logger.set_level("warning")  # keep stdout quieter for the suite

    # Build attention config.
    attn_cfg = {"backend": args.backend}
    if args.sage_blk_k > 0:
        assert args.backend == "TRTLLM"
        attn_cfg["sage_attention_config"] = {
            "num_elts_per_blk_q": 1,
            "num_elts_per_blk_k": args.sage_blk_k,
            "num_elts_per_blk_v": 1,
            "qk_int8": True,
        }

    vg_args = VisualGenArgs(
        attention=attn_cfg,
        parallel={"dit_cfg_size": 1, "dit_ulysses_size": 1, "enable_parallel_vae": False},
        cuda_graph={"enable_cuda_graph": False},
        torch_compile={
            "enable_torch_compile": True,
            "enable_autotune": True,
            "enable_fullgraph": False,
        },
    )

    t0 = time.time()
    vg = VisualGen(model=args.model_path, args=vg_args)
    init_s = time.time() - t0
    print(f"[{args.backend_tag}] init={init_s:.1f}s on GPU {args.gpu_id}", flush=True)

    prompts = [(pid, p) for pid, p in PROMPTS if (args.prompts is None or pid in args.prompts)]
    out_root = Path(args.out_dir) / "prompts"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    try:
        for prompt_id, prompt in prompts:
            tag = f"{prompt_id}_{args.backend_tag}"
            mp4_path = out_root / f"{tag}.mp4"
            npy_path = out_root / f"{tag}.npy"
            json_path = out_root / f"{tag}.json"
            if mp4_path.exists() and npy_path.exists():
                print(f"[{args.backend_tag}] skip {prompt_id} (already done)", flush=True)
                continue
            t1 = time.time()
            try:
                out = vg.generate(
                    inputs=prompt,
                    params=VisualGenParams(
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.steps,
                        seed=args.seed,
                        num_frames=args.num_frames,
                        frame_rate=float(args.fps),
                    ),
                )
            except Exception as e:
                print(
                    f"[{args.backend_tag}] {prompt_id} FAILED: {type(e).__name__}: {e}", flush=True
                )
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "backend": args.backend_tag,
                        "error": str(e),
                        "gen_s": None,
                    }
                )
                continue
            gen_s = time.time() - t1
            video = out.video
            video_np = video.cpu().numpy() if hasattr(video, "cpu") else np.asarray(video)
            if video_np.ndim == 5 and video_np.shape[0] == 1:
                video_np = video_np[0]
            np.save(npy_path, video_np)
            writer = imageio.get_writer(
                mp4_path,
                fps=int(args.fps),
                codec="libx264",
                macro_block_size=1,
                output_params=["-pix_fmt", "yuv420p"],
            )
            for f in video_np:
                writer.append_data(f)
            writer.close()
            row = {
                "prompt_id": prompt_id,
                "backend": args.backend_tag,
                "prompt": prompt,
                "gen_s": gen_s,
                "n_frames": int(video_np.shape[0]),
                "h": int(video_np.shape[1]),
                "w": int(video_np.shape[2]),
            }
            with open(json_path, "w") as jf:
                json.dump(row, jf, indent=2)
            rows.append(row)
            print(f"[{args.backend_tag}] {prompt_id} gen={gen_s:.1f}s -> {tag}.mp4", flush=True)
    finally:
        vg.shutdown()

    summary_path = out_root / f"summary_{args.backend_tag}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "backend": args.backend,
                "backend_tag": args.backend_tag,
                "init_s": init_s,
                "rows": rows,
                "config": {
                    "height": args.height,
                    "width": args.width,
                    "num_frames": args.num_frames,
                    "steps": args.steps,
                    "seed": args.seed,
                    "sage_blk_k": args.sage_blk_k,
                },
            },
            f,
            indent=2,
        )
    print(f"[{args.backend_tag}] DONE — {len(rows)} prompts, summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
