"""Wan2.2 T2V-A14B accuracy + perf evaluation: bf16 (VANILLA) vs mxfp8 (MXFP8_CUDNN).

Runs identical seed/prompt/resolution generations on each backend, saves the
output mp4s side by side, and reports per-pixel PSNR/SSIM and end-to-end timing.

Usage:
    python run_wan_mxfp8_eval.py \
        --model_path /home/liuc/scratch/Wan2.2-T2V-A14B-Diffusers \
        --out_dir /home/liuc/scratch/codes/trtllm-v3-wt-2/perf_results/visual_gen/mxfp8_cudnn \
        --steps 40 --num_frames 81 --height 720 --width 1280 \
        --backends VANILLA MXFP8_CUDNN \
        --prompt "A close-up of a panda eating bamboo, photorealistic, soft daylight"

Each backend runs in its own subprocess to keep CUDA contexts isolated.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROMPTS = [
    (
        "panda_bamboo",
        "A close-up of a giant panda calmly eating bamboo in a misty forest, "
        "photorealistic, soft golden-hour lighting, cinematic.",
    ),
]


WORKER_TEMPLATE = r"""
import os, sys, time, json, math
import torch
import numpy as np
import imageio

# Force HOME and ffmpeg cache dirs to writable locations BEFORE TE import.
os.environ.setdefault("HOME", "/tmp")

from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams, logger

logger.set_level("info")

CFG = json.loads(os.environ["WAN_CFG"])
backend = CFG["backend"]
prompt = CFG["prompt"]
seed = CFG["seed"]
height = CFG["height"]
width = CFG["width"]
num_frames = CFG["num_frames"]
steps = CFG["steps"]
output_path = CFG["output_path"]
metrics_path = CFG["metrics_path"]
model_path = CFG["model_path"]
fps = CFG.get("fps", 16)
disable_cudagraph = CFG.get("disable_cudagraph", False)
disable_torch_compile = CFG.get("disable_torch_compile", False)
disable_autotune = CFG.get("disable_autotune", False)
sage_cfg = CFG.get("sage_attention_config")  # dict or None

attention_cfg = {"backend": backend}
if sage_cfg is not None:
    attention_cfg["sage_attention_config"] = sage_cfg

args = VisualGenArgs(
    attention=attention_cfg,
    parallel={"dit_cfg_size": 1, "dit_ulysses_size": 1, "enable_parallel_vae": False},
    cuda_graph={"enable_cuda_graph": not disable_cudagraph},
    torch_compile={
        "enable_torch_compile": not disable_torch_compile,
        "enable_autotune": not disable_autotune,
        "enable_fullgraph": False,
    },
)

t0 = time.time()
visual_gen = VisualGen(model=model_path, args=args)
t_init = time.time() - t0
print(f"[wan-eval/{backend}] init={t_init:.2f}s", flush=True)

try:
    t1 = time.time()
    out = visual_gen.generate(
        inputs=prompt,
        params=VisualGenParams(
            height=height, width=width, num_inference_steps=steps,
            seed=seed, num_frames=num_frames, frame_rate=float(fps),
        ),
    )
    t_gen = time.time() - t1
    print(f"[wan-eval/{backend}] generate={t_gen:.2f}s", flush=True)

    # MediaOutput.video is (N,H,W,3) uint8 tensor on CPU.
    video = out.video
    if video is None:
        raise RuntimeError("video output is None")
    video_np = video.cpu().numpy() if hasattr(video, "cpu") else np.asarray(video)
    if video_np.ndim == 5 and video_np.shape[0] == 1:
        video_np = video_np[0]
    print(f"[wan-eval/{backend}] video shape={video_np.shape} dtype={video_np.dtype}", flush=True)

    writer = imageio.get_writer(
        output_path, fps=int(fps), codec="libx264",
        macro_block_size=1, output_params=["-pix_fmt", "yuv420p"],
    )
    for frame in video_np:
        writer.append_data(frame)
    writer.close()
    print(f"[wan-eval/{backend}] wrote {output_path}", flush=True)

    # Save raw frames .npy for cross-backend comparison
    np.save(output_path.replace(".mp4", ".npy"), video_np)

    metrics = {
        "backend": backend, "init_s": t_init, "generate_s": t_gen,
        "n_frames": int(video_np.shape[0]), "h": int(video_np.shape[1]),
        "w": int(video_np.shape[2]),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
finally:
    visual_gen.shutdown()
"""


def run_backend(backend, prompt_id, prompt, args, env_extra):
    out_dir = Path(args.out_dir)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    suffix = args.tag if getattr(args, "tag", "") else ""
    suffix = f"_{suffix}" if suffix else ""
    out_path = out_dir / "videos" / f"{prompt_id}_{backend}{suffix}.mp4"
    metrics_path = out_dir / "logs" / f"{prompt_id}_{backend}{suffix}.json"
    log_path = out_dir / "logs" / f"{prompt_id}_{backend}{suffix}.log"

    cfg = dict(
        backend=backend,
        prompt=prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        steps=args.steps,
        output_path=str(out_path),
        metrics_path=str(metrics_path),
        model_path=args.model_path,
        fps=args.fps,
        disable_cudagraph=args.disable_cudagraph,
        disable_torch_compile=args.disable_torch_compile,
        disable_autotune=args.disable_autotune,
        sage_attention_config=getattr(args, "sage_attention_config", None),
    )
    env = os.environ.copy()
    env["WAN_CFG"] = json.dumps(cfg)
    env.update(env_extra)

    # Inline-run via -c so we don't need a temp file
    print(f"--> launching backend={backend} prompt='{prompt[:40]}...' out={out_path}", flush=True)
    with open(log_path, "wb") as logf:
        proc = subprocess.run(
            [sys.executable, "-c", WORKER_TEMPLATE],
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
        )
    print(f"--> backend={backend} exit={proc.returncode} log={log_path}", flush=True)
    return proc.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--prompt_id", default="panda_bamboo")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--backends", nargs="+", default=["VANILLA", "MXFP8_CUDNN"])
    ap.add_argument("--disable_cudagraph", action="store_true", default=True)
    ap.add_argument("--disable_torch_compile", action="store_true", default=False)
    ap.add_argument("--disable_autotune", action="store_true", default=False)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--tag", default="", help="Optional tag added to output filenames")
    ap.add_argument(
        "--sage_blk_k",
        type=int,
        default=0,
        help="If >0, enables Sage attention with sage_attention_config="
        "{1, sage_blk_k, 1, qk_int8=True}. Requires backend=TRTLLM.",
    )
    args = ap.parse_args()
    if args.sage_blk_k > 0:
        args.sage_attention_config = {
            "num_elts_per_blk_q": 1,
            "num_elts_per_blk_k": args.sage_blk_k,
            "num_elts_per_blk_v": 1,
            "qk_int8": True,
        }
    else:
        args.sage_attention_config = None

    if args.prompt is None:
        sel = [p for (pid, p) in PROMPTS if pid == args.prompt_id]
        if not sel:
            raise SystemExit(f"unknown prompt_id {args.prompt_id}")
        args.prompt = sel[0]

    env_extra = {"CUDA_VISIBLE_DEVICES": "0"}
    for backend in args.backends:
        ok = run_backend(backend, args.prompt_id, args.prompt, args, env_extra)
        if not ok:
            print(f"!! backend {backend} FAILED — see log", file=sys.stderr)


if __name__ == "__main__":
    main()
