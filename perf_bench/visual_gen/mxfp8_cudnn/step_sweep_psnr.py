"""Step-count sweep for fixed prompt/seed/resolution.

For each backend, reuse a single VisualGen instance so model load /
cuDNN graph build is paid once, then run the same generation at
multiple step counts. Save raw frames to .npy, then compare each
step count's MXFP8 vs VANILLA reference.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

WORKER_TEMPLATE = r"""
import os, sys, time, json, math
import numpy as np
import torch
import imageio

os.environ.setdefault("HOME", "/tmp")

from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams, logger
logger.set_level("info")

CFG = json.loads(os.environ["WAN_SWEEP_CFG"])
backend = CFG["backend"]
prompt = CFG["prompt"]
height, width, num_frames = CFG["height"], CFG["width"], CFG["num_frames"]
seeds = CFG["seeds"]
step_counts = CFG["step_counts"]
out_dir = CFG["out_dir"]
model_path = CFG["model_path"]
fps = CFG.get("fps", 16)
out_tag = CFG.get("out_tag", backend)
sage_cfg = CFG.get("sage_attention_config")

attention_cfg = {"backend": backend}
if sage_cfg is not None:
    attention_cfg["sage_attention_config"] = sage_cfg

args = VisualGenArgs(
    attention=attention_cfg,
    parallel={"dit_cfg_size": 1, "dit_ulysses_size": 1, "enable_parallel_vae": False},
    cuda_graph={"enable_cuda_graph": False},
    torch_compile={"enable_torch_compile": False, "enable_autotune": False, "enable_fullgraph": False},
)

t0 = time.time()
vg = VisualGen(model=model_path, args=args)
print(f"[sweep/{backend}] init={time.time()-t0:.1f}s", flush=True)

results = []
try:
    for seed in seeds:
        for steps in step_counts:
            t1 = time.time()
            out = vg.generate(
                inputs=prompt,
                params=VisualGenParams(
                    height=height, width=width, num_inference_steps=steps,
                    seed=seed, num_frames=num_frames, frame_rate=float(fps),
                ),
            )
            gen = time.time() - t1
            video = out.video
            video_np = video.cpu().numpy() if hasattr(video, "cpu") else np.asarray(video)
            if video_np.ndim == 5 and video_np.shape[0] == 1:
                video_np = video_np[0]
            tag = f"{out_tag}_seed{seed}_steps{steps}"
            np.save(os.path.join(out_dir, f"{tag}.npy"), video_np)
            mp4 = os.path.join(out_dir, f"{tag}.mp4")
            w = imageio.get_writer(mp4, fps=int(fps), codec="libx264",
                                   macro_block_size=1, output_params=["-pix_fmt","yuv420p"])
            for fr in video_np:
                w.append_data(fr)
            w.close()
            print(f"[sweep/{backend}] seed={seed} steps={steps} gen={gen:.1f}s "
                  f"frames={video_np.shape[0]} -> {tag}.mp4", flush=True)
            results.append({"backend": backend, "seed": seed, "steps": steps,
                            "gen_s": gen, "out": tag})
finally:
    vg.shutdown()

with open(os.path.join(out_dir, f"sweep_{out_tag}.json"), "w") as f:
    json.dump(results, f, indent=2)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--prompt",
        default="A close-up of a giant panda calmly eating bamboo "
        "in a misty forest, photorealistic.",
    )
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num_frames", type=int, default=9)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--step_counts", nargs="+", type=int, default=[2, 5, 10, 20, 40])
    ap.add_argument("--backends", nargs="+", default=["VANILLA", "MXFP8_CUDNN"])
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument(
        "--sage_blk_k",
        type=int,
        default=0,
        help="If >0, use Sage attention with sage_attention_config="
        "(1, sage_blk_k, 1, qk_int8=True). Implies backend=TRTLLM."
        " Output filenames are tagged sage_blk{N}.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / "step_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    for backend in args.backends:
        sage_attention_config = None
        out_tag = backend
        if args.sage_blk_k > 0:
            assert backend == "TRTLLM", "sage requires backend=TRTLLM"
            sage_attention_config = {
                "num_elts_per_blk_q": 1,
                "num_elts_per_blk_k": args.sage_blk_k,
                "num_elts_per_blk_v": 1,
                "qk_int8": True,
            }
            out_tag = f"sage_blk{args.sage_blk_k}"
        cfg = dict(
            backend=backend,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seeds=args.seeds,
            step_counts=args.step_counts,
            out_dir=str(out_dir),
            model_path=args.model_path,
            sage_attention_config=sage_attention_config,
            out_tag=out_tag,
        )
        env = os.environ.copy()
        env["WAN_SWEEP_CFG"] = json.dumps(cfg)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        log_path = out_dir / f"sweep_{out_tag}.log"
        print(f"--> launching {out_tag} (backend={backend}) -> {log_path}", flush=True)
        with open(log_path, "wb") as logf:
            proc = subprocess.run(
                [sys.executable, "-c", WORKER_TEMPLATE],
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
        print(f"--> {backend} exit={proc.returncode}", flush=True)


if __name__ == "__main__":
    main()
