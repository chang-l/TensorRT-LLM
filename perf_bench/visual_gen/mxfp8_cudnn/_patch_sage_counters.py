"""One-shot patch: add sage_calls / fallback_calls counters and per-call trace
to wheel's visual_gen/attention_backend/trtllm.py SageAttention path.
Mirrors the MXFP8CudnnAttention contract so the same audit pattern applies.

Adds two env vars:
  TRTLLM_VISUAL_GEN_SAGE_TRACE          - per-instance dump on __del__
  TRTLLM_VISUAL_GEN_SAGE_PER_CALL_TRACE - per-call append (timestamp, layer, shape, path)
"""

p = "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/visual_gen/attention_backend/trtllm.py"
s = open(p).read()

if "sage_calls" in s:
    print("already patched")
    raise SystemExit(0)

# 1) Add `import os, time` if missing (top of file)
if "\nimport os" not in s.split("class ", 1)[0]:
    # Insert os, time right after the SPDX header / first imports block.
    s = s.replace(
        "from typing import Optional",
        "import os\nimport time as _time\nfrom typing import Optional",
        1,
    )

# 2) Add counters init after the existing assignment "self.sage_attention_config = sage_attention_config"
s = s.replace(
    "        # SageAttention: presence of config object implies enablement\n"
    "        self.sage_attention_config = sage_attention_config",
    "        # SageAttention: presence of config object implies enablement\n"
    "        self.sage_attention_config = sage_attention_config\n"
    "        # Counters mirroring MXFP8CudnnAttention to verify Sage path actually fires.\n"
    "        self.sage_calls: int = 0\n"
    "        self.fallback_calls: int = 0\n"
    "        self.layer_idx = layer_idx\n",
)

# 3) Add __del__ trace dump method after the counters init.
# Find the end of __init__ — the class body before forward(). Insert __del__ before "@torch.compiler.disable" decorator.
del_method = """
    def __del__(self):
        try:
            path = os.environ.get("TRTLLM_VISUAL_GEN_SAGE_TRACE")
            if path:
                with open(path, "a") as f:
                    f.write(
                        f"layer_idx={getattr(self, 'layer_idx', '?')} "
                        f"sage_calls={getattr(self, 'sage_calls', 0)} "
                        f"fallback_calls={getattr(self, 'fallback_calls', 0)}\\n"
                    )
        except Exception:
            pass

    def _per_call_log(self, q, path):
        path_var = os.environ.get("TRTLLM_VISUAL_GEN_SAGE_PER_CALL_TRACE")
        if not path_var:
            return
        try:
            shape = tuple(q.shape)
            with open(path_var, "a") as f:
                f.write(
                    f"{_time.time():.6f} layer_idx={self.layer_idx} "
                    f"path={path} shape={shape} dtype={q.dtype}\\n"
                )
                f.flush()  # supervisor ask: tail must survive a mid-step crash
        except Exception:
            pass

"""
s = s.replace(
    "    # Needed to work with torch compile cause of attention metadata\n"
    "    # make attn metadata as input for it to work\n"
    "    @torch.compiler.disable",
    del_method + "    # Needed to work with torch compile cause of attention metadata\n"
    "    # make attn metadata as input for it to work\n"
    "    @torch.compiler.disable",
)

# 4) Inside forward(): bump counters in both Sage branch and fallback branch.
s = s.replace(
    "        if self.sage_attention_config is not None:\n"
    "            assert k is not None and v is not None, (\n"
    '                "SageAttention requires separate Q, K, V tensors"\n'
    "            )",
    "        if self.sage_attention_config is not None:\n"
    "            assert k is not None and v is not None, (\n"
    '                "SageAttention requires separate Q, K, V tensors"\n'
    "            )\n"
    "            self.sage_calls += 1\n"
    '            self._per_call_log(q, "sage")',
)

s = s.replace(
    "        else:\n"
    "            if k is None and v is None:\n"
    "                qkv = q.reshape(batch_size * seq_len, -1)\n"
    "            else:\n"
    "                qkv = self._concat_qkv(q, k, v, batch_size, seq_len, kv_seq_len)",
    "        else:\n"
    "            self.fallback_calls += 1\n"
    '            self._per_call_log(q, "trtllm_bf16")\n'
    "            if k is None and v is None:\n"
    "                qkv = q.reshape(batch_size * seq_len, -1)\n"
    "            else:\n"
    "                qkv = self._concat_qkv(q, k, v, batch_size, seq_len, kv_seq_len)",
)

open(p, "w").write(s)
print("patched OK")
print("--- sanity ---")
print(
    "\n".join(
        [
            line
            for line in s.split("\n")
            if "sage_calls" in line or "fallback_calls" in line or "_per_call_log" in line
        ]
    )[:1000]
)
