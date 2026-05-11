"""One-shot patch: re-add MXFP8_CUDNN factory branch to wheel's utils.py
after the PR #13570 Sage overlay overwrote it."""

p = "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/visual_gen/attention_backend/utils.py"
s = open(p).read()
if "MXFP8_CUDNN" in s:
    print("already patched")
else:
    old = """    elif backend_name == "FA4":
        return FlashAttn4Attention
    else:"""
    new = """    elif backend_name == "FA4":
        return FlashAttn4Attention
    elif backend_name == "MXFP8_CUDNN":
        from .mxfp8_cudnn import MXFP8CudnnAttention
        return MXFP8CudnnAttention
    else:"""
    if old in s:
        s = s.replace(old, new)
        open(p, "w").write(s)
        print("patched OK")
    else:
        print("pattern not found")
