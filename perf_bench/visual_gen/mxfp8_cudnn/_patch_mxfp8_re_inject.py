"""Re-inject MXFP8_CUDNN into wheel-installed config.py + utils.py after the
PR #13570 Sage overlay overwrote them. Idempotent."""

SITE = "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/visual_gen"

p = SITE + "/config.py"
s = open(p).read()
if "MXFP8_CUDNN" not in s:
    s = s.replace(
        'Literal["VANILLA", "TRTLLM", "FA4"]',
        'Literal["VANILLA", "TRTLLM", "FA4", "MXFP8_CUDNN"]',
    )
    open(p, "w").write(s)
    print("  patched config.py literal")
else:
    print("  config.py already has MXFP8_CUDNN")

p = SITE + "/attention_backend/utils.py"
s = open(p).read()
if "MXFP8_CUDNN" not in s:
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
        print("  patched utils.py factory")
    else:
        print("  utils.py pattern not found")
else:
    print("  utils.py already has MXFP8_CUDNN")
