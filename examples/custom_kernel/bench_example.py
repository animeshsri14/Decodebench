# Usage:
#   cd examples/custom_kernel
#   pip install -e .
#   decodebench profile bench_example.py:build_unfused   # stream vs graph verdict
#   decodebench profile bench_example.py:build_fused     # measure your fused kernel

import torch
import custom_kernel
from decodebench.sequence import Sequence

# ── Unfused: two separate kernels ─────────────────────────────────────────────
# Run this first — the tool tells you whether fusion is worth it before you write it.

unfused = Sequence("scale_bias_unfused")

@unfused.stage
def scale(x, w):
    return custom_kernel.scale(x, w)

@unfused.stage
def bias(x, b):
    return custom_kernel.bias(x, b)

def build_unfused():
    n = 1 << 20
    x = torch.randn(n, device="cuda", dtype=torch.float16)
    w = torch.randn(n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, device="cuda", dtype=torch.float16)
    return unfused, {"x": x, "w": w, "b": b}


# ── Fused: single kernel doing both ops ───────────────────────────────────────
# Run this after writing your fused kernel to compare against the graph baseline.

fused = Sequence("scale_bias_fused")

@fused.stage
def scale_bias_fused(x, w, b):
    return custom_kernel.scale_bias_fused(x, w, b)

def build_fused():
    n = 1 << 20
    x = torch.randn(n, device="cuda", dtype=torch.float16)
    w = torch.randn(n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, device="cuda", dtype=torch.float16)
    return fused, {"x": x, "w": w, "b": b}
