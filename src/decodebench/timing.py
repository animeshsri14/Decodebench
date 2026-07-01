# timing.py — CUDA-event timing with adaptive K, warmup, and L2 replica sizing (v3 §8.2/§6.2)
from __future__ import annotations

import math
from typing import Callable


def n_weight_replicas(weight_bytes: int, l2_bytes: int | None = None) -> int:
    """N_copies = min(8, max(4, ceil(2 * L2_bytes / weight_bytes)))."""
    if l2_bytes is None:
        import torch

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        # Attribute was renamed l2_cache_size → L2_cache_size in PyTorch 2.x
        l2_bytes = getattr(props, "L2_cache_size", None) or getattr(props, "l2_cache_size")
    if weight_bytes <= 0:
        return 4
    return min(8, max(4, math.ceil(2 * l2_bytes / weight_bytes)))


def time_callable(
    fn: Callable[[], object],
    trials: int = 30,
    target_ms: float = 20.0,
    warmup: int = 50,
) -> list[float]:
    """us-per-invocation for each trial.  fn() performs exactly one invocation."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    stop.record()
    torch.cuda.synchronize()
    t_one_ms = max(start.elapsed_time(stop), 1e-4)

    k = max(200, math.ceil(target_ms / t_one_ms))

    out: list[float] = []
    for _ in range(trials):
        start.record()
        for _ in range(k):
            fn()
        stop.record()
        torch.cuda.synchronize()
        out.append((start.elapsed_time(stop) / k) * 1000.0)
    return out
