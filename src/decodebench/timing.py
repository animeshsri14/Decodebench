# timing.py — CUDA-event timing with adaptive K, warmup, and L2 replica sizing (v3 §8.2/§6.2)
from __future__ import annotations

import math
from typing import Callable


def n_weight_replicas(weight_bytes: int, l2_bytes: int | None = None) -> int:
    """N_copies = min(8, max(4, ceil(2 * L2_bytes / weight_bytes)))."""
    if weight_bytes <= 0:
        raise ValueError("weight_bytes must be positive")
    if l2_bytes is None:
        import torch

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        # Attribute was renamed l2_cache_size → L2_cache_size in PyTorch 2.x
        l2_bytes = getattr(props, "L2_cache_size", None) or getattr(props, "l2_cache_size")
    if l2_bytes is None or l2_bytes < 0:
        raise ValueError("l2_bytes must be non-negative")
    return min(8, max(4, math.ceil(2 * l2_bytes / weight_bytes)))


def time_callable(
    fn: Callable[[], object],
    trials: int = 30,
    target_ms: float = 20.0,
    warmup: int = 50,
) -> list[float]:
    """us-per-invocation for each trial.  fn() performs exactly one invocation."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not math.isfinite(target_ms) or target_ms <= 0:
        raise ValueError("target_ms must be finite and positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    # Average a small probe batch instead of deriving K from one noisy launch.
    calibration_iters = 5
    start.record()
    for _ in range(calibration_iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    t_one_ms = max(start.elapsed_time(stop) / calibration_iters, 1e-4)

    # Avoid an unbounded enqueue loop if an event reports a near-zero probe.
    k = min(1_000_000, max(200, math.ceil(target_ms / t_one_ms)))

    out: list[float] = []
    for _ in range(trials):
        start.record()
        for _ in range(k):
            fn()
        stop.record()
        torch.cuda.synchronize()
        out.append((start.elapsed_time(stop) / k) * 1000.0)
    return out
