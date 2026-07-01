# graph.py — torch.cuda.CUDAGraph capture/replay with explicit failure detection
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable


@dataclass
class Captured:
    ok: bool
    reason: str = ""
    _graph: object = None

    def replay(self) -> None:
        if not self.ok:
            raise RuntimeError(f"cannot replay: capture failed ({self.reason})")
        self._graph.replay()


def try_capture(body: Callable[[], None], warmup_iters: int = 3) -> Captured:
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup_iters):
            body()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            body()
    except Exception as exc:
        # A failed capture leaves the CUDA generator registered with the
        # aborted graph (graphsUsingGenerator non-empty). Neither manual_seed
        # nor set_rng_state clears this metadata — only a successful
        # capture_end() does. Performing a trivial no-op capture on a fresh
        # stream drives capture_end() to completion, evicting the stale
        # registration and restoring randn on PyTorch 2.x. (PyTorch bug.)
        _flush_rng_registration()
        return Captured(ok=False, reason=f"{type(exc).__name__}: {exc}")
    return Captured(ok=True, _graph=graph)


def _flush_rng_registration() -> None:
    """Evict stale graphsUsingGenerator entry left by a failed capture."""
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    dummy = torch.zeros(1, device="cuda")
    g = torch.cuda.CUDAGraph()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.cuda.stream(s):
            g.capture_begin()
            dummy.fill_(0.0)
            g.capture_end()
    torch.cuda.current_stream().wait_stream(s)
