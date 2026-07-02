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


def try_capture(body: Callable[[], None], warmup_iters: int = 3, pool=None) -> Captured:
    """Capture ``body`` and return an explicit success/failure object.

    ``pool`` may be a ``torch.cuda.graph_pool_handle()`` shared by graphs that
    are guaranteed to replay sequentially in capture order.
    """
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup_iters):
            body()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, pool=pool):
            body()
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        # Best-effort recovery for a PyTorch 2.x quirk: a failed capture leaves
        # the CUDA generator registered with the aborted graph
        # (graphsUsingGenerator non-empty). Neither manual_seed nor
        # set_rng_state clears this metadata — only a successful capture_end()
        # does, so a trivial no-op capture evicts the stale registration.
        # This pokes at internal PyTorch behavior, so it is strictly guarded:
        # it runs only if the current stream is no longer in capture mode, and
        # any failure inside the cleanup is reported, never raised.
        cleanup_err = _flush_rng_registration()
        if cleanup_err:
            reason += (
                f" [additionally, RNG-registration cleanup failed: {cleanup_err}; "
                "torch.randn on this device may raise until the process restarts]"
            )
        return Captured(ok=False, reason=reason)
    return Captured(ok=True, _graph=graph)


def _flush_rng_registration() -> str:
    """Best-effort eviction of the stale graphsUsingGenerator entry left by a
    failed capture. Returns "" on success, or an error description; never raises.
    """
    import torch

    try:
        if torch.cuda.is_current_stream_capturing():
            # The aborted capture left the current stream in capture mode; a
            # synthetic capture here would nest inside it and make things worse.
            return "current stream is still in capture mode; skipped cleanup"
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        # Allocate the scratch tensor before capture begins (allocations inside
        # capture would be recorded into the graph); keep it alive past
        # capture_end() so the graph never references freed memory.
        dummy = torch.zeros(1, device="cuda")
        g = torch.cuda.CUDAGraph()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.cuda.stream(s):
                g.capture_begin()
                dummy.fill_(0.0)
                g.capture_end()
        torch.cuda.current_stream().wait_stream(s)
        del dummy, g
        return ""
    except Exception as exc:  # cleanup must never mask the original failure
        return f"{type(exc).__name__}: {exc}"
