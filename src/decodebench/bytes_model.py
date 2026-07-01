"""Analytic byte accounting from observed per-stage tensor I/O.

Each fusion pattern is modelled as a sequence of :class:`StageTrace` objects
representing the *unfused* pipeline.  A fused kernel eliminates the
intermediate write (and the downstream read of that write) for every stage
that is not the final consumer.

DecodeBench accounts bytes at declared stage boundaries. Stage-internal
temporaries (e.g., dtype casts, intermediate buffers) are invisible to
StageTrace and can cause the byte model to undercount real HBM traffic.
Sequence.trace() diffs CUDA memory stats around each stage call and emits
a warning if internal allocations exceed the declared output (RI-6).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageTrace:
    """A single stage in the unfused pipeline.

    Attributes
    ----------
    name : str
        Human-readable label (e.g. ``"rms_norm"``).
    reads : list[int]
        Bytes **read** by this stage from memory (includes weight reads).
    write : int
        Bytes **written** to memory by this stage.
    is_final : bool
        If ``True`` this write is the final output consumed downstream and
        is *not* eligible for elimination in the fused kernel.
    """

    name: str
    reads: list[int]
    write: int
    is_final: bool = False


def total_bytes(traces: list[StageTrace]) -> int:
    """Total unfused memory traffic in bytes (reads + writes)."""
    return sum(sum(t.reads) + t.write for t in traces)


def eliminable_bytes(traces: list[StageTrace]) -> int:
    """Bytes that can be eliminated by fusion.

    This is **2 ×** the write volume of every non-final stage, modelling
    that both the write and the downstream read are saved.
    """
    return 2 * sum(t.write for t in traces if not t.is_final)
