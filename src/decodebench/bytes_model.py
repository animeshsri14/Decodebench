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
    consumers : int
        Number of downstream stage arguments that read this output.  A fully
        fused region eliminates the producer write plus one materialized read
        per consumer.  The default of one preserves the linear-chain/manual
        ``StageTrace`` contract.
    """

    name: str
    reads: list[int]
    write: int
    is_final: bool = False
    consumers: int = 1

    def __post_init__(self) -> None:
        if self.write < 0 or any(r < 0 for r in self.reads):
            raise ValueError("StageTrace byte counts must be non-negative")
        if self.consumers < 0:
            raise ValueError("StageTrace consumers must be non-negative")
        if not self.is_final and self.consumers == 0:
            raise ValueError("non-final StageTrace must have at least one consumer")


def total_bytes(traces: list[StageTrace]) -> int:
    """Total unfused memory traffic in bytes (reads + writes)."""
    return sum(sum(t.reads) + t.write for t in traces)


def eliminable_bytes(traces: list[StageTrace]) -> int:
    """Bytes that can be eliminated by fusion.

    For each non-final stage this is its write plus one same-sized read per
    declared consumer.  A linear chain therefore retains the familiar
    ``2 * write`` result, while fan-out is accounted rather than silently
    under-counted.
    """
    return sum((1 + t.consumers) * t.write for t in traces if not t.is_final)
