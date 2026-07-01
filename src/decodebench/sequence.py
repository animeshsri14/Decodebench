"""Linear decode-chain wrapper: register stages, execute in order, capture byte traces."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from decodebench.bytes_model import StageTrace
from decodebench.verdict import DEFAULT_BYTE_THRESHOLD


@dataclass
class _Stage:
    fn: Callable[..., Any]
    name: str
    params: list[str]


class Sequence:
    def __init__(self, name: str):
        self.name = name
        self._stages: list[_Stage] = []

    def stage(self, fn):
        self._stages.append(
            _Stage(fn, fn.__name__, list(inspect.signature(fn).parameters))
        )
        return fn

    def _bind_args(self, stage, idx, prev_out, inputs):
        args = []
        for j, p in enumerate(stage.params):
            if idx > 0 and j == 0:
                args.append(prev_out)
            else:
                if p not in inputs:
                    raise KeyError(
                        f"stage '{stage.name}' parameter '{p}' not in inputs={list(inputs)}"
                    )
                args.append(inputs[p])
        return args

    def _run_once(self, inputs, _check_temporaries=False):
        import warnings

        import torch

        if not self._stages:
            raise ValueError(
                "Sequence has no stages; register at least one with @seq.stage"
            )
        prev_out, per_stage_io = None, []
        for idx, st in enumerate(self._stages):
            args = self._bind_args(st, idx, prev_out, inputs)
            alloc_before = (
                torch.cuda.memory_stats()["allocated_bytes.all.allocated"]
                if _check_temporaries and torch.cuda.is_available()
                else None
            )
            out = st.fn(*args)
            if not isinstance(out, torch.Tensor):
                raise TypeError(
                    f"stage '{st.name}' must return a single torch.Tensor "
                    f"(got {type(out).__name__}). DecodeBench v1 supports "
                    "linear, single-output chains only."
                )
            if alloc_before is not None:
                alloc_after = torch.cuda.memory_stats()["allocated_bytes.all.allocated"]
                internal_mb = (alloc_after - alloc_before - out.nbytes) / (1024 * 1024)
                if internal_mb > 0.1:  # >100 KB triggers warning
                    warnings.warn(
                        f"stage '{st.name}' allocated {internal_mb:.1f} MB internally; "
                        f"byte model undercounts — split it into single-op stages",
                        stacklevel=2,
                    )
            per_stage_io.append(
                (st.name, [a for a in args if isinstance(a, torch.Tensor)], out)
            )
            prev_out = out
        return prev_out, per_stage_io

    def trace(self, inputs) -> list[StageTrace]:
        _, io = self._run_once(inputs, _check_temporaries=True)
        return [
            StageTrace(
                name,
                [t.nbytes for t in reads],
                out.nbytes,
                i == len(io) - 1,
            )
            for i, (name, reads, out) in enumerate(io)
        ]

    def profile(
        self,
        inputs,
        trials: int = 30,
        target_ms: float = 20.0,
        warmup: int = 50,
        byte_threshold: float = DEFAULT_BYTE_THRESHOLD,
        seed: int = 42,
        input_replicas: list[dict] | None = None,
    ):
        """input_replicas (RI-3): optional list of input dicts with identical
        shapes/dtypes; invocation i uses replica i % N to defeat L2 residency
        (v3 §6.2).  Demos pass weight-replicated dicts; user chains may omit
        (binary verdict unaffected).
        """
        import torch

        from decodebench.graph import try_capture
        from decodebench.report import Report
        from decodebench.timing import time_callable

        torch.manual_seed(seed)

        replicas = input_replicas if input_replicas else [inputs]
        traces = self.trace(replicas[0])

        counter = {"i": 0}

        def stream_body():
            inp = replicas[counter["i"] % len(replicas)]
            counter["i"] += 1
            prev = None
            for idx, st in enumerate(self._stages):
                prev = st.fn(*self._bind_args(st, idx, prev, inp))
            return prev

        stream_us = time_callable(
            stream_body, trials=trials, target_ms=target_ms, warmup=warmup
        )

        captured = [
            try_capture(lambda inp=inp: self._replay_body(inp)) for inp in replicas
        ]
        bad = next((c for c in captured if not c.ok), None)
        if bad is not None:
            return Report(
                self.name,
                stream_us,
                stream_us,
                traces,
                byte_threshold=byte_threshold,
                graph_ok=False,
                graph_skip_reason=bad.reason,
            )

        gcounter = {"i": 0}

        def graph_body():
            captured[gcounter["i"] % len(captured)].replay()
            gcounter["i"] += 1

        graph_us = time_callable(
            graph_body, trials=trials, target_ms=target_ms, warmup=warmup
        )
        return Report(self.name, stream_us, graph_us, traces, byte_threshold=byte_threshold)

    def _replay_body(self, inp):
        prev = None
        for idx, st in enumerate(self._stages):
            prev = st.fn(*self._bind_args(st, idx, prev, inp))
