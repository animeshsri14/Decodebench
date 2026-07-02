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
        sig = inspect.signature(fn)
        for p in sig.parameters.values():
            if p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                raise TypeError(
                    f"stage '{fn.__name__}': parameter '{p.name}' is "
                    f"{p.kind.description}; stages must use plain positional "
                    "parameters (no *args/**kwargs/keyword-only parameters — "
                    "DecodeBench binds by name and position explicitly)"
                )
        if any(s.name == fn.__name__ for s in self._stages):
            raise ValueError(f"duplicate stage name '{fn.__name__}'")
        self._stages.append(_Stage(fn, fn.__name__, list(sig.parameters)))
        return fn

    def _bind_args(self, stage, idx, prev_out, inputs, stage_outputs):
        """Resolve each parameter, in precedence order:
        1. an earlier stage's output whose stage name matches the parameter name;
        2. an entry in the `inputs` dict with that name;
        3. (first parameter of a non-first stage only) the previous stage's output.
        """
        args = []
        for j, p in enumerate(stage.params):
            if p in stage_outputs:
                args.append(stage_outputs[p])
            elif p in inputs:
                args.append(inputs[p])
            elif idx > 0 and j == 0:
                args.append(prev_out)
            else:
                raise KeyError(
                    f"stage '{stage.name}' parameter '{p}' not found in earlier "
                    f"stage outputs {list(stage_outputs)} or inputs {list(inputs)}"
                )
        return args

    def _dependency_consumer_counts(self, inputs) -> dict[str, int]:
        """Validate static bindings and count reads of every stage output.

        Keeping this independent of tensor execution makes malformed DAGs fail
        before a potentially expensive GPU stage runs and gives byte accounting
        an explicit fan-out multiplicity.
        """
        stage_names: list[str] = []
        counts: dict[str, int] = {}
        for idx, st in enumerate(self._stages):
            for j, p in enumerate(st.params):
                if p in stage_names:
                    counts[p] = counts.get(p, 0) + 1
                elif p in inputs:
                    continue
                elif idx > 0 and j == 0:
                    prev_name = stage_names[-1]
                    counts[prev_name] = counts.get(prev_name, 0) + 1
                else:
                    raise KeyError(
                        f"stage '{st.name}' parameter '{p}' not found in earlier "
                        f"stage outputs {stage_names} or inputs {list(inputs)}"
                    )
            stage_names.append(st.name)
        unconsumed = [name for name in stage_names[:-1] if counts.get(name, 0) == 0]
        if unconsumed:
            raise ValueError(
                f"non-final stage output(s) {unconsumed} are never consumed by a "
                "later stage; the eliminable-byte model assumes every non-final "
                "write has a downstream read, so this chain is not a valid "
                "DecodeBench model"
            )
        return counts

    def _run_once(self, inputs, _check_temporaries=False):
        import warnings

        import torch

        if not self._stages:
            raise ValueError(
                "Sequence has no stages; register at least one with @seq.stage"
            )
        consumer_counts = self._dependency_consumer_counts(inputs)
        prev_out, per_stage_io = None, []
        stage_outputs: dict[str, Any] = {}
        for idx, st in enumerate(self._stages):
            args = self._bind_args(st, idx, prev_out, inputs, stage_outputs)
            hidden = self._closure_tensors(st.fn, torch)
            if hidden:
                raise ValueError(
                    f"stage '{st.name}' captures tensor(s) {hidden} outside its "
                    "declared parameters; pass them through inputs so byte "
                    "accounting can see them"
                )
            for j, arg in enumerate(args):
                if isinstance(arg, torch.Tensor) and not arg.is_contiguous():
                    raise ValueError(
                        f"stage '{st.name}' tensor argument {j} is non-contiguous "
                        f"(shape={tuple(arg.shape)}, stride={arg.stride()}); "
                        "materialize it explicitly as a declared stage"
                    )
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
            if not out.is_contiguous():
                raise ValueError(
                    f"stage '{st.name}' returned a non-contiguous tensor "
                    f"(shape={tuple(out.shape)}, stride={out.stride()}); materialize "
                    "it explicitly before using DecodeBench byte accounting"
                )
            out_storage = self._storage_data_ptr(out)
            aliased_args = [
                j for j, arg in enumerate(args)
                if isinstance(arg, torch.Tensor)
                and out.nbytes > 0 and arg.nbytes > 0
                and self._storage_data_ptr(arg) == out_storage
            ]
            if aliased_args:
                raise ValueError(
                    f"stage '{st.name}' output aliases tensor argument(s) {aliased_args}; "
                    "in-place/view outputs do not represent a materialized stage write"
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
            stage_outputs[st.name] = out
            prev_out = out
        return prev_out, per_stage_io, consumer_counts

    @staticmethod
    def _storage_data_ptr(tensor) -> int:
        storage = getattr(tensor, "untyped_storage", None)
        return storage().data_ptr() if storage is not None else tensor.storage().data_ptr()

    @staticmethod
    def _closure_tensors(fn, torch_module) -> list[str]:
        """Names of directly captured tensor values invisible to parameters."""
        closure = inspect.getclosurevars(fn)
        found = []
        for scope, values in (("nonlocal", closure.nonlocals), ("global", closure.globals)):
            for name, value in values.items():
                if isinstance(value, torch_module.Tensor):
                    found.append(f"{scope}:{name}")
        return found

    def trace(self, inputs) -> list[StageTrace]:
        _, io, consumer_counts = self._run_once(inputs, _check_temporaries=True)
        return [
            StageTrace(
                name,
                [t.nbytes for t in reads],
                out.nbytes,
                i == len(io) - 1,
                0 if i == len(io) - 1 else consumer_counts[name],
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
        self._validate_replicas(replicas)
        traces = self.trace(replicas[0])

        counter = {"i": 0}

        def stream_body():
            inp = replicas[counter["i"] % len(replicas)]
            counter["i"] += 1
            return self._replay_body(inp)

        stream_us = time_callable(
            stream_body, trials=trials, target_ms=target_ms, warmup=warmup
        )

        # Replica graphs always replay sequentially in this same round-robin
        # order, so they may safely share a graph-private allocator pool rather
        # than multiplying intermediate-allocation memory by replica count.
        pool_factory = getattr(torch.cuda, "graph_pool_handle", None)
        graph_pool = pool_factory() if pool_factory is not None else None
        captured = [
            try_capture(lambda inp=inp: self._replay_body(inp), pool=graph_pool)
            for inp in replicas
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
        stage_outputs = {}
        for idx, st in enumerate(self._stages):
            prev = st.fn(*self._bind_args(st, idx, prev, inp, stage_outputs))
            stage_outputs[st.name] = prev
        return prev

    @staticmethod
    def _validate_replicas(replicas: list[dict]) -> None:
        """All replicas must be structurally identical to replica 0: same keys,
        and per-key matching shape/dtype/stride/device for tensor values."""
        import torch

        base = replicas[0]
        for i, rep in enumerate(replicas[1:], start=1):
            if set(rep) != set(base):
                raise ValueError(
                    f"input replica {i} keys {sorted(rep)} != replica 0 keys {sorted(base)}"
                )
            for k, v0 in base.items():
                v = rep[k]
                if isinstance(v0, torch.Tensor) != isinstance(v, torch.Tensor):
                    raise ValueError(f"replica {i}['{k}'] tensor-ness differs from replica 0")
                if isinstance(v0, torch.Tensor) and (
                    v.shape != v0.shape or v.dtype != v0.dtype
                    or v.stride() != v0.stride() or v.device != v0.device
                ):
                    raise ValueError(
                        f"replica {i}['{k}'] (shape={tuple(v.shape)}, dtype={v.dtype}, "
                        f"stride={v.stride()}, device={v.device}) does not match replica 0"
                    )
