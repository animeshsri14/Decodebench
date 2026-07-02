"""Llama decode-chain fusion demos: F1 (RMSNorm+GEMV), F2 (gate+up+SiLU), F4 (scores+softmax+V).

All tensors FP16, device "cuda", torch.manual_seed(42) before allocation.
Batch semantics: activations always 2-D [B, d] (default B=1), except F4
tensors which carry a leading batch dim and extra head/seq axes.
"""

from __future__ import annotations

from decodebench.sequence import Sequence
from decodebench.timing import n_weight_replicas

DEMO_NAMES = ("f1", "f2", "f4")
_FFN = {2048: 8192, 4096: 11008}
_L = 1024
_HEAD_DIM = 128


# ---------------------------------------------------------------------------
# F1: RMSNorm  +  GEMV  (QKV projection)
# ---------------------------------------------------------------------------


def _build_f1(dim: int, batch: int) -> tuple[Sequence, dict, list[dict]]:
    import torch

    d = dim
    B = batch

    x = torch.randn(B, d, dtype=torch.float16, device="cuda")
    g = torch.randn(d, dtype=torch.float16, device="cuda")
    W = torch.randn(d, d, dtype=torch.float16, device="cuda")

    seq = Sequence("f1")

    @seq.stage
    def rmsnorm(x, g):
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + 1e-6)).half() * g

    @seq.stage
    def gemv(xh, W):
        return torch.nn.functional.linear(xh, W)

    inputs = {"x": x, "g": g, "W": W}

    weight_bytes = W.nbytes
    nrep = n_weight_replicas(weight_bytes)
    replicas = []
    for _ in range(nrep):
        replicas.append({"x": x, "g": g, "W": W.clone()})

    return seq, inputs, replicas


# ---------------------------------------------------------------------------
# F2: gate  +  up & SiLU  (FFN gating)
# ---------------------------------------------------------------------------


def _build_f2(dim: int, batch: int) -> tuple[Sequence, dict, list[dict]]:
    import torch

    d = dim
    B = batch
    ff = _FFN[d]

    xh = torch.randn(B, d, dtype=torch.float16, device="cuda")
    Wg = torch.randn(ff, d, dtype=torch.float16, device="cuda")
    Wu = torch.randn(ff, d, dtype=torch.float16, device="cuda")

    seq = Sequence("f2")

    # Three declared stages so BOTH intermediates (g and u) are visible to the
    # byte model — the unfused pipeline really materializes both. Parameters
    # named after earlier stages ("gate", "up") bind to those stages' outputs.
    @seq.stage
    def gate(xh, Wg):
        return torch.nn.functional.linear(xh, Wg)

    @seq.stage
    def up(xh, Wu):
        return torch.nn.functional.linear(xh, Wu)

    @seq.stage
    def swiglu(up, gate):
        return torch.nn.functional.silu(gate) * up

    inputs = {"xh": xh, "Wg": Wg, "Wu": Wu}

    weight_bytes = Wg.nbytes + Wu.nbytes
    nrep = n_weight_replicas(weight_bytes)
    replicas = []
    for _ in range(nrep):
        replicas.append({"xh": xh, "Wg": Wg.clone(), "Wu": Wu.clone()})

    return seq, inputs, replicas


# ---------------------------------------------------------------------------
# F4: scores  +  softmax  +  weighted V  (attention sink)
# ---------------------------------------------------------------------------


def _build_f4(dim: int, batch: int) -> tuple[Sequence, dict, list[dict]]:
    import torch

    d = dim
    B = batch
    H = d // _HEAD_DIM  # number of heads (32 at dim=4096)
    D = _HEAD_DIM
    L = _L

    q = torch.randn(B, H, D, dtype=torch.float16, device="cuda")
    K = torch.randn(B, H, L, D, dtype=torch.float16, device="cuda")
    V = torch.randn(B, H, L, D, dtype=torch.float16, device="cuda")

    seq = Sequence("f4")

    @seq.stage
    def scores(q, K):
        # q:[B,H,D] K:[B,H,L,D] -> [B,H,L]. FP16 einsum, upcast ONLY the
        # small result to FP32 (RI-2: never K.float()/V.float() — that
        # materialises 16.8 MB hidden copies StageTrace cannot see).
        return torch.einsum("bhd,bhld->bhl", q, K).float()

    @seq.stage
    def softmax(s):
        return torch.softmax(s, dim=-1)  # FP32 in, FP32 out

    @seq.stage
    def weighted_v(p, V):
        # cast small p, not V (RI-2)
        return torch.einsum("bhl,bhld->bhd", p.half(), V)

    inputs = {"q": q, "K": K, "V": V}

    weight_bytes = K.nbytes + V.nbytes
    nrep = n_weight_replicas(weight_bytes)
    replicas = []
    for _ in range(nrep):
        replicas.append({"q": q, "K": K.clone(), "V": V.clone()})

    return seq, inputs, replicas


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


_BUILDERS = {"f1": _build_f1, "f2": _build_f2, "f4": _build_f4}


def build_demo(
    name: str, dim: int = 4096, batch: int = 1
) -> tuple[Sequence, dict, list[dict]]:
    """Build a named fusion demo. Returns (seq, inputs, input_replicas)."""
    if name not in _BUILDERS:
        raise ValueError(f"unknown demo '{name}'; choose from {list(_BUILDERS)}")
    return _BUILDERS[name](dim, batch)
