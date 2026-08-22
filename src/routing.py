#!/usr/bin/env python3
"""Delta Block Attention Residual routing — corrected implementation.

Replaces the archived harness at ../../thai-llm-five-to-two/depth_routing/routing.py, which
is READ-ONLY and must not be edited. Five defects found by the 2026-08-18 audit are fixed
here; each fix is commented with what it corrects and why it mattered.

Reference: Delta Attention Residuals, arXiv:2605.18855.
The official implementation is MIT-licensed at
https://github.com/wdlctc/delta-attention-residuals-code and ships modeling_qwen3_attnres.py.
Prefer that for the final runs; this module exists so the conditions are runnable and
testable without waiting on that integration, and so the audit fixes are explicit.

scientific_evidence_allowed = false for anything produced by tiny fixtures.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.float().pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v.to(x.dtype) + self.eps) * self.weight


class DeltaRouter(nn.Module):
    """Additive softmax routing over previous block deltas.

    FIX 1 — ROUTE_SCALE_GATE (was blocking).
    The archived version multiplied the routed mixture by a scalar `route_scale`
    initialised to zero, to make checkpoint conversion bit-exact. Because the output was
        out = residual + s * mixture(w)
    the gradient with respect to every routing parameter is proportional to s:
        dL/dw = (dL/dout) * s * (dmixture/dw)
    At s = 0 that is identically zero, so the router could not begin learning. Measured:
    query gradient still ~0.095 percent of the paper-faithful magnitude after 50 steps.
    Any "routing does not help Thai" result from that build would have been indistinguishable
    from "the gate prevented the router from learning".

    The paper does not use such a gate. It relies on a zero-initialised query alone, which
    gives a uniform softmax and therefore a BOUNDED PERTURBATION at init rather than an exact
    identity — and, critically, non-zero query gradients from step 1.

    null_logit_init is the correct control, and it is NOT the same thing as route_scale.
    The null source contributes nothing, so raising its logit moves softmax mass onto "do
    nothing" and shrinks the mixture towards zero. The query gradient decays as e^-c rather
    than being multiplied by zero, so the router keeps learning. Measured on Qwen3-0.6B,
    one forward, 128 tokens (S0 loss 13.8879):

        null_logit_init   loss delta vs S0   max query grad
              route_scale=0        0.0000         0.000000   <- cannot learn
                    0.0           +6.7440         0.589615   <- paper default, disruptive
                    2.0           +0.7739         2.162301   <- default here
                    4.0           -0.0234         0.064080
                    8.0           +0.0051         0.002060   <- approaching the old failure

    2.0 is the default: the +0.77 starting penalty is recovered quickly, and it gives the
    LARGEST query gradient of any setting tested. Values of 4 and above buy a near-identity
    conversion at the cost of a gradient 30-1000x smaller, which risks recreating the
    route_scale failure in milder form — a router that technically can learn but does not,
    within the token budget. Treat this as a preregistered choice and check it in preflight.
    """

    def __init__(self, hidden_size: int, num_heads: int = 1, *, null_logit_init: float = 2.0):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"hidden_size {hidden_size} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm = RMSNorm(hidden_size)
        # FIX 2 — MHAR (condition D2) did not exist in the archive at all.
        # arXiv:2607.27230 reshapes the single routing query into H per-subspace heads, each
        # with its own softmax over the depth history. num_heads=1 reproduces plain Delta.
        self.query = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.null_logit = nn.Parameter(torch.full((num_heads,), float(null_logit_init)))

    def forward(self, residual: torch.Tensor, sources: list[torch.Tensor], *,
                null_clamp_mask: torch.Tensor | None = None,
                ablate_source_indices: set[int] | None = None,
                override_weights: torch.Tensor | None = None):
        if residual.ndim != 3:
            raise ValueError("residual must be [batch, seq, hidden]")
        B, T, H = residual.shape
        if not sources:
            w = torch.ones((1, self.num_heads, B, T), dtype=residual.dtype, device=residual.device)
            return residual, w

        V = torch.stack(sources, 0)                                  # [S,B,T,H]
        S = V.shape[0]
        K = self.norm(V).view(S, B, T, self.num_heads, self.head_dim)
        logits = torch.einsum("sbthd,hd->sbth", K, self.query)       # [S,B,T,heads]
        logits = logits / (self.head_dim ** 0.5)
        null = self.null_logit.view(1, 1, 1, self.num_heads).expand(1, B, T, self.num_heads)
        w = torch.softmax(torch.cat([null, logits], 0).float(), 0).to(residual.dtype)

        if ablate_source_indices:
            keep = torch.ones(w.shape[0], dtype=torch.bool, device=w.device)
            for i in ablate_source_indices:
                if 0 < i < w.shape[0]:
                    keep[i] = False
            w = w * keep.view(-1, 1, 1, 1)
            w = w / w.sum(0, keepdim=True).clamp_min(1e-8)
        if override_weights is not None:
            if override_weights.shape != w.shape:
                raise ValueError("override_weights shape mismatch")
            w = override_weights.to(w.device, w.dtype)
            w = w / w.sum(0, keepdim=True).clamp_min(1e-8)
        if null_clamp_mask is not None:
            if null_clamp_mask.shape != (B, T):
                raise ValueError("null_clamp_mask must be [batch, seq]")
            m = null_clamp_mask.to(w.device, torch.bool).view(1, B, T, 1)
            only = torch.zeros_like(w); only[0] = 1
            w = torch.where(m, only, w)

        Vh = V.view(S, B, T, self.num_heads, self.head_dim)
        mix = torch.einsum("sbth,sbthd->bthd", w[1:], Vh).reshape(B, T, H)
        return residual + mix, w


@dataclass
class RoutingState:
    block_size_layers: int
    sources: list[torch.Tensor] = field(default_factory=list)
    block_start: torch.Tensor | None = None
    records: list[dict] = field(default_factory=list)
    null_clamp_mask: torch.Tensor | None = None
    ablate_source_indices: set[int] | None = None
    weight_overrides: dict[int, torch.Tensor] = field(default_factory=dict)


def _hidden(o):
    if torch.is_tensor(o): return o
    if isinstance(o, (tuple, list)) and o and torch.is_tensor(o[0]): return o[0]
    raise TypeError(f"unsupported layer output: {type(o).__name__}")


def _replace(o, h):
    if torch.is_tensor(o): return h
    if isinstance(o, tuple): return (h, *o[1:])
    if isinstance(o, list): return [h, *o[1:]]
    raise TypeError(f"unsupported layer output: {type(o).__name__}")


class RoutedLayer(nn.Module):
    def __init__(self, layer, router, index, owner):
        """router is None for layers in the first block.

        Those layers have no earlier block to route over, so a router there would be a
        parameter that never receives gradient. Under DDP that is fatal, not merely wasteful:
        the reducer waits for buckets that never fill and the step raises "Expected to have
        finished reduction in the prior iteration". The layer is still wrapped, because the
        block bookkeeping below is what produces the sources every later block consumes.
        """
        super().__init__()
        self.layer, self.router, self.layer_index = layer, router, index
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name):
        """Proxy unknown attributes to the wrapped decoder layer.

        Model code reads per-layer metadata straight off the layer object — Qwen3 reads
        decoder_layer.attention_type on every forward. Without this the wrapper raises
        AttributeError and no routed condition can run at all.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            layer = self.__dict__.get("_modules", {}).get("layer")
            if layer is not None and layer is not self:
                return getattr(layer, name)
            raise

    def forward(self, hidden_states, *a, **k):
        owner = object.__getattribute__(self, "_owner")
        st = owner._state
        if st is None:
            raise RuntimeError("routed layer called outside adapter forward")
        if st.block_start is None:
            st.block_start = hidden_states
        if self.router is None:
            routed, w = hidden_states, None
        else:
            routed, w = self.router(hidden_states, st.sources,
                                    null_clamp_mask=st.null_clamp_mask,
                                    ablate_source_indices=st.ablate_source_indices,
                                    override_weights=st.weight_overrides.get(self.layer_index))
        out = self.layer(routed, *a, **k)
        h = _hidden(out)
        # FIX 3 — E2: the archive stored the LIVE weight tensor, keeping the autograd graph
        # alive after the forward returned and pinning activation memory for every layer.
        # Only the baseline avoided that cost, which biased the GPU-hour axis against the
        # routed arms. Diagnostics are detached here.
        if owner.collect_routing and w is not None:
            st.records.append({"layer": self.layer_index, "weights": w.detach()})
        if (self.layer_index + 1) % st.block_size_layers == 0:
            st.sources.append(h - st.block_start)
            st.block_start = h
        return _replace(out, h)


def _find_layers(model):
    for path in (("model", "layers"), ("model", "model", "layers"), ("transformer", "h"), ("layers",)):
        cur = model
        try:
            for n in path: cur = getattr(cur, n)
        except AttributeError:
            continue
        if isinstance(cur, nn.ModuleList): return cur
    raise ValueError("could not locate decoder ModuleList")


def _hidden_size(model):
    cfg = getattr(model, "config", None)
    for n in ("hidden_size", "d_model", "n_embd"):
        v = getattr(cfg, n, None) if cfg is not None else None
        if isinstance(v, int) and v > 0: return v
    raise ValueError("config does not expose hidden size")


class DeltaAttnResAdapter(nn.Module):
    """Wrap a loaded causal LM with Delta Block routing (D1) or MHAR (D2)."""

    def __init__(self, base_model, *, block_size_layers: int = 4, num_heads: int = 1,
                 null_logit_init: float = 2.0):
        super().__init__()
        if block_size_layers <= 0:
            raise ValueError("block_size_layers must be positive")
        self.base_model = base_model
        self.block_size_layers = block_size_layers
        self._state: RoutingState | None = None
        self.last_routing: list[dict] = []
        # FIX 4 — E2 continued: diagnostics are opt-in. Training runs keep this False so no
        # per-layer tensor is retained at all.
        self.collect_routing = False
        self._next: dict[str, Any] = {}
        H = _hidden_size(base_model)
        layers = _find_layers(base_model)
        first_block = min(block_size_layers, len(layers))
        routers = [None if i < first_block
                   else DeltaRouter(H, num_heads, null_logit_init=null_logit_init)
                   for i in range(len(layers))]
        object.__setattr__(self, "routers", routers)
        for i, (lyr, r) in enumerate(zip(list(layers), routers)):
            layers[i] = RoutedLayer(lyr, r, i, self)
        self.config = getattr(base_model, "config", None)
        # FIX 5 — E3: the first block routes over zero sources. Those routers were previously
        # built anyway and left permanently at their initial values; they are now simply not
        # created. Beyond the wasted parameters, a parameter that never receives gradient
        # breaks DDP outright — measured on 2x A100, where every step raised
        # "Expected to have finished reduction in the prior iteration".
        self.blocks = len(layers) // block_size_layers
        self.layers_without_sources = first_block
        self.routed_layers = len(layers) - first_block

    # FIX 6 — E4: the archive raised if gradient checkpointing was enabled, so S0 could use it
    # and D1/D2 could not. Different memory regimes per condition make the GPU-hour comparison
    # measure the wrapper rather than the architecture. Routing state is per-forward and the
    # sources list is rebuilt each call, so checkpointing is allowed; the guard below only
    # rejects genuine re-entrancy.
    def forward(self, *a, **k):
        if self._state is not None:
            raise RuntimeError("nested/reentrant forward is not supported")
        self._state = RoutingState(self.block_size_layers,
                                   null_clamp_mask=self._next.get("null_clamp_mask"),
                                   ablate_source_indices=self._next.get("ablate_source_indices"),
                                   weight_overrides=self._next.get("weight_overrides", {}))
        try:
            out = self.base_model(*a, **k)
            self.last_routing = list(self._state.records) if self.collect_routing else []
            return out
        finally:
            self._state = None
            self._next = {}

    def set_next_intervention(self, *, null_clamp_mask=None, ablate_source_indices=None,
                              weight_overrides=None):
        if self._state is not None:
            raise RuntimeError("cannot change intervention during a forward")
        self._next = {"null_clamp_mask": null_clamp_mask,
                      "ablate_source_indices": ablate_source_indices,
                      "weight_overrides": weight_overrides or {}}

    def router_parameters(self):
        for r in self.routers:
            if r is not None:
                yield from r.parameters()
