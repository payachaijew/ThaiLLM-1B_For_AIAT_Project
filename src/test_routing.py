#!/usr/bin/env python3
"""Tests for the corrected router.

The archived harness passed 25/25 of its own tests while shipping a defect that made the
router unable to learn. Those tests checked "does the function do what it says"; none checked
"are the three conditions actually comparable". The first test below is the one that would
have caught it, and it is deliberately first.
"""
from __future__ import annotations
import sys, unittest
from types import SimpleNamespace
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from routing import DeltaRouter, DeltaAttnResAdapter


class TinyLayer(nn.Module):
    def __init__(s, h): super().__init__(); s.lin = nn.Linear(h, h, bias=False)
    def forward(s, x, *a, **k): return (x + torch.tanh(s.lin(x)),)


class TinyBack(nn.Module):
    def __init__(s, h, n): super().__init__(); s.layers = nn.ModuleList([TinyLayer(h) for _ in range(n)])


class TinyLM(nn.Module):
    def __init__(s, v=17, h=16, n=8):
        super().__init__(); s.config = SimpleNamespace(hidden_size=h)
        s.embed = nn.Embedding(v, h); s.model = TinyBack(h, n)
        s.head = nn.Linear(h, v, bias=False); s.is_gradient_checkpointing = False
    def forward(s, ids):
        x = s.embed(ids)
        for l in s.model.layers: x = l(x)[0]
        return SimpleNamespace(logits=s.head(x))


def _loss(model, ids):
    out = model(ids).logits
    return nn.functional.cross_entropy(out[:, :-1].reshape(-1, 17), ids[:, 1:].reshape(-1))


class TheRegressionTest(unittest.TestCase):
    """THE test the archived suite was missing."""

    def test_router_receives_nonzero_gradient_at_step_one(self):
        for heads in (1, 4):
            with self.subTest(heads=heads):
                torch.manual_seed(0)
                m = DeltaAttnResAdapter(TinyLM(), block_size_layers=2, num_heads=heads)
                _loss(m, torch.tensor([[1, 2, 3, 4, 5, 6]])).backward()
                grads = [float(r.query.grad.abs().sum()) for r in m.routers
                         if r is not None and r.query.grad is not None]
                self.assertTrue(grads, "no router received a gradient at all")
                # routers in the first block see no sources yet, so only later ones must move
                self.assertGreater(max(grads), 0.0,
                                   "router query gradient is zero at initialisation — "
                                   "this is the ROUTE_SCALE_GATE defect")


class DDPContractTests(unittest.TestCase):
    """Every parameter must receive a gradient every step, or DDP raises.

    Measured on 2x A100: leaving routers on the first block made every DDP step fail with
    "Expected to have finished reduction in the prior iteration". A single-GPU run never
    notices, so the invariant is pinned here rather than left to the next rental.
    """

    def test_first_block_has_no_routers(self):
        m = DeltaAttnResAdapter(TinyLM(n=8), block_size_layers=4)
        self.assertEqual([r is None for r in m.routers][:4], [True] * 4)
        self.assertTrue(all(r is not None for r in m.routers[4:]))

    def test_every_parameter_receives_a_gradient(self):
        torch.manual_seed(0)
        m = DeltaAttnResAdapter(TinyLM(n=8), block_size_layers=4)
        _loss(m, torch.tensor([[1, 2, 3, 4, 5, 6]])).backward()
        missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
        self.assertEqual(missing, [], f"these would stall the DDP reducer: {missing}")


class RouterTests(unittest.TestCase):
    def test_weights_sum_to_one_per_head(self):
        r = DeltaRouter(16, num_heads=4)
        _, w = r(torch.randn(2, 3, 16), [torch.randn(2, 3, 16) for _ in range(3)])
        torch.testing.assert_close(w.sum(0), torch.ones_like(w[0]))

    def test_uniform_softmax_at_init_is_bounded_not_identity(self):
        """The paper's zero-init query gives a bounded perturbation, not an exact identity.
        Exactness was what the removed gate bought, at the cost of learning."""
        torch.manual_seed(1)
        r = DeltaRouter(16)
        res = torch.randn(2, 3, 16)
        out, _ = r(res, [torch.randn(2, 3, 16) for _ in range(3)])
        self.assertFalse(torch.equal(out, res))
        self.assertLess(float((out - res).norm() / res.norm()), 2.0)

    def test_no_sources_is_exact_identity(self):
        r = DeltaRouter(16)
        res = torch.randn(2, 3, 16)
        out, _ = r(res, [])
        self.assertTrue(torch.equal(out, res))

    def test_mhar_heads_can_disagree(self):
        """D2's whole premise: different subspaces read different depths."""
        torch.manual_seed(3)
        r = DeltaRouter(16, num_heads=4)
        with torch.no_grad():
            r.query.copy_(torch.randn(4, 4) * 5)
        _, w = r(torch.randn(1, 2, 16), [torch.randn(1, 2, 16) for _ in range(4)])
        spread = w.std(dim=-1).max()
        self.assertGreater(float(spread), 1e-4, "all heads produced the same distribution")


class AdapterTests(unittest.TestCase):
    def test_diagnostics_do_not_retain_the_graph(self):
        """E2: the archive kept live tensors in last_routing, pinning activation memory
        for the routed arms only."""
        m = DeltaAttnResAdapter(TinyLM(), block_size_layers=2)
        m.collect_routing = True
        m(torch.tensor([[1, 2, 3, 4]]))
        for rec in m.last_routing:
            self.assertFalse(rec["weights"].requires_grad)
            self.assertIsNone(rec["weights"].grad_fn)

    def test_diagnostics_off_by_default(self):
        m = DeltaAttnResAdapter(TinyLM(), block_size_layers=2)
        m(torch.tensor([[1, 2, 3, 4]]))
        self.assertEqual(m.last_routing, [])

    def test_gradient_checkpointing_is_allowed(self):
        """E4: S0 could use checkpointing and D1/D2 could not, so the GPU-hour axis was
        measuring the wrapper rather than the architecture."""
        base = TinyLM(); base.is_gradient_checkpointing = True
        m = DeltaAttnResAdapter(base, block_size_layers=2)
        _loss(m, torch.tensor([[1, 2, 3, 4]])).backward()

    def test_first_block_has_no_sources_and_says_so(self):
        """E3: those routers are dead weight; surfaced rather than hidden."""
        m = DeltaAttnResAdapter(TinyLM(n=8), block_size_layers=4)
        self.assertEqual(m.layers_without_sources, 4)
        self.assertEqual(m.blocks, 2)

    def test_clamp_intervention_changes_output_and_clears(self):
        torch.manual_seed(5)
        m = DeltaAttnResAdapter(TinyLM(), block_size_layers=2)
        with torch.no_grad():
            for r in m.routers:
                if r is not None:
                    r.query.copy_(torch.randn_like(r.query))
        ids = torch.tensor([[1, 2, 3, 4]])
        m.set_next_intervention(null_clamp_mask=torch.ones_like(ids, dtype=torch.bool))
        a = m(ids).logits.detach()
        b = m(ids).logits.detach()
        self.assertFalse(torch.equal(a, b))

    def test_reentrant_forward_is_rejected(self):
        m = DeltaAttnResAdapter(TinyLM(), block_size_layers=2)
        m._state = object()
        with self.assertRaises(RuntimeError):
            m(torch.tensor([[1, 2]]))
        m._state = None


if __name__ == "__main__":
    unittest.main(verbosity=2)
