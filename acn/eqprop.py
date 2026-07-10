"""Centered equilibrium-propagation contrast (the no-BPTT gradient).

  1. free settle (β=0)             -> s_free
  2. nudged+ settle (β>0, +y)      -> s_plus
  3. nudged- settle (β<0, -y)      -> s_minus   [centered — kills the O(β) bias]
  4. contrast = (E_plus - E_minus) / (2β)
  5. ONE autograd backward through the two scalar E evaluations -> ∂E/∂θ

Settles run under no_grad (no tape across rounds). Only the two final scalar E
evaluations build a (tiny) graph. Memory O(1) in rounds.

For PCN the free equilibrium is the feedforward pass, so the contrast is between
two well-defined minima — the gradient is well-conditioned (no saddle, no
ill-conditioned adjoint).
"""
from __future__ import annotations

import torch

from acn.flatstate import State, LiveCtx, Scalars
from acn.settle import settle
from acn.energy import energy_E


def eqprop_loss(state0: State, ctx: LiveCtx, sc: Scalars, target: torch.Tensor,
                *, beta: float, **settle_kw) -> torch.Tensor:
    """Returns the scalar contrast whose backward is the EP gradient."""
    s_free, _ = settle(state0, ctx, sc, beta=0.0, target=None, **settle_kw)
    s_plus, _ = settle(s_free, ctx, sc, beta=beta, target=target, **settle_kw)
    s_minus, _ = settle(s_free, ctx, sc, beta=-beta, target=target, **settle_kw)
    E_plus = energy_E(s_plus.clone_detach(), ctx, sc, beta=beta, target=target)
    E_minus = energy_E(s_minus.clone_detach(), ctx, sc, beta=-beta, target=target)
    return (E_plus - E_minus) / (2.0 * beta)
