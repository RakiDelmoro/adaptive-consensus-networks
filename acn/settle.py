"""PCN-ACN settle: gradient descent on the prediction-error energy.

The settle finds the minimum of E (every prediction matches). Free phase (β=0):
the minimum is the feedforward pass with lateral agreement. Nudged phase (β≠0):
re-settle with the output pulled toward the label.

One round:  h ← h − α · ∇_h E   for each free variable (h1, ℓ).
Runs under no_grad in both phases (no autograd tape across rounds); only the
final scalar E in the contrast builds a graph.

The free equilibrium is fast to reach because at β=0 the minimum is the
feedforward pass (errors → 0), so a moderate number of rounds converges well.
"""
from __future__ import annotations

import torch

from acn.flatstate import State, LiveCtx, Scalars
from acn.energy import energy_E, state_grad


@torch.no_grad()
def settle(state: State, ctx: LiveCtx, sc: Scalars, *,
           k_max: int = 20, alpha: float = 0.5,
           beta: float = 0.0, target: torch.Tensor | None = None) -> tuple[State, float]:
    """Gradient descent on E to the minimum. Returns (settled_state, E_value)."""
    s = state.clone_detach()
    for k in range(k_max):
        grads, _ = state_grad(s, ctx, sc, beta=beta, target=target)
        s.h1 = s.h1 - alpha * grads["h1"] * ctx.active.unsqueeze(-1)
        s.llogits = s.llogits - alpha * grads["llogits"]
    E_final = float(energy_E(s, ctx, sc, beta=beta, target=target).detach())
    return s, E_final
