"""PCN-ACN energy: one scalar prediction-error energy (a minimum, not a saddle).

    E(θ, x, h) =
        ½ λ_col  Σ_i  ‖ h1_i − f_1(patch_i) ‖²          (per-column prediction error)
      + ½ κ      Σ_{(i,j)}  ‖ h1_i − g_θ(h1_j) ‖²       (lateral consensus — each column
                       + ‖ h1_j − g_θ(h1_i) ‖²            matches its neighbors' prediction)
      + ½ λ_o    Σ_i  ‖ ℓ_i − D(h1_i) ‖²                (per-column output prediction error)

Every term is a squared prediction error → the energy is 0 at the minimum where
every prediction matches (the feedforward pass with lateral agreement). ONE
scalar, ONE minimum — EP-native. The per-column term gives every encoder weight
a LOCAL gradient (no 1/N dilution); the lateral term is the consensus regularizer.

Readout is per-column (Sheaf-ADMM-style): D decodes each column's h1_i
individually into logits_i, the label is broadcast to all (active) columns, and
the global prediction is the average of the per-column logits (see ACNv2.forward).
The nudge adds β · (1/n_act) Σ_i active_i · CE(ℓ_i, y) in the nudged phase — every
active column is pushed toward the (broadcast) label.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from acn.flatstate import State, LiveCtx, Scalars


def energy_E(state: State, ctx: LiveCtx, sc: Scalars,
             beta: float = 0.0, target: torch.Tensor | None = None) -> torch.Tensor:
    """Scalar E(state, θ). States may be live (settle) or detached (contrast);
    `ctx` is always live so θ receives gradient in the contrast."""
    active = ctx.active
    n_act = active.sum(1).clamp(min=1.0)                  # (B,) active columns

    # ── per-column prediction error: h1_i vs f_1(patch_i) ──
    col_err = (state.h1 - ctx.h1_pred).pow(2).sum(-1) * active    # (B, N)
    E = 0.5 * sc.lam_col * (col_err.sum(1) / n_act).mean()

    # ── lateral consensus: each column vs its neighbors' prediction of it ──
    ei, ej = ctx.edges[0], ctx.edges[1]
    pred_i, pred_j = ctx.lateral_predict(state.h1, ctx.edges)   # (B,E,d) each, live
    lat_err_i = (state.h1[:, ei] - pred_i).pow(2).sum(-1)      # (B, E)
    lat_err_j = (state.h1[:, ej] - pred_j).pow(2).sum(-1)      # (B, E)
    edge_mask = active[:, ei] * active[:, ej]                   # (B, E)
    n_edge = edge_mask.sum(1).clamp(min=1.0)
    lat_err = ((lat_err_i + lat_err_j) * edge_mask).sum(1) / n_edge
    E = E + 0.5 * sc.kappa * lat_err.mean()

    # ── per-column output prediction error: ℓ_i vs D(h1_i) ──
    llogits_pred = ctx.column_decode(state.h1)          # (B, N, C) live, per column
    out_err = (state.llogits - llogits_pred).pow(2).sum(-1) * active   # (B, N)
    E = E + 0.5 * sc.lam_output * (out_err.sum(1) / n_act).mean()

    # ── nudge: CE on each active column's logits toward the (broadcast) label ──
    if beta != 0.0 and target is not None:
        # broadcast the per-sample label to all columns: (B,) -> (B, N)
        y_broadcast = target.unsqueeze(1).expand(-1, state.llogits.shape[1])
        logp = F.log_softmax(state.llogits, dim=-1)                 # (B, N, C)
        ce_per_col = -logp.gather(2, y_broadcast.unsqueeze(-1)).squeeze(-1)  # (B, N)
        ce = (ce_per_col * active).sum(1) / n_act                   # (B,)
        E = E + beta * ce.mean()

    return E


def state_grad(state: State, ctx: LiveCtx, sc: Scalars,
               beta: float = 0.0, target: torch.Tensor | None = None):
    """∇_state E via autograd. Returns ({key: grad}, float(E))."""
    keys = state.keys()
    with torch.enable_grad():
        clones = {k: getattr(state, k).detach().clone().requires_grad_(True) for k in keys}
        s = State(**clones)
        E = energy_E(s, ctx, sc, beta=beta, target=target)
        grads = torch.autograd.grad(E, [clones[k] for k in keys],
                                    retain_graph=False, create_graph=False)
    return dict(zip(keys, [g.detach() for g in grads])), float(E.detach())
