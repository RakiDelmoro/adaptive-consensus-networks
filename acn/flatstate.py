"""PCN-ACN state containers (per-column readout, Sheaf-ADMM-style).

State  : the dynamical variables — per-column latents h_1^i and per-column
          output logits ℓ_i (one prediction per column, the label broadcast to
          all columns). These settle by gradient descent on the energy.
          In the contrast they are DETACHED (EP takes ∂F/∂θ at the settled state).
LiveCtx: the θ-dependent quantities computed once in _prepare (patch predictions,
          lateral predictor, the shared per-column decoder D, edge structure).
          Stay on the autograd graph so the single contrast backward reaches θ.
Scalars: the strengths. κ (lateral consensus) is the key knob.

Readout is per-column: D(h1_i) -> logits_i for each column, then the global
prediction is the average of the per-column (softmax) logits — the same shape as
Sheaf-ADMM's MNIST path. The decoder only ever sees a single column, so the
per-column color grid is the trained, in-distribution readout.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class State:
    """The free variables of the PCN energy (settle by gradient descent)."""
    h1: torch.Tensor              # (B, N, d)    per-column latents (layer 1)
    llogits: torch.Tensor         # (B, N, C)    per-column output logits (the verdict, one per column)

    def keys(self) -> list[str]:
        return ["h1", "llogits"]

    def clone_detach(self) -> "State":
        return State(**{k: getattr(self, k).detach().clone() for k in self.keys()})


@dataclass
class LiveCtx:
    """θ-dependent quantities, computed once, live on the autograd graph."""
    h1_pred: torch.Tensor                # (B, N, d) per-column prediction f_1(patch_i)
    active: torch.Tensor                 # (B, N) activity mask
    edges: torch.Tensor                  # (2, E)
    lateral_predict: object              # h1 (B,N,d) + edges -> (pred_i, pred_j)  [live g_θ]
    column_decode: object                # h1 (B,N,d) -> per-column logits (B,N,C)  [shared head D]


@dataclass
class Scalars:
    """Energy strengths. κ = lateral consensus weight (the key knob)."""
    kappa: float = 0.1          # lateral consensus (prediction-error coupling)
    lam_col: float = 1.0        # per-column prediction error weight
    lam_output: float = 1.0     # per-column output prediction error weight
