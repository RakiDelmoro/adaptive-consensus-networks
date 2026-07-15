"""Adaptive Consensus Networks — Lean Sheaf-ADMM consensus network.

A lean Sheaf-ADMM model for MNIST classification.
49 agents on a 7x7 grid, each owning a 4x4 patch, negotiate a global digit
prediction through sheaf-structured communication over 84 edges. Trained
end-to-end via backprop through the unrolled ADMM rounds.

Key differences from the sibling ``acn`` package:

* **Local objective**: convex quadratic ``f(x) = 1/2 x^T Q x + p^T x`` with
  ``Q = L L^T + eps I`` (PSD), not Lasso (L1).
* **x-update**: closed-form Woodbury solve — exact, not proximal.
* **z-update**: gradient-descent sheaf diffusion, not unrolled CG with hard
  ``ker(F)`` projection.
* **Restriction maps**: per-edge (84 pairs), not 8 directional + LoRA.
* **Smaller**: ``d=16, k=8`` vs ``d_v=32, d_e=24``; ~33K params.

The result is a smaller, faster-converging sheaf-ADMM variant.
"""

from .config import ModelConfig, TrainConfig
from .graph import build_grid, patchify, sinusoidal_pos_code
from .model import AdaptiveConsensusModel
from .train import train

__all__ = [
    "AdaptiveConsensusModel",
    "ModelConfig",
    "TrainConfig",
    "build_grid",
    "patchify",
    "sinusoidal_pos_code",
    "train",
]

__version__ = "0.1.0"
