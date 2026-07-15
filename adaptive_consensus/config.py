"""Configuration for the Lean Sheaf-ADMM consensus network.

A single :class:`ModelConfig` (architecture) and :class:`TrainConfig`
(optimizer/data/viz). No curriculum, no phases — the model runs a fixed number
of ADMM rounds with full communication.

Solver options:
* ``objective_mode='lasso'`` + diagonal-prox x-update (soft-thresholding) —
  the default local solve.
* ``z_solver='cg_project'`` — unrolled conjugate-gradient z-update with hard
  ``ker(F)`` projection (``Fz -> 0``), the consensus step.
Legacy ('quadratic' / 'gd') modes are retained as the original lean variant.

Reference-frame additions:
* ``d_pos``: reference-frame / "where" code appended to each agent's input
  (grid-cell-like location signal). 0 disables.
* ``confidence_weighted``: *confidence-weighted voting* — each column's vote is
  scaled by a learned confidence instead of a uniform mean.
* ``edge_energy_weight`` / ``disagreement_weight``: turn the ADMM disagreement
  signal into a *learning* signal by adding a small regularizer on the final
  sheaf edge-energy / dual disagreement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Lean Sheaf-ADMM architecture hyperparameters."""

    num_classes: int = 10

    # --- latent + channel ---
    d: int = 16               # agent state dim (x, z, u)
    k: int = 8                # shared communication channel dim (k < d)
    q_rank: int = 4           # low-rank factor for Q_i = L L^T + eps I
    q_eps: float = 1e-3       # keeps Q_i strictly PD

    # --- ADMM ---
    # Round counts: K_train=20, K_eval=100, inner z-solver steps T=5. Fixed counts.
    K: int = 20               # number of ADMM rounds for TRAINING
    K_eval: int = 100         # number of ADMM rounds for EVAL / inference
    T: int = 5                # inner z-solver steps per z-update
    rho_init: float = 1.0     # ADMM penalty (learnable, kept positive)
    lr_z_init: float = 0.1    # sheaf-diffusion step size (used only by z_solver='gd')

    # --- local objective / x-solver ---
    # ``objective_mode``: 'quadratic' (exact Woodbury solve on Q=L L^T+eps I)
    # or 'lasso' (diagonal-prox soft-thresholding on a diagonal Q + L1).
    objective_mode: str = "lasso"   # 'lasso' | 'quadratic' (legacy)
    l1_weight: float = 0.00634     # L1 weight (active for 'lasso')
    l2_weight: float = 0.0         # extra L2 curvature

    # --- z-solver (consensus) ---
    # 'cg_project': unrolled conjugate gradient with hard ker(F) projection
    # (z = z_target - (L+eps I)^{-1} L z_target), Fz -> 0.
    # 'gd' (legacy): plain gradient-descent sheaf diffusion (soft, no projection).
    z_solver: str = "cg_project"   # 'cg_project' | 'gd' (legacy)
    tikhonov_eps: float = 1e-5     # project-mode regularizer on singular L

    # --- encoder / decoder MLPs (no CNN) ---
    enc_hidden: int = 64
    dec_hidden: int = 32
    dec_readout: str = "x"          # which ADMM variable to decode: 'x' | 'z' (legacy)

    # --- reference-frame additions ---
    d_pos: int = 8                   # reference-frame code dim appended to input (0 = off)
    confidence_weighted: bool = True # confidence-weighted voting (vs uniform mean)
    edge_energy_weight: float = 0.01 # regularize final sheaf edge-energy (disagreement)
    disagreement_weight: float = 0.0 # regularize final dual disagreement (0 = off)

    # --- loss ---
    round_aux_weight: float = 0.5   # weight on the per-round CE auxiliary

    # --- graph ---
    image_size: int = 28
    patch_size: int = 4
    stride: int = 4            # 28/4 -> 7x7 = 49 agents
    connectivity: int = 4      # 4-connected grid -> 84 edges


@dataclass
class TrainConfig:
    """Optimizer / data / viz."""

    seed: int = 42
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    val_frac: float = 0.1
    data_root: str = "./data"
    save_dir: str = "results/adaptive_consensus"
    # viz
    viz: bool = True
    viz_interval: int = 1
    viz_samples: int = 4
