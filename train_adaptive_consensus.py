#!/usr/bin/env python
"""Convenience launcher for Lean Sheaf-ADMM training on MNIST.

Usage:
    python train_adaptive_consensus.py
    python train_adaptive_consensus.py --epochs 20 --d 32 --k 8 --K 15
"""
from __future__ import annotations

import argparse

from adaptive_consensus.config import ModelConfig, TrainConfig
from adaptive_consensus.train import train


def main():
    p = argparse.ArgumentParser(description="Train Lean Sheaf-ADMM on MNIST")
    # model
    p.add_argument("--d", type=int, default=16, help="agent latent dim")
    p.add_argument("--k", type=int, default=8, help="communication channel dim")
    p.add_argument("--q-rank", type=int, default=4, help="low-rank factor for Q_i")
    p.add_argument("--K", type=int, default=20, help="ADMM rounds for training (K_train=20)")
    p.add_argument("--K-eval", type=int, default=100, help="ADMM rounds for eval/inference (K_eval=100)")
    p.add_argument("--T", type=int, default=5, help="inner sheaf-diffusion steps (cg_iters=5)")
    p.add_argument("--rho-init", type=float, default=1.0)
    p.add_argument("--lr-z-init", type=float, default=0.1)
    # solver options
    p.add_argument("--objective-mode", type=str, default="lasso",
                   choices=["lasso", "quadratic"],
                   help="local objective: 'lasso' (diagonal-prox) | 'quadratic' (Woodbury)")
    p.add_argument("--l1-weight", type=float, default=0.00634, help="L1 weight (lasso mode)")
    p.add_argument("--z-solver", type=str, default="cg_project",
                   choices=["cg_project", "gd"],
                   help="z-update: 'cg_project' (hard ker(F)) | 'gd' (soft diffusion)")
    p.add_argument("--tikhonov-eps", type=float, default=1e-5, help="project-mode Tikhonov eps")
    p.add_argument("--dec-readout", type=str, default="x", choices=["x", "z"],
                   help="decode 'x' (local proposal) | 'z' (consensus, legacy)")
    p.add_argument("--enc-hidden", type=int, default=64)
    p.add_argument("--dec-hidden", type=int, default=32)
    # reference-frame additions (Tier 1)
    p.add_argument("--d-pos", type=int, default=8, help="reference-frame code dim (0 = off)")
    p.add_argument("--no-confidence-weighted", action="store_true",
                   help="disable confidence-weighted voting (use uniform mean)")
    p.add_argument("--edge-energy-weight", type=float, default=0.01,
                   help="weight on final sheaf edge-energy regularizer")
    p.add_argument("--disagreement-weight", type=float, default=0.0,
                   help="weight on final dual-disagreement regularizer")
    # train
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--save-dir", type=str, default="results/adaptive_consensus")
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--viz-interval", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    model_cfg = ModelConfig(
        d=args.d, k=args.k, q_rank=args.q_rank, K=args.K, K_eval=args.K_eval, T=args.T,
        rho_init=args.rho_init, lr_z_init=args.lr_z_init,
        enc_hidden=args.enc_hidden, dec_hidden=args.dec_hidden,
        objective_mode=args.objective_mode, l1_weight=args.l1_weight,
        z_solver=args.z_solver, tikhonov_eps=args.tikhonov_eps,
        dec_readout=args.dec_readout,
        d_pos=args.d_pos, confidence_weighted=not args.no_confidence_weighted,
        edge_energy_weight=args.edge_energy_weight,
        disagreement_weight=args.disagreement_weight,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, data_root=args.data_root,
        save_dir=args.save_dir, viz=not args.no_viz, viz_interval=args.viz_interval,
        seed=args.seed,
    )
    train(model_cfg, train_cfg)


if __name__ == "__main__":
    main()
