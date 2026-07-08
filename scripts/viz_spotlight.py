#!/usr/bin/env python
"""Generate the Thousand-Brains Spotlight GIFs from a trained ACN checkpoint.

Produces two GIFs:
  1. spotlight.gif — 10 samples (one per digit) showing the full 6-panel flow
  2. robustness_spotlight.gif — one digit under 7 corruptions

Usage:
    python scripts/viz_spotlight.py
    python scripts/viz_spotlight.py --n-samples 5 --sample-idx 3
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torchvision import datasets, transforms

from acn.config import get_preset
from acn.model import AdaptiveConsensusNetwork
from acn.visualize import make_spotlight_gif, make_robustness_spotlight_gif


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="mnist")
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--sample-idx", type=int, default=0,
                   help="which test sample to use for the robustness GIF")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="results/runs/acn_result")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = get_preset(args.config)

    # load model
    print("loading model...")
    model = AdaptiveConsensusNetwork(
        cfg.data.image_size, cfg.data.patch_size, cfg.data.stride,
        cfg.model, data_cfg=cfg.data
    ).to(dev)
    ckpt = Path(args.out_dir) / "model_best.pt"
    state = torch.load(ckpt, map_location=dev)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    # load test set
    print("loading MNIST test set...")
    ds = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    Xte = torch.stack([ds[i][0] for i in range(len(ds))])
    yte = torch.tensor([ds[i][1] for i in range(len(ds))])

    out = Path(args.out_dir)

    # 1. Spotlight GIF (10 samples, one per digit)
    print(f"generating spotlight.gif ({args.n_samples} samples)...")
    make_spotlight_gif(
        model, Xte, yte,
        path=out / "spotlight.gif",
        n_samples=args.n_samples,
        duration_ms=200,
        linger_frames=15,
    )

    # 2. Robustness Spotlight GIF (one digit, 7 corruptions)
    print(f"generating robustness_spotlight.gif (sample {args.sample_idx})...")
    make_robustness_spotlight_gif(
        model, Xte, yte,
        path=out / "robustness_spotlight.gif",
        sample_idx=args.sample_idx,
        duration_ms=250,
        linger_frames=20,
    )

    print(f"\nDone! GIFs saved to {out}/")


if __name__ == "__main__":
    main()
