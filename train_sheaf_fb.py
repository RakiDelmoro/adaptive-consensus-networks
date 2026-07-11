#!/usr/bin/env python
"""Convenience launcher for Sheaf-FB training (Equilibrium Propagation, MNIST)."""
import argparse

import yaml

from sheaf_fb.config import get_preset
from sheaf_fb.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sheaf-FB on MNIST (Equilibrium Propagation)")
    parser.add_argument("preset", nargs="?", default="mnist",
                        help="config preset name (default: mnist)")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, default=[],
                        help="key=value overrides (values are YAML-parsed). "
                             "Use dotted keys for nested fields, e.g. "
                             "'train.epochs=100' or 'model.K=20'")
    args = parser.parse_args()

    cfg = get_preset(args.preset)
    changes = {}
    for ov in args.overrides:
        if "=" in ov:
            k, v = ov.split("=", 1)
            try:
                v = yaml.safe_load(v)
            except Exception:
                pass
            changes[k] = v

    if changes:
        cfg = cfg.override(**changes)
    print(cfg.to_yaml())
    model, summary = train(cfg)
