#!/usr/bin/env python
"""Convenience launcher for ACN training."""
import sys, argparse
import yaml
from acn.config import get_preset
from acn.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ACN")
    parser.add_argument("preset", nargs="?", default="mnist")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, default=[])
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
