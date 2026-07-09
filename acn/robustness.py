"""ACN — Robustness evaluation utilities.

Builds a fixed set of corrupted versions of a test batch (clean + 7
corruptions: rot15, rot30, rot45, shift20, shift30, occ3×7, noise30) and
measures the model's accuracy on each version. Used during training at every
snapshot interval so we can track robustness as the model learns.

The corruption transforms live in :mod:`acn.visualize` (shared with the
robustness spotlight GIF); this module just wraps them with reproducible
seeding and an evaluation loop.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

from acn.visualize import rotate_batch, shift_batch, occlude_batch, noise_batch

# canonical version order (clean first) — shared with the spotlight GIF
ROBUSTNESS_VERSIONS = [
    "clean", "rot15", "rot30", "rot45",
    "shift20", "shift30", "occ3x7", "noise30",
]


def build_robustness_versions(X: torch.Tensor, seed: int = 42) -> OrderedDict:
    """Return an ordered dict {version_name: corrupted_X} for one tensor.

    X: (N, 1, H, W). All corruptions are generated with a fixed seed so the
    same input set always produces the same corrupted sets (reproducible
    comparisons across epochs).
    """
    torch.manual_seed(seed)
    return OrderedDict([
        ("clean",   X.clone()),
        ("rot15",   rotate_batch(X, 15)),
        ("rot30",   rotate_batch(X, 30)),
        ("rot45",   rotate_batch(X, 45)),
        ("shift20", shift_batch(X, 0.2)),
        ("shift30", shift_batch(X, 0.3)),
        ("occ3x7",  occlude_batch(X, 3, 7)),
        ("noise30", noise_batch(X, 0.3)),
    ])


@torch.no_grad()
def evaluate_robustness(
    model, X: torch.Tensor, y: torch.Tensor,
    seed: int = 42, batch_size: int = 128,
) -> OrderedDict:
    """Measure model accuracy on clean + 7 corrupted versions of (X, y).

    Returns an ordered dict {version_name: accuracy_float}.
    """
    dev = next(model.parameters()).device
    model.eval()
    versions = build_robustness_versions(X, seed=seed)
    results: "OrderedDict[str, float]" = OrderedDict()
    for name, Xv in versions.items():
        correct = 0
        for i in range(0, len(Xv), batch_size):
            xb = Xv[i:i + batch_size].to(dev)
            yb = y[i:i + batch_size].to(dev)
            pred, _ = model(xb)
            correct += (pred.argmax(dim=1) == yb).sum().item()
        results[name] = correct / len(Xv)
    return results
