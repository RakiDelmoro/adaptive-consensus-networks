"""ACN — Robustness evaluation utilities.

Builds a fixed set of corrupted versions of a test batch (clean + 7
corruptions: rot15, rot30, rot45, shift20, shift30, occ3×7, noise30) and
measures the model's accuracy on each version. Used during training at every
snapshot interval so we can track robustness as the model learns.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F


# ── corruption transforms ──────────────────────────────────────────────

def rotate_batch(X, degrees):
    n = len(X)
    ang = torch.empty(n, device=X.device).uniform_(-degrees, degrees)
    theta = torch.zeros(n, 2, 3, device=X.device)
    rad = ang * (np.pi / 180.0)
    theta[:, 0, 0] = torch.cos(rad); theta[:, 0, 1] = -torch.sin(rad)
    theta[:, 1, 0] = torch.sin(rad);  theta[:, 1, 1] = torch.cos(rad)
    grid = F.affine_grid(theta, X.shape, align_corners=False)
    return F.grid_sample(X, grid, align_corners=False, padding_mode="zeros")


def occlude_batch(X, n_boxes=3, box_size=7):
    out = X.clone()
    H, W = X.shape[-2], X.shape[-1]
    for _ in range(n_boxes):
        r = torch.randint(0, H - box_size + 1, (len(X),), device=X.device)
        c = torch.randint(0, W - box_size + 1, (len(X),), device=X.device)
        for b in range(len(X)):
            out[b, :, r[b]:r[b]+box_size, c[b]:c[b]+box_size] = 0.0
    return out


def noise_batch(X, sigma=0.3):
    return (X + sigma * torch.randn_like(X)).clamp(0, 1)


def shift_batch(X, frac=0.2):
    n = len(X)
    theta = torch.zeros(n, 2, 3, device=X.device)
    theta[:, 0, 0] = 1.0; theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = torch.empty(n, device=X.device).uniform_(-frac, frac)
    theta[:, 1, 2] = torch.empty(n, device=X.device).uniform_(-frac, frac)
    grid = F.affine_grid(theta, X.shape, align_corners=False)
    return F.grid_sample(X, grid, align_corners=False, padding_mode="zeros")

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
