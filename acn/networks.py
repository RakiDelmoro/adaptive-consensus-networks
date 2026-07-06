"""Encoder, decoder, diagonal restriction maps, and fusion.

Encoder produces the parameters ``(L_i, b_i)`` of each mini network's local convex
problem. The local quadratic is ``(1/2) z^T A_i z + b_i^T z`` with
``A_i = L_i L_i^T + eps * I`` (always SPD), so the primal solve
``(A_i + rho I)^{-1} (...)`` is well-posed and differentiable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from acn.topology import overlap_weights


class Encoder(nn.Module):
    """patch (P,) -> (A_i (d,d) SPD, b_i (d,)).

    Outputs ``L_flat`` of size ``d*(d+1)/2`` (lower-triangular entries of L),
    ``b`` of size ``d``, and a per-column relevance logit ``s`` of size 1
    (used by the learned column gate). ``A = L L^T + eps I``.
    """

    def __init__(self, patch_dim: int, d: int, hidden: tuple[int, ...] = (), eps: float = 1e-2,
                 relevance_init_bias: float = 3.0):
        super().__init__()
        self.d = d
        self.eps = eps
        self.tril_size = d * (d + 1) // 2
        out_dim = self.tril_size + d + 1   # +1 for the per-column relevance logit

        layers: list[nn.Module] = []
        in_dim = patch_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

        # Bias-init the relevance head high so gates start OPEN (all columns
        # active). The learned-gate sparsity penalty then prunes the useless
        # columns DOWN over training, instead of starting at ~0.5 and sliding
        # to 0 (the gate-collapse failure mode; see LOG_2026-07-05).
        # sigmoid(relevance_init_bias) is the starting gate value; +3 -> ~0.95.
        if relevance_init_bias != 0.0:
            with torch.no_grad():
                self.mlp[-1].bias[-1] = float(relevance_init_bias)

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """patches: (B, N, patch_dim) -> A: (B, N, d, d), b: (B, N, d), s: (B, N).

        s is the per-column relevance logit, used by the learned column gate.
        """
        B, N, _ = patches.shape
        out = self.mlp(patches)                          # (B, N, out_dim)
        L_flat = out[..., : self.tril_size]              # (B, N, tril_size)
        b = out[..., self.tril_size : self.tril_size + self.d]   # (B, N, d)
        s = out[..., -1]                                        # (B, N) relevance logit

        d = self.d
        L = torch.zeros(B, N, d, d, device=patches.device, dtype=patches.dtype)
        rows, cols = torch.tril_indices(d, d, device=patches.device)
        L[:, :, rows, cols] = L_flat
        A = L @ L.transpose(-1, -2) + self.eps * torch.eye(
            d, device=patches.device, dtype=patches.dtype
        )
        return A, b, s


class Decoder(nn.Module):
    """z (d,) -> logits (num_classes,)."""

    def __init__(self, d: int, num_classes: int, hidden: tuple[int, ...] = ()):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = d
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, N, d) -> logits: (B, N, num_classes)."""
        return self.mlp(z)


class RestrictionMaps(nn.Module):
    """Diagonal restriction maps F_ij = diag(r_ij).

    Symmetric: one vector r_e per edge; F_ij = F_ji = diag(r_e) — the "shared
    channel" interpretation.
    """

    def __init__(self, num_edges: int, d: int, init_scale: float = 0.5):
        super().__init__()
        self.d = d
        # softplus-inv so init ~ init_scale positive
        raw = torch.randn(num_edges, d) * 0.1 + torch.log(torch.expm1(torch.tensor(init_scale)))
        self.r = nn.Parameter(raw)

    def maps(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (r_ij, r_ji) each (E, d), positive via softplus (symmetric)."""
        r = torch.nn.functional.softplus(self.r)
        return r, r


def sparse_fuse(logits: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Fuse only over active columns. logits: (B, N, C), active: (B, N) in [0,1].

    Inactive columns (active≈0) contribute nothing to the vote. This is the
    Thousand-Brains sparse vote: only the relevant columns decide.
    """
    w = active.unsqueeze(-1)                      # (B, N, 1)
    summed = (w * logits).sum(dim=1)              # (B, C)
    count = w.sum(dim=1).clamp(min=1.0)           # (B, 1)
    return summed / count


def confident_fuse(logits: torch.Tensor, active: torch.Tensor,
                    tau: float = 1.0) -> torch.Tensor:
    """Confidence-weighted sparse fusion (the adaptive-robustness mechanism).

    Like sparse_fuse, but each active column's vote is further weighted by a
    confidence score derived from the sharpness of its OWN prediction — the
    normalized entropy of its per-class scorecard:

        H_i      = entropy(softmax(logits_i))            # 0 (sharp) .. log(C) (flat)
        H_norm_i = H_i / log(C)                           # in [0, 1]
        confidence_i = exp(−H_norm_i / tau)               # in (0, 1]

    A column whose patch is informative produces a sharp scorecard (one class
    dominates → low entropy → confidence ≈ 1) and votes at full strength. A
    column seeing a noisy / occluded / ambiguous patch produces a flat
    scorecard (high entropy → confidence → 0) and gets suppressed.

    Why entropy and not the ADMM dual ‖u_i‖? The dual measures how much a
    column *disagreed with its neighbors*, so it only grows when the column
    has active neighbors to pull on its consensus state z_i. Under the sparse
    spatial graph (~6% columns active) most active columns have NO active
    neighbor, so ‖u_i‖ stays at 0 and the dual-based confidence silently
    degenerates to 1 for ~90% of votes — i.e. it degenerates to sparse_fuse
    and the robustness mechanism does nothing. Entropy uses only the column's
    own logits, so it works for isolated columns too: a garbage patch is
    detected from its flat scorecard alone, with no graph dependency.

    The confidence is computed at runtime from the decoder output — per
    column, per input — so this is a genuinely adaptive mechanism, not a
    learned threshold.

    Args:
      logits: (B, N, C) per-column decoder output
      active: (B, N) in [0,1] — the learned gate
      tau:    temperature on the normalized entropy; smaller = more aggressive
              suppression of flat-scorecard columns. tau=1.0 → a perfectly
              flat column gets confidence exp(-1)≈0.37; tau=0.5 → exp(-2)≈0.14.
    """
    C = logits.shape[-1]
    logC = float(C)
    p = torch.softmax(logits, dim=-1)                   # (B, N, C)
    H = -(p * (p + 1e-9).log()).sum(dim=-1)             # (B, N) entropy in [0, log C]
    H_norm = H / logC                                   # (B, N) in [0, 1]
    confidence = torch.exp(-H_norm / tau)               # (B, N) in (0, 1]
    w = (active * confidence).unsqueeze(-1)             # (B, N, 1) gate × confidence
    summed = (w * logits).sum(dim=1)                    # (B, C)
    count = w.sum(dim=1).clamp(min=1e-6)                # (B, 1)
    return summed / count


def column_gate(
    relevance_logits: torch.Tensor,   # (B, N) per-column relevance logits from the encoder
    *,
    inhibition: float = 0.0,
) -> torch.Tensor:
    """Compute the per-column activation gate g_i in [0,1] (B, N).

    The gate is sigmoid(s_i) where s_i is the encoder's per-column relevance
    logit. The model LEARNS which columns fire for which inputs, and a sparsity
    penalty (column_sparsity_weight in the training loss) pushes the active
    count as low as possible while maintaining accuracy. The active count is
    DISCOVERED per input, not fixed — a "1" recruits few columns, an "8" many.

    Lateral inhibition (optional) sharpens the gate: each column is suppressed
    by the mean logit of its peers, so only columns more relevant than the
    average survive. Set `inhibition=0` (the default) to disable it and let the
    sparsity penalty handle sparsity alone.

    Returns soft gates in [0,1]; g_i≈0 means the column is SILENT in the model
    (no primal, no consensus, no dual update, no vote — see BLUEPRINT.md §3).
    """
    score = relevance_logits                             # (B, N) learned logits
    if inhibition > 0:
        # lateral inhibition: suppress each column by the peer mean logit.
        mean_score = score.mean(dim=-1, keepdim=True)    # (B, 1)
        inhibited = score - inhibition * mean_score      # (B, N)
    else:
        inhibited = score
    return torch.sigmoid(inhibited)
