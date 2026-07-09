"""Encoder, decoder, positional binding, diagonal restriction maps, fusion, and gating."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Positional Encoder (Reference Frame: WHERE)
# --------------------------------------------------------------------------- #

class PositionalEncoder(nn.Module):
    """Fixed (row, col) coordinates -> learned positional embedding."""

    def __init__(self, d: int, max_size: int = 28):
        super().__init__()
        self.pos_embed = nn.Linear(2, d)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (N, 2)  (row, col) integer coordinates
        c = coords.to(dtype=self.pos_embed.weight.dtype)
        return self.pos_embed(c)  # (N, d)


# --------------------------------------------------------------------------- #
# Motor system (efference copy + path integration in latent space)
# --------------------------------------------------------------------------- #

class MotorSystem(nn.Module):
    """Emits motor vector from z, then projects it to z and u channels.

    motor      = tanh(W_head @ z)          (B, N, motor_dim)
    shift_z    = W_to_z @ motor            (B, N, d)  -> path integration on z
    shift_u    = W_to_u @ motor            (B, N, d)  -> efference copy on u
    """

    def __init__(self, d: int, motor_dim: int = 2):
        super().__init__()
        self.d = d
        self.motor_dim = motor_dim
        self.head = nn.Linear(d, motor_dim)
        self.to_z = nn.Linear(motor_dim, d)
        self.to_u = nn.Linear(motor_dim, d)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """z: (B, N, d) -> motor (B, N, m), shift_z (B, N, d), shift_u (B, N, d)"""
        motor = torch.tanh(self.head(z))            # (B, N, m)
        shift_z = self.to_z(motor)                  # (B, N, d)
        shift_u = self.to_u(motor)                  # (B, N, d)
        return motor, shift_z, shift_u


# --------------------------------------------------------------------------- #
# Primer (Top-down preview signal from global state to bottom columns)
# --------------------------------------------------------------------------- #

class Primer(nn.Module):
    """Projects a global preview vector to a priming signal for all columns."""

    def __init__(self, d_in: int, d_out: int, hidden: int | None = None):
        super().__init__()
        if hidden is not None and hidden > 0:
            self.mlp = nn.Sequential(
                nn.Linear(d_in, hidden), nn.ReLU(),
                nn.Linear(hidden, d_out),
            )
        else:
            self.mlp = nn.Linear(d_in, d_out)

    def forward(self, z_preview: torch.Tensor) -> torch.Tensor:
        # z_preview: (B, d_in) -> p: (B, d_out)
        return self.mlp(z_preview)


# --------------------------------------------------------------------------- #
# Hard top-k gate (Gumbel-Softmax Straight-Through Estimator)
# --------------------------------------------------------------------------- #


def hard_topk_gate(
    relevance_logits: torch.Tensor, k: int, tau: float = 0.5
) -> torch.Tensor:
    """Hard top-k gate. Forward: exact binary mask. Backward: Gumbel-Softmax.

    Returns (B, N) in {0,1} with exactly k ones per sample.
    """
    B, N = relevance_logits.shape
    if k >= N:
        return torch.ones_like(relevance_logits)

    gumbel = torch.rand_like(relevance_logits).log().neg().log().neg()
    noisy = (relevance_logits + gumbel) / tau
    soft = F.softmax(noisy, dim=-1)            # (B, N)

    # hard mask: top-k entries set to 1.0
    _, idx = torch.topk(soft, k, dim=-1)       # (B, k)
    hard = torch.zeros_like(soft).scatter_(1, idx, 1.0)

    # Straight-Through: forward = hard, backward = soft gradient
    active = hard + (soft - soft.detach())
    return active


# --------------------------------------------------------------------------- #
# Restriction Maps (unchanged canonical channel)
# --------------------------------------------------------------------------- #

class RestrictionMaps(nn.Module):
    def __init__(self, num_edges: int, d: int, init_scale: float = 0.5):
        super().__init__()
        self.d = d
        raw = torch.randn(num_edges, d) * 0.1 + torch.log(torch.expm1(torch.tensor(init_scale)))
        self.r = nn.Parameter(raw)

    def maps(self) -> tuple[torch.Tensor, torch.Tensor]:
        r = F.softplus(self.r)
        return r, r


# --------------------------------------------------------------------------- #
# Predictive-coding maps (hierarchical fusion: predictions down, errors up)
# --------------------------------------------------------------------------- #

class PredictiveMaps(nn.Module):
    """Two small LINEAR maps for the inter-layer predictive-coding exchange.

    `decode_down` turns the top belief into a prediction of the bottom's
    aggregate signal (the vote distribution when use_vote_input, else the
    average z); `encode_up` turns the resulting error into a correction for
    the top. When the aggregate signal is the vote (dim != latent_dim), an
    extra `decode_down_z` map predicts the bottom z directly for the
    top-down z-nudge (so the bottom columns get guidance in their own space).
    """
    def __init__(self, d_bottom: int, d_top: int, d_z: int | None = None):
        super().__init__()
        self.d_bottom = d_bottom
        self.d_top = d_top
        self.decode_down = nn.Linear(d_top, d_bottom, bias=False)
        self.encode_up = nn.Linear(d_bottom, d_top, bias=False)
        nn.init.normal_(self.decode_down.weight, std=0.1)
        nn.init.normal_(self.encode_up.weight, std=0.1)
        # extra map for the bottom z-injection when the inter-layer signal
        # (d_bottom) is not the bottom latent dim (d_z) — e.g. vote input.
        self.decode_down_z = None
        if d_z is not None and d_z != d_bottom:
            self.decode_down_z = nn.Linear(d_top, d_z, bias=False)
            nn.init.normal_(self.decode_down_z.weight, std=0.1)


# --------------------------------------------------------------------------- #
# Encoder (What + Where binding)
# --------------------------------------------------------------------------- #

class Encoder(nn.Module):
    """patch (B,N,P) + coords (N,2) -> A (B,N,d,d), b (B,N,d), s (B,N)."""

    def __init__(
        self,
        patch_dim: int,
        d: int,
        hidden: tuple[int, ...] = (),
        eps: float = 1e-2,
        use_positional: bool = True,
        pos_embed_dim: int | None = None,
        relevance_init_bias: float = 3.0,
    ):
        super().__init__()
        self.d = d
        self.eps = eps
        self.use_positional = use_positional
        self.tril_size = d * (d + 1) // 2
        out_dim = self.tril_size + d + 1   # +1 for relevance logit

        if self.use_positional:
            pe_dim = pos_embed_dim if pos_embed_dim is not None else d
            self.pos_encoder = PositionalEncoder(pe_dim)
            self.project_where = nn.Linear(pe_dim, d)
            in_dim = patch_dim
        else:
            self.pos_encoder = None
            self.project_where = None
            in_dim = patch_dim

        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

        if relevance_init_bias != 0.0:
            with torch.no_grad():
                self.mlp[-1].bias[-1] = float(relevance_init_bias)

    def forward(
        self, patches: torch.Tensor, coords: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = patches.shape
        what = self.mlp(patches)                          # (B, N, out_dim)

        if self.use_positional and coords is not None and self.pos_encoder is not None:
            where_emb = self.pos_encoder(coords)            # (N, pe_dim)
            where = self.project_where(where_emb)          # (N, d)
            where = where.unsqueeze(0).expand(B, -1, -1)   # (B, N, d)
            # project what to d for binding (if mlp last layer is not d)
            # Actually what is out_dim, not d. We split L_flat,b,s from it.
            # But we want to add `where` to the latent representation before heads.
            # The simple way: add where to the first d dims of what (or we can
            # split and inject). Here we keep it simple: the binding happens at
            # the head level by adding to `b`.
            # Re-design: what is already out_dim. We'll just add where to the
            # b-component after extraction.
            pass

        L_flat = what[..., : self.tril_size]              # (B, N, tril_size)
        b = what[..., self.tril_size : self.tril_size + self.d]   # (B, N, d)
        s = what[..., -1]                                 # (B, N)

        if self.use_positional and coords is not None and self.pos_encoder is not None:
            where = self.project_where(self.pos_encoder(coords))  # (N, d)
            where = where.unsqueeze(0).expand(B, -1, -1)
            b = b + where  # Reference-frame binding: where modulates linear term

        d = self.d
        L = torch.zeros(B, N, d, d, device=patches.device, dtype=patches.dtype)
        rows, cols = torch.tril_indices(d, d, device=patches.device)
        L[:, :, rows, cols] = L_flat
        A = L @ L.transpose(-1, -2) + self.eps * torch.eye(
            d, device=patches.device, dtype=patches.dtype
        )
        return A, b, s


# --------------------------------------------------------------------------- #
# Abstract Encoder (Layer 2)
# --------------------------------------------------------------------------- #

class AbstractEncoder(nn.Module):
    """Global state z (B, d_in) + abstract positions (N, 2) -> A, b, s for N abstract columns."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_cols: int,
        hidden: tuple[int, ...] = (),
        eps: float = 1e-2,
    ):
        super().__init__()
        self.d_out = d_out
        self.num_cols = num_cols
        self.eps = eps
        self.tril_size = d_out * (d_out + 1) // 2
        self.out_dim = self.tril_size + d_out + 1

        self.pos_embed = nn.Linear(2, d_out)
        self.mlp = _build_mlp(d_in, hidden, d_out)
        self.head = nn.Linear(d_out, self.out_dim)

    def forward(self, z_global: torch.Tensor, abstract_pos: torch.Tensor) -> tuple:
        # z_global: (B, d_in)   -> processed by mlp -> (B, d_out)
        # abstract_pos: (N, 2)  -> learned positions
        what = self.mlp(z_global)                          # (B, d_out)
        where = self.pos_embed(abstract_pos)              # (N, d_out)
        h = what.unsqueeze(1) + where.unsqueeze(0)          # (B, N, d_out)
        out = self.head(h)                                   # (B, N, out_dim)

        L_flat = out[..., : self.tril_size]
        b = out[..., self.tril_size : self.tril_size + self.d_out]
        s = out[..., -1]

        B, N = h.shape[:2]
        d = self.d_out
        L = torch.zeros(B, N, d, d, device=h.device, dtype=h.dtype)
        rows, cols = torch.tril_indices(d, d, device=h.device)
        L[:, :, rows, cols] = L_flat
        A = L @ L.transpose(-1, -2) + self.eps * torch.eye(d, device=h.device, dtype=h.dtype)
        return A, b, s


# --------------------------------------------------------------------------- #
# Decoder (unchanged canonical)
# --------------------------------------------------------------------------- #

class Decoder(nn.Module):
    def __init__(self, d: int, num_classes: int, hidden: tuple[int, ...] = ()):
        super().__init__()
        self.mlp = _build_mlp(d, hidden, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


# --------------------------------------------------------------------------- #
# Fusion helpers
# --------------------------------------------------------------------------- #

def sparse_fuse(logits: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Fuse only over active columns. logits: (B, N, C), active: (B, N)."""
    w = active.unsqueeze(-1)                      # (B, N, 1)
    summed = (w * logits).sum(dim=1)                 # (B, C)
    count = w.sum(dim=1).clamp(min=1.0)              # (B, 1)
    return summed / count


def sparse_fuse_vectors(z: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Fuse (average) vector states over active columns. z: (B, N, d), active: (B, N)."""
    w = active.unsqueeze(-1)                       # (B, N, 1)
    summed = (w * z).sum(dim=1)                      # (B, d)
    count = w.sum(dim=1).clamp(min=1.0)               # (B, 1)
    return summed / count


def vote_fuse(z: torch.Tensor, active: torch.Tensor, decoder: nn.Module,
              num_classes: int, temperature: float = 1.0) -> torch.Tensor:
    """Soft vote distribution: mean over active columns of softmax(decoder(z)).

    z: (B, N, d), active: (B, N), decoder: Module mapping z (B,N,d) -> logits (B,N,C).
    Returns (B, C) — a differentiable soft vote histogram. Each active column's
    softmax(decoder(z_i)) is that column's smooth vote over the C classes; the
    mean of these is the vote distribution the top layer sees.

    Unlike argmax+histogram, this is fully smooth (softmax + mean), so BPTT
    flows through it with no special handling. As columns get confident their
    softmaxes sharpen toward one-hot, so the soft vote approaches a hard vote
    histogram in the limit — but stays differentiable throughout.
    """
    logits = decoder(z)                            # (B, N, C) — grad flows to decoder
    probs = F.softmax(logits / temperature, dim=-1)  # (B, N, C) — smooth per-column vote
    w = active.unsqueeze(-1)                       # (B, N, 1)
    summed = (w * probs).sum(dim=1)                # (B, C)
    count = w.sum(dim=1).clamp(min=1e-6)           # (B, 1)
    return summed / count                          # (B, C)


def logit_margin_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample confidence = top1 - top2 logit margin, floored to avoid zeros.

    logits: (B, C) -> (B,). Higher = more confident (a clear winner vs an
    almost-tie). Used to weight the two layers' cooperative readout: whichever
    layer is more sure on THIS input leads the decision, the other still
    contributes (helping each other, neither discarded). We use the margin
    (not softmax) because softmax saturates (~0.9-0.998 for every sample, no
    usable range); the margin has a wide, honest dynamic range.
    """
    top2 = torch.topk(logits, 2, dim=-1).values       # (B, 2)
    margin = (top2[..., 0] - top2[..., 1]).clamp(min=1e-3)  # (B,)
    return margin


# --------------------------------------------------------------------------- #
# MLP builder
# --------------------------------------------------------------------------- #

def _build_mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU()]
        in_dim = h
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
