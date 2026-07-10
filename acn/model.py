"""PCN-ACN model: predictive-coding consensus network, EP-trained.

Architecture (one squared-error energy, one minimum — EP-native):
  - f_1 (per-column encoder): patch_i -> predicted h1_i
  - g_θ (lateral predictor): neighbor's h1 -> predicted h1_i  (the "consensus")
  - D   (shared per-column decoder): h1_i -> logits_i (one verdict per column)

Per-column readout (Sheaf-ADMM-style): the decoder runs on each column's latent
individually, the label is broadcast to all columns, and the global prediction is
the average of the per-column logits. So every column is decoded on its own —
the per-column color grid is the trained, in-distribution readout, not an OOD
isolation hack.

The free variables (h1, ℓ) settle by gradient descent on the energy to the
minimum where every prediction matches (the feedforward pass with lateral
agreement). Trained with centered EP — no BPTT.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from acn.config import DataConfig, ModelConfig
from acn.flatstate import State, LiveCtx, Scalars
from acn.settle import settle
from acn.eqprop import eqprop_loss as _eqprop_loss


# ------------------------------------------------------------------ #
# Image decomposition into overlapping patches (inlined)
# ------------------------------------------------------------------ #

@dataclass
class _Decomposition:
    """patches (B, N, C*P*P), coords (N, 2) top-left grid positions, grid (Hg, Wg)."""
    patches: torch.Tensor
    coords: torch.Tensor
    grid: tuple[int, int]

    @property
    def num_patches(self) -> int:
        return self.patches.shape[1]


def _num_patches_along(size: int, patch: int, stride: int) -> int:
    if patch > size:
        raise ValueError(f"patch {patch} > size {size}")
    n = (size - patch) // stride + 1
    return max(n, 1)


def _decompose(images: torch.Tensor, patch_size: int, stride: int) -> _Decomposition:
    """images (B, C, H, W) -> patches (B, N, C*P*P), row-major over the grid."""
    if images.dim() != 4:
        raise ValueError(f"expected (B,C,H,W); got {tuple(images.shape)}")
    B, C, H, W = images.shape
    Hg = _num_patches_along(H, patch_size, stride)
    Wg = _num_patches_along(W, patch_size, stride)
    unfolded = F.unfold(images, kernel_size=patch_size, stride=stride)  # (B, C*P*P, L)
    patches = unfolded.transpose(1, 2).contiguous()                    # (B, L, C*P*P)
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(Hg) * stride, torch.arange(Wg) * stride, indexing="ij",
        ), dim=-1,
    ).reshape(-1, 2).long()                                            # (N, 2)
    return _Decomposition(patches=patches, coords=coords, grid=(Hg, Wg))


# ------------------------------------------------------------------ #
# Edge topology (inlined): spatial neighbors from overlapping patches
# ------------------------------------------------------------------ #

def _build_spatial_neighbors(deco: _Decomposition, patch_size: int,
                             overlap_min: int = 1) -> torch.Tensor:
    """Edges (2, E) between patches overlapping by >= overlap_min on BOTH axes."""
    coords = deco.coords
    r, c = coords[:, 0], coords[:, 1]
    dr = (r[:, None] - r[None, :]).abs()
    dc = (c[:, None] - c[None, :]).abs()
    overlap = torch.minimum((patch_size - dr).clamp(min=0), (patch_size - dc).clamp(min=0))
    mask = (overlap >= overlap_min)
    mask.fill_diagonal_(False)
    tri = torch.triu(torch.ones_like(mask), diagonal=1).bool()
    edges = (mask & tri).nonzero(as_tuple=False).T
    return edges.contiguous()


# ------------------------------------------------------------------ #
# Hard top-k gate (Gumbel-Softmax Straight-Through) (inlined)
# ------------------------------------------------------------------ #

def _hard_topk_gate(relevance: torch.Tensor, k: int, tau: float = 0.5) -> torch.Tensor:
    """Hard top-k mask (B, N) in {0,1}; ST estimator so grads flow as Gumbel-Softmax."""
    B, N = relevance.shape
    if k >= N:
        return torch.ones_like(relevance)
    gumbel = torch.rand_like(relevance).log().neg().log().neg()
    soft = F.softmax((relevance + gumbel) / tau, dim=-1)
    _, idx = torch.topk(soft, k, dim=-1)
    hard = torch.zeros_like(soft).scatter_(1, idx, 1.0)
    return hard + (soft - soft.detach())


class ColumnEncoder(nn.Module):
    """f_1: patch (B,N,P) + coords (N,2) -> predicted per-column latent h1 (B,N,d).

    A shared MLP applied per column, with positional binding (the patch's
    coordinates modulate the prediction so columns know 'where' they are).
    """
    def __init__(self, patch_dim, d, hidden):
        super().__init__()
        self.d = d
        in_dim = patch_dim + 2  # +2 for (row, col) positional binding
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]; in_dim = h
        layers.append(nn.Linear(in_dim, d))
        self.mlp = nn.Sequential(*layers)

    def forward(self, patches, coords):
        # patches (B,N,P), coords (N,2) -> (B,N,d)
        B, N, P = patches.shape
        cp = coords.unsqueeze(0).expand(B, -1, -1).to(patches.dtype)
        x = torch.cat([patches, cp], dim=-1)
        return self.mlp(x)


class LateralPredictor(nn.Module):
    """g_θ: a column's latent -> predicted latent for each neighbor.

    A shared MLP. For edge (i,j): g_θ(h1_j) predicts h1_i, g_θ(h1_i) predicts h1_j.
    This is the "consensus" — each column tries to match what its neighbors
    predict it should be. (The sheaf/restriction role, re-homed as a predictor.)
    """
    def __init__(self, d, hidden=()):
        super().__init__()
        layers = []
        in_dim = d
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]; in_dim = h
        layers.append(nn.Linear(in_dim, d))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h1, edges):
        ei, ej = edges[0], edges[1]
        pred_i = self.mlp(h1[:, ej])   # predict h1_i from h1_j
        pred_j = self.mlp(h1[:, ei])   # predict h1_j from h1_i
        return pred_i, pred_j


class ColumnDecoder(nn.Module):
    """D: shared per-column decoder. h1_i (B,N,d) -> per-column logits (B,N,C).

    Applied per column (the Linear acts on the last axis, shared across
    columns). This is the trained readout — each column is decoded individually,
    matching Sheaf-ADMM's per-agent classification head (readout_mode=x_only).
    The global prediction is the average of the per-column logits (see
    ACNv2.forward), so the fuse happens *after* the decode, not before.
    """
    def __init__(self, d, num_classes):
        super().__init__()
        self.mlp = nn.Linear(d, num_classes)

    def forward(self, h1):
        # h1 (B,N,d) -> (B,N,C); the Linear acts on the last axis, shared.
        return self.mlp(h1)


class ACNv2(nn.Module):
    """Predictive-coding consensus network, EP-trained (per-column readout)."""

    def __init__(self, image_size, patch_size, stride, model_cfg: ModelConfig,
                 data_cfg: DataConfig | None = None):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.cfg = model_cfg
        self.data_cfg = data_cfg
        patch_dim = patch_size * patch_size

        d, C = model_cfg.latent_dim, model_cfg.num_classes
        self.d, self.C = d, C

        self.column_encoder = ColumnEncoder(
            patch_dim=patch_dim, d=d, hidden=model_cfg.enc_hidden)
        self.lateral_predictor = LateralPredictor(d=d, hidden=model_cfg.lateral_hidden)
        self.column_decoder = ColumnDecoder(d=d, num_classes=C)
        self.topk_k = model_cfg.topk_k
        self.topk_tau = model_cfg.topk_tau

        self._topo = None
        self._build_topology()

    # ------------------------------------------------------------------ #
    # topology
    # ------------------------------------------------------------------ #

    def _build_topology(self):
        dummy = torch.zeros(1, 1, self.image_size, self.image_size)
        deco = _decompose(dummy, self.patch_size, self.stride)
        edges = _build_spatial_neighbors(deco, self.patch_size)
        self._topo = dict(edges=edges, coords=deco.coords)
        self._N = deco.num_patches

    def _to_device(self, device, dtype):
        self._topo["edges"] = self._topo["edges"].to(device)
        self._topo["coords"] = self._topo["coords"].to(device)

    # ------------------------------------------------------------------ #
    # prepare: images -> LiveCtx + State + Scalars
    # ------------------------------------------------------------------ #

    def _prepare(self, images):
        device, dtype = images.device, images.dtype
        self._to_device(device, dtype)

        deco = _decompose(images, self.patch_size, self.stride)
        patches = deco.patches.to(dtype)                       # (B, N, P)
        coords = self._topo["coords"].to(device)               # (N, 2)
        edges = self._topo["edges"]                            # (2, E)
        B, N, _ = patches.shape

        # per-column prediction f_1(patch_i) — live
        h1_pred = self.column_encoder(patches, coords)         # (B, N, d)

        # gate (all-active by default; top-k optional)
        # use the norm of the prediction as the relevance signal
        rel = h1_pred.norm(dim=-1)                             # (B, N)
        active = _hard_topk_gate(rel, k=self.topk_k, tau=self.topk_tau)

        ctx = LiveCtx(
            h1_pred=h1_pred, active=active, edges=edges,
            lateral_predict=self.lateral_predictor,
            column_decode=self.column_decoder,
        )
        sc = Scalars(kappa=self.cfg.kappa, lam_col=self.cfg.lam_col,
                     lam_output=self.cfg.lam_output)

        # initial state: warm-start at the predictions (the free minimum is nearby)
        h1_init = h1_pred.detach() * active.unsqueeze(-1)
        llogits_init = self.column_decoder(h1_init).detach()   # (B, N, C) per-column
        state0 = State(h1=h1_init, llogits=llogits_init)
        return ctx, sc, state0

    def _settle_kwargs(self):
        return dict(k_max=self.cfg.k_max, alpha=self.cfg.alpha)

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #

    def forward(self, images):
        ctx, sc, state0 = self._prepare(images)
        s, _ = settle(state0, ctx, sc, beta=0.0, target=None, **self._settle_kwargs())
        # global prediction = average of the per-column logits over active columns
        active = ctx.active                              # (B, N)
        n_act = active.sum(1).clamp(min=1.0)             # (B,)
        per_col_logits = self.column_decoder(s.h1)       # (B, N, C) — the trained readout
        fused = (active.unsqueeze(-1) * per_col_logits).sum(1) / n_act.unsqueeze(-1)  # (B, C)
        return fused, s

    # ------------------------------------------------------------------ #
    # training: centered equilibrium propagation
    # ------------------------------------------------------------------ #

    def eqprop_loss(self, images, targets):
        ctx, sc, state0 = self._prepare(images)
        return _eqprop_loss(state0, ctx, sc, targets,
                            beta=self.cfg.beta,
                            **self._settle_kwargs())
