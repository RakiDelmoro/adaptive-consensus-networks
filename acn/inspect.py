"""Extract & serialize interpretable ACN state, plus summary statistics.

The model's :class:`acn.consensus.ConsensusState` exposes everything we need:
per-mini-network ``x, z, u`` and per-link ``D, Q``. This module turns those tensors
into numpy arrays for visualization and into compact summary scalars for logging.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch

from acn.consensus import ConsensusState


@dataclass
class StateSnapshot:
    """Numpy snapshot of one forward pass's interpretable state."""

    x: np.ndarray         # (B, N, d) primal
    z: np.ndarray         # (B, N, d) consensus
    u: np.ndarray         # (B, N, d) dual
    D: np.ndarray         # (B, E) conductance
    Q: np.ndarray         # (B, E) flux
    edges: np.ndarray     # (2, E)
    coords: np.ndarray    # (N, 2)
    logits_per_node: np.ndarray | None = None  # (B, N, C)
    pred: np.ndarray | None = None             # (B, C)
    active: np.ndarray | None = None           # (B, N) column gate in [0,1]
    history: list[dict] | None = None          # per-round snapshots

    def save(self, path: str) -> None:
        savez = dict(x=self.x, z=self.z, u=self.u, D=self.D, Q=self.Q,
                     edges=self.edges, coords=self.coords,
                     logits_per_node=self.logits_per_node, pred=self.pred,
                     active=self.active)
        np.savez(path, **savez)

    @classmethod
    def load(cls, path: str) -> "StateSnapshot":
        d = np.load(path, allow_pickle=False)
        return cls(
            x=d["x"], z=d["z"], u=d["u"], D=d["D"], Q=d["Q"],
            edges=d["edges"], coords=d["coords"],
            logits_per_node=d["logits_per_node"] if "logits_per_node" in d.files else None,
            pred=d["pred"] if "pred" in d.files else None,
            active=d["active"] if "active" in d.files else None,
            history=None,
        )


def snapshot(
    model,
    images: torch.Tensor,
    labels: torch.Tensor | None = None,
    record: bool = True,
) -> tuple[StateSnapshot, torch.Tensor]:
    """Run a forward pass and capture a full state snapshot (no grad)."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(images, record=record)
        pred, state = out  # type: ignore[misc]
        state: ConsensusState = state
    if was_training:
        model.train()
    snap = StateSnapshot(
        x=state.x.detach().cpu().numpy(),
        z=state.z.detach().cpu().numpy(),
        u=state.u.detach().cpu().numpy(),
        D=state.D.detach().cpu().numpy(),
        Q=state.Q.detach().cpu().numpy(),
        edges=model.topology.edges.cpu().numpy(),
        coords=model.topology.coords.cpu().numpy(),
        logits_per_node=model.decoder(state.z).detach().cpu().numpy(),
        pred=pred.detach().cpu().numpy(),
        active=state.active.detach().cpu().numpy(),
        history=(
            [{k: v.cpu().numpy() for k, v in h.items()} for h in state.history]
            if state.history else None
        ),
    )
    return snap, pred


@dataclass
class LinkSummary:
    """Per-edge summary scalars (averaged over batch)."""

    active_link_frac: float     # fraction of edges with mean D >= prune_eps
    mean_D: float
    max_D: float
    mean_Q: float
    num_edges: int
    num_nodes: int
    active_column_frac: float = 1.0   # fraction of columns with gate > 0.5


@dataclass
class NodeSummary:
    """Per-node summary scalars (averaged over batch)."""

    mean_u_norm: float          # stubbornness
    max_u_norm: float
    mean_x_z_gap: float         # ||x - z||


def link_summary(snap: StateSnapshot, prune_eps: float = 1e-3) -> LinkSummary:
    D = snap.D.mean(axis=0)                      # (E,)
    Q = snap.Q.mean(axis=0)                      # (E,)
    active = (D >= prune_eps).mean()
    col_frac = (snap.active.mean(axis=0) > 0.5).mean()
    return LinkSummary(
        active_link_frac=float(active),
        mean_D=float(D.mean()), max_D=float(D.max()),
        mean_Q=float(Q.mean()),
        num_edges=snap.D.shape[1], num_nodes=snap.x.shape[1],
        active_column_frac=float(col_frac),
    )


def node_summary(snap: StateSnapshot) -> NodeSummary:
    u_norm = np.linalg.norm(snap.u, axis=-1)     # (B, N)
    xz_gap = np.linalg.norm(snap.x - snap.z, axis=-1)
    return NodeSummary(
        mean_u_norm=float(u_norm.mean()),
        max_u_norm=float(u_norm.max()),
        mean_x_z_gap=float(xz_gap.mean()),
    )


def conductance_matrix(snap: StateSnapshot, agg: str = "mean") -> np.ndarray:
    """N x N conductance adjacency (averaged or maxed over batch)."""
    N = snap.x.shape[1]
    M = np.zeros((N, N))
    ei, ej = snap.edges[0], snap.edges[1]
    D = snap.D if agg == "max" else snap.D.mean(axis=0, keepdims=True)
    D = D if agg == "max" else snap.D.mean(axis=0)
    if agg == "max":
        Dv = D.max(axis=0)
    else:
        Dv = D
    for e, (i, j) in enumerate(zip(ei, ej)):
        v = float(Dv[e])
        M[i, j] = v
        M[j, i] = v
    return M
