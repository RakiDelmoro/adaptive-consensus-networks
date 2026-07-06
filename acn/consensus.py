"""ACN consensus core: primal / consensus / dual updates + Physarum conductance.

Two implementations live here:

* :func:`numpy_consensus_step` and friends — a deliberately simple NumPy reference
  matching the spec pseudocode exactly. Used by tests to validate the torch path.
* :class:`ACNCore` — the batched, differentiable torch implementation that the model
  backprops through for all K rounds.

Math (per mini network i, edge (i,j)):

  Primal:      x_i = (A_i + rho I)^{-1} (b_i + rho (z_i - u_i))
  Flux:        q_ij = || F_ij z_i - F_ji z_j ||^2
  Conductance: D_ij = clip( D_ij + eta_D * q/(1+q) - gamma_D * D_ij, 0, D_clip )
  Consensus:   z_i -= eta_z * sum_j D_ij * F_ij^T (F_ij z_i - F_ji z_j)
  Dual:        u_i += x_i - z_i

Restriction maps are diagonal: F_ij = diag(r_ij), symmetric (r_ij = r_ji),
realizing the "shared channel" interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# ===================================================================== #
# NumPy reference
# ===================================================================== #


def _np_lower_tri(flat: np.ndarray, d: int) -> np.ndarray:
    """Build a (d, d) lower-triangular matrix from d*(d+1)/2 flat params."""
    L = np.zeros((d, d))
    idx = np.tril_indices(d)
    L[idx] = flat
    return L


def numpy_consensus_step(
    A: np.ndarray,        # (N, d, d)
    b: np.ndarray,        # (N, d)
    r_ij: np.ndarray,     # (E, d) restriction for direction i->j
    r_ji: np.ndarray,     # (E, d) restriction for direction j->i
    edges: np.ndarray,    # (E, 2)
    x: np.ndarray,        # (N, d)
    z: np.ndarray,        # (N, d)
    u: np.ndarray,        # (N, d)
    D: np.ndarray,        # (E,)
    rho: float,
    eta_z: float,
    eta_D: float,
    gamma_D: float,
    D_clip: float,
    diffusion_steps: int = 1,
):
    """One ADMM round (primal + T diffusion+conductance steps + dual), NumPy.

    The consensus update is a gradient step on
        (rho/2) ||x_i - z_i + u_i||^2  +  (1/2) sum_ij D_ij ||F_ij z_i - F_ji z_j||^2
    i.e. the standard ADMM consensus pull toward (x_i + u_i) *plus* pairwise
    disagreement smoothing. The pull term is required: without it z=0 (or any
    constant) is a fixed point of pure diffusion and the decoder sees nothing.
    """
    N, d = A.shape[:2]
    I = np.eye(d)
    # --- primal ---
    for i in range(N):
        M = A[i] + rho * I
        rhs = b[i] + rho * (z[i] - u[i])
        x[i] = np.linalg.solve(M, rhs)

    Q = np.zeros(edges.shape[0])
    # === Option A: measure disagreement on RAW proposals x (before consensus) ===
    if edges.shape[0] > 0:
        ei, ej = edges[:, 0], edges[:, 1]
        diff_x = r_ij * x[ei] - r_ji * x[ej]      # (E, d) flux on raw proposals
        Q = (diff_x ** 2).sum(-1)                  # (E,)
        q_max = Q.max() + 1e-8 if Q.size > 0 else 1.0
        phi = Q / q_max                            # relative feedback (max-normalized, matches torch)
        D = np.clip(D + eta_D * phi - gamma_D * D, 0.0, D_clip) if Q.size > 0 else D

    for _ in range(diffusion_steps):
        # consensus update (flows through the just-updated D)
        ei, ej = edges[:, 0], edges[:, 1]
        diff = r_ij * z[ei] - r_ji * z[ej]          # (E, d) flux on consensus state
        z_new = z.copy()
        for i in range(N):
            z_new[i] = z_new[i] + eta_z * rho * (x[i] + u[i] - z_new[i])
        for e, (i, j) in enumerate(edges):
            grad_i = D[e] * (r_ij[e] * diff[e])      # F_ij^T (F_ij z_i - F_ji z_j)
            z_new[i] = z_new[i] - eta_z * grad_i
            grad_j = D[e] * (r_ji[e] * (-diff[e]))   # F_ji^T (F_ji z_j - F_ij z_i)
            z_new[j] = z_new[j] - eta_z * grad_j
        z = z_new

    # --- dual ---
    u = u + (x - z)
    return x, z, u, D, Q


# ===================================================================== #
# Torch core
# ===================================================================== #


@dataclass
class ConsensusState:
    """Tensors carried across consensus rounds (batched).

    All have a leading batch dim B (except D which is per-sample per-edge).

    x, z, u: (B, N, d)
    D:      (B, E)
    Q:      (B, E) flux of the last diffusion step
    history: list of dicts (one per round) with x/z/u/D/Q snapshots (if record=True)
    """

    x: torch.Tensor
    z: torch.Tensor
    u: torch.Tensor
    D: torch.Tensor
    Q: torch.Tensor
    history: list[dict] | None = None
    # per-node logits (B, N, C); set by the model after decoding z
    logits: torch.Tensor | None = None
    # per-column activation gate (B, N) in [0,1]. Set by the model from the
    # learned column_gate(); inactive columns (~0) stay silent in consensus and
    # are excluded from fusion.
    active: torch.Tensor | None = None
    # per-column relevance logits (B, N) from the encoder, used by the gate
    # (active = sigmoid(relevance_logits)) and by the z-loss regularizer in the
    # training loss (keeps the gate's logits bounded so it can't saturate).
    relevance_logits: torch.Tensor | None = None


def make_state(
    B: int, N: int, E: int, d: int,
    device: torch.device | str,
    dtype: torch.dtype,
    D_init: torch.Tensor | None = None,
) -> ConsensusState:
    z = torch.zeros(B, N, d, device=device, dtype=dtype)
    x = torch.zeros_like(z)
    u = torch.zeros_like(z)
    if D_init is None:
        D = torch.zeros(B, E, device=device, dtype=dtype)
    else:
        D = D_init.to(device=device, dtype=dtype).expand(B, -1).clone()
    Q = torch.zeros(B, E, device=device, dtype=dtype)
    return ConsensusState(x=x, z=z, u=u, D=D, Q=Q)


class ACNCore:
    """Batched, differentiable consensus loop.

    Stateless helper object; holds no parameters. The model supplies A, b, r, edges.
    """

    @staticmethod
    def primal(
        A: torch.Tensor,    # (B, N, d, d)
        b: torch.Tensor,    # (B, N, d)
        z: torch.Tensor,    # (B, N, d)
        u: torch.Tensor,    # (B, N, d)
        rho: float,
    ) -> torch.Tensor:
        """x_i = (A_i + rho I)^{-1} (b_i + rho (z_i - u_i))."""
        B, N, d, _ = A.shape
        I = torch.eye(d, device=A.device, dtype=A.dtype)
        M = A + rho * I                              # (B, N, d, d)
        rhs = b + rho * (z - u)                       # (B, N, d)
        x = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)
        return x

    @staticmethod
    def flux(
        z: torch.Tensor,      # (B, N, d)
        r_ij: torch.Tensor,   # (E, d) or (B, E, d)
        r_ji: torch.Tensor,   # (E, d) or (B, E, d)
        ei: torch.Tensor,     # (E,) long
        ej: torch.Tensor,     # (E,) long
    ) -> torch.Tensor:
        """Per-edge F_ij z_i - F_ji z_j, shape (B, E, d)."""
        z_ei = z[:, ei]                              # (B, E, d)
        z_ej = z[:, ej]                              # (B, E, d)
        if r_ij.dim() == 2:
            r_ij = r_ij.unsqueeze(0)                 # (1, E, d)
            r_ji = r_ji.unsqueeze(0)
        diff = r_ij * z_ei - r_ji * z_ej             # (B, E, d)
        return diff

    @staticmethod
    def conductance_update(
        D: torch.Tensor,     # (B, E)
        Q: torch.Tensor,     # (B, E)
        eta_D: float | torch.Tensor,
        gamma_D: float | torch.Tensor,
        D_clip: float,
    ) -> torch.Tensor:
        """Physarum rule on detached flux (D is dynamics, not backprop).

        Growth signal phi = Q / Q_max (RELATIVE, max-normalized per batch): the
        strongest link gets phi=1, the weakest gets phi<<1, so the full range of
        Q is preserved and weak links actually drop toward zero. eps guards
        division (and the no-edge case).
        """
        Qd = Q.detach()
        Q_max = Qd.amax(dim=1, keepdim=True)
        Q_max = torch.where(torch.isfinite(Q_max) & (Q_max > 0), Q_max,
                            torch.ones_like(Q_max)) + 1e-8
        phi = Qd / Q_max
        D = D + eta_D * phi - gamma_D * D
        return D.clamp(0.0, D_clip)

    @staticmethod
    def consensus_step(
        z: torch.Tensor,     # (B, N, d)
        x: torch.Tensor,     # (B, N, d)
        u: torch.Tensor,     # (B, N, d)
        r_ij: torch.Tensor,  # (E, d)
        r_ji: torch.Tensor,  # (E, d)
        D: torch.Tensor,     # (B, E)
        ei: torch.Tensor, ej: torch.Tensor,
        eta_z: float,
        rho: float,
    ) -> torch.Tensor:
        """z_i <- z_i + eta_z * (rho (x_i + u_i - z_i) - sum_j D_ij F_ij^T(...))."""
        diff = ACNCore.flux(z, r_ij, r_ji, ei, ej)   # (B, E, d)
        if r_ij.dim() == 2:
            r_ij_b = r_ij.unsqueeze(0)
            r_ji_b = r_ji.unsqueeze(0)
        else:
            r_ij_b, r_ji_b = r_ij, r_ji
        grad_i = D.unsqueeze(-1) * (r_ij_b * diff)    # (B, E, d)  F_ij^T(...)
        grad_j = D.unsqueeze(-1) * (r_ji_b * (-diff))
        disagreement = torch.zeros_like(z)
        disagreement.index_add_(1, ei, grad_i)
        disagreement.index_add_(1, ej, grad_j)
        pull = rho * (x + u - z)
        return z + eta_z * (pull - disagreement)

    @staticmethod
    def run(
        A: torch.Tensor, b: torch.Tensor,
        r_ij: torch.Tensor, r_ji: torch.Tensor,
        edges: torch.Tensor,        # (2, E) or (E, 2)
        D_init: torch.Tensor,       # (E,) initial conductance per edge
        num_nodes: int,
        *,
        rounds: int,
        diffusion_steps: int,
        rho: float, eta_z: float,
        eta_D, gamma_D, D_clip: float,
        record: bool = False,
        detach_after: int | None = None,
        active: torch.Tensor,   # (B, N) in [0,1] — sparse column gate (always provided)
    ) -> ConsensusState:
        """Run K ADMM rounds. Returns final ConsensusState (+history if record).

        `active` (B, N) gates columns: inactive columns (active≈0) do not
        propose (x=0), are not pulled by neighbors (z stays 0, edges to/from
        them carry no flow), and do not update their dual. This is the
        Thousand-Brains sparse-column mechanism.
        """
        if edges.dim() == 2 and edges.shape[0] == 2:
            ei, ej = edges[0], edges[1]
        elif edges.dim() == 2 and edges.shape[1] == 2:
            ei, ej = edges[:, 0], edges[:, 1]
        else:
            raise ValueError(f"bad edges shape {tuple(edges.shape)}")

        B, N, d = b.shape
        device, dtype = b.device, b.dtype
        state = make_state(B, N, edges.shape[-1], d, device, dtype, D_init=D_init)
        state.active = active
        history: list[dict] | None = [] if record else None

        # column mask: (B, N) -> broadcast to (B, N, 1) for state gating
        col_mask = active.unsqueeze(-1)   # (B, N, 1)
        # edge mask: an edge carries flow only if BOTH endpoints are active
        edge_mask = active[:, ei] * active[:, ej]    # (B, E)

        for k in range(rounds):
            # optionally detach graph to save memory on later rounds
            if detach_after is not None and k >= detach_after:
                state.x = state.x.detach().requires_grad_(False)
                state.z = state.z.detach().requires_grad_(False)
                state.u = state.u.detach().requires_grad_(False)

            # primal: each patch's RAW local proposal. Inactive columns get b=0
            # so their solve yields x=0 (no proposal).
            state.x = ACNCore.primal(A, b * col_mask, state.z, state.u, rho)

            # conductance on raw-proposal disagreement, masked by column activity
            # (edges with an inactive endpoint carry no flow, so no Q, no growth).
            diff = ACNCore.flux(state.x, r_ij, r_ji, ei, ej)        # (B, E, d)
            Q = diff.pow(2).sum(-1)                                  # (B, E)
            Q = Q * edge_mask
            state.Q = Q
            state.D = ACNCore.conductance_update(state.D, Q, eta_D, gamma_D, D_clip)
            state.D = state.D * edge_mask    # silent columns -> no wire flow

            # T diffusion steps: consensus now flows through the just-updated D
            for _ in range(diffusion_steps):
                state.z = ACNCore.consensus_step(
                    state.z, state.x, state.u, r_ij, r_ji,
                    state.D, ei, ej, eta_z, rho,
                )
                # inactive columns are not pulled by consensus: keep their z=0
                state.z = state.z * col_mask

            # dual: only active columns accumulate compromise
            state.u = state.u + col_mask * (state.x - state.z)

            if history is not None:
                hist = {
                    "x": state.x.detach(), "z": state.z.detach(),
                    "u": state.u.detach(), "D": state.D.detach(),
                    "Q": state.Q.detach(), "active": state.active.detach(),
                }
                history.append(hist)
        state.history = history
        return state
