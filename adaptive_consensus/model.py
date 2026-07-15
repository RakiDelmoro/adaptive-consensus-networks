"""The Lean Sheaf-ADMM consensus network model.

A lean Sheaf-ADMM model for MNIST classification. 49 agents on a 7x7 grid, each
owning a 4x4 patch, negotiate a global digit prediction through sheaf-structured
communication over 84 edges. Trained end-to-end via backprop through the
unrolled ADMM rounds.

Solver options:
* **Local objective / x-update**: ``objective_mode='lasso'`` uses a diagonal
  quadratic + L1 solved by the **diagonal-prox** soft-thresholding operator
  (exact, no inner iterations) — the default local solve. The legacy
  ``'quadratic'`` mode uses ``Q = L L^T + eps I`` with a closed-form Woodbury
  solve.
* **z-update (consensus)**: ``z_solver='cg_project'`` runs an **unrolled
  conjugate-gradient** solve on the sheaf Laplacian with a hard ``ker(F)``
  projection (``z = z_target - (L_s + eps I)^{-1} L_s z_target``, ``Fz -> 0``) —
  the consensus step. The legacy ``'gd'`` mode uses plain gradient-descent
  sheaf diffusion (soft, no projection).
* **Restriction maps**: per-edge ``[E, 2, k, d]`` learned matrices.
* **Latent/channel**: ``d=16, k=8``.

Tensor conventions
------------------
* ``x, z, u``: ``(N, B, d)``  (node, batch, latent)
* ``edge_indices``: ``(E, 2)``
* per-edge tensors: ``(E, B, ...)``
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .graph import sinusoidal_pos_code

# ---------------------------------------------------------------------------
# Shared encoder (MLP, no CNN) and shared decoder
# ---------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    """Flat local view (+ optional reference-frame code) -> local objective params.

    Two modes:
    * ``objective_mode='quadratic'`` (legacy): outputs ``L_i`` (low-rank factor,
      ``Q_i = L_i L_i^T + q_eps I``) and ``p_i`` (linear term) for the exact
      Woodbury x-solve.
    * ``objective_mode='lasso'``: outputs ``q_diag_i`` (diagonal curvature,
      > 0 via softplus), ``q_i`` (linear term), and a per-dim ``l1_weight_i``
      for the diagonal-prox soft-thresholding x-solve.
    Shared across all agents. ``input_dim`` should already include the
    reference-frame code dim when ``d_pos > 0``.
    """

    def __init__(self, input_dim: int, hidden: int, d: int, q_rank: int,
                 objective_mode: str = "lasso", l1_init: float = 0.01):
        super().__init__()
        self.d = d
        self.q_rank = q_rank
        self.objective_mode = objective_mode
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.p_head = nn.Linear(hidden, d)            # linear term (both modes)
        if objective_mode == "quadratic":
            self.L_head = nn.Linear(hidden, d * q_rank)
        elif objective_mode == "lasso":
            # diagonal curvature > 0 via softplus; per-dim L1 weight >= 0 via softplus
            self.q_diag_head = nn.Linear(hidden, d)
            self.l1_head = nn.Linear(hidden, d)
            self._l1_init = float(l1_init)
        else:
            raise ValueError(f"unknown objective_mode {objective_mode!r}")

    def forward(self, patch: torch.Tensor) -> dict[str, torch.Tensor]:
        """``patch: (..., input_dim)`` -> dict of objective params.

        Returns ``{L, p}`` (quadratic) or ``{q_diag, q, l1_weight}`` (lasso),
        each with leading dims matching ``patch``.
        """
        h = self.trunk(patch)
        p = self.p_head(h)                               # (..., d)
        if self.objective_mode == "quadratic":
            L = self.L_head(h).reshape(*patch.shape[:-1], self.d, self.q_rank)
            return {"L": L, "p": p}
        # lasso: diagonal curvature (strictly positive) + per-dim L1 (>= 0)
        q_diag = F.softplus(self.q_diag_head(h)) + 1e-3   # (..., d), > 0
        raw_l1 = self.l1_head(h)
        l1_weight = F.softplus(raw_l1 + self._inv_softplus(self._l1_init))
        return {"q_diag": q_diag, "q": p, "l1_weight": l1_weight}

    @staticmethod
    def _inv_softplus(y: float) -> float:
        y = max(float(y), 1e-6)
        return float(torch.log(torch.expm1(torch.tensor(y))).item())


class SharedDecoder(nn.Module):
    """Agent latent ``z_i`` -> ``num_classes`` logits **and** a scalar confidence.

    Shared across all agents. The confidence head is a sigmoid scalar in
    ``[0, 1]`` used to weight the column's vote in confidence-weighted voting.
    """

    def __init__(self, d: int, hidden: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
        self.conf_head = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(logits, conf)`` where ``conf: (..., 1)`` in ``[0, 1]``."""
        logits = self.net(z)
        conf = torch.sigmoid(self.conf_head(z))
        return logits, conf


# ---------------------------------------------------------------------------
# Restriction maps (one learned [k, d] matrix per edge direction)
# ---------------------------------------------------------------------------

class RestrictionMaps(nn.Module):
    """Per-edge learned sheaf restriction maps ``F_ij in R^{k x d}``.

    Stored as ``[E, 2, k, d]`` with ``F[e, 0] = F_{u->v}`` and
    ``F[e, 1] = F_{v->u}`` for undirected edge ``e = (u, v)``.
    """

    def __init__(self, num_edges: int, d: int, k: int, init_scale: float = 0.1):
        super().__init__()
        self.F = nn.Parameter(torch.randn(num_edges, 2, k, d) * init_scale)

    def forward(self) -> torch.Tensor:
        return self.F


# ---------------------------------------------------------------------------
# Convergence log (diagnostic)
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceLog:
    """Per-round record of the solver's convergence (diagnostic, for viz).

    * ``avg_disagreement``: ``(K,)`` mean ``||u_i||^2`` per agent per round
    * ``edge_energy``: ``(K, E)`` per-edge prediction-error energy per round
    * ``edge_indices``: ``(E, 2)`` for plotting
    * ``t_used``: ``(K,)`` inner diffusion steps actually run per round (== ``T``)
    """

    avg_disagreement: torch.Tensor
    edge_energy: torch.Tensor
    edge_indices: torch.Tensor
    t_used: torch.Tensor


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class AdaptiveConsensusModel(nn.Module):
    """End-to-end lean sheaf-ADMM model for MNIST classification."""

    def __init__(self, cfg: ModelConfig, patch_dim: int,
                 edge_indices: torch.Tensor, n_agents: int):
        super().__init__()
        self.cfg = cfg
        self.patch_dim = patch_dim
        self.n_agents = n_agents
        self.num_edges = int(edge_indices.shape[0])
        self.register_buffer("edge_indices", edge_indices)

        # reference-frame ("where") code, location signal for each column
        self.d_pos = cfg.d_pos
        pos_code = sinusoidal_pos_code(n_agents, cfg.d_pos, dtype=torch.float32)
        self.register_buffer("pos_code", pos_code)        # (N, d_pos)
        enc_input_dim = patch_dim + cfg.d_pos

        self.encoder = SharedEncoder(enc_input_dim, cfg.enc_hidden, cfg.d, cfg.q_rank,
                                    objective_mode=cfg.objective_mode,
                                    l1_init=cfg.l1_weight)
        self.restriction = RestrictionMaps(self.num_edges, cfg.d, cfg.k)
        self.decoder = SharedDecoder(cfg.d, cfg.dec_hidden, cfg.num_classes)

        # learnable ADMM scalars (kept positive via softplus)
        self.rho_log = nn.Parameter(torch.tensor(cfg.rho_init - 1.0))
        self.lr_z_log = nn.Parameter(torch.tensor(cfg.lr_z_init - 1.0))

    @property
    def rho(self) -> torch.Tensor:
        return F.softplus(self.rho_log)

    @property
    def lr_z(self) -> torch.Tensor:
        return F.softplus(self.lr_z_log) * 0.1

    # ------------------------------------------------------------------
    # x-update (local solve)
    # ------------------------------------------------------------------

    @staticmethod
    def _local_solve_quadratic(z, u, L, p, rho) -> torch.Tensor:
        """x = (Q + rho I)^-1 (rho (z - u) - p),  Q = L L^T + eps I (legacy).

        Solved via the Woodbury identity using the low-rank factor ``L``:
        ``M = rho I + L L^T`` and
        ``M^-1 = (1/rho)(I - L (rho I + L^T L)^-1 L^T)``.
        ``L: (..., d, r)``, ``z,u,p: (..., d)``.
        """
        _d, r = L.shape[-2], L.shape[-1]
        eye_r = torch.eye(r, dtype=L.dtype, device=L.device)

        rhs = rho * (z - u) - p                               # (..., d)
        LtL = torch.matmul(L.transpose(-1, -2), L)          # (..., r, r) = L^T L
        inner = torch.linalg.inv(rho * eye_r + LtL)            # (..., r, r)
        Lt_rhs = torch.einsum("...dr,...d->...r", L, rhs)      # (..., r)
        prod = (inner @ Lt_rhs.unsqueeze(-1)).squeeze(-1)      # (..., r)
        corr = torch.einsum("...dr,...r->...d", L, prod)       # (..., d)
        return (rhs - corr) / rho

    @staticmethod
    def _local_solve_lasso(z, u, q_diag, q, l1_weight, rho, l2=0.0) -> torch.Tensor:
        """Diagonal-prox x-update: soft-thresholding on a diagonal quadratic + L1.

            v = z - u
            a = q_diag + l2 + rho
            t = (rho * v - q) / a
            x = soft_threshold(t, l1_weight / a)
        Exact, no inner iterations. ``q_diag`` is strictly positive; ``l1_weight``
        is >= 0. All tensors broadcast on the leading dims.
        """
        v = z - u
        a = q_diag + l2 + rho
        t = (rho * v - q) / a
        thr = l1_weight / a
        return torch.sign(t) * torch.clamp(torch.abs(t) - thr, min=0.0)

    def _x_update(self, z, u, enc_out, rho) -> torch.Tensor:
        if self.cfg.objective_mode == "quadratic":
            return self._local_solve_quadratic(z, u, enc_out["L"], enc_out["p"], rho)
        return self._local_solve_lasso(z, u, enc_out["q_diag"], enc_out["q"],
                                       enc_out["l1_weight"], rho, self.cfg.l2_weight)

    # ------------------------------------------------------------------
    # z-update (consensus) — the communication step
    # ------------------------------------------------------------------

    def _laplacian_apply(self, x, Fm) -> torch.Tensor:
        """``L_s x`` for the sheaf (graph) Laplacian ``L_s = B^T diag(F) B``.

        For each undirected edge (u, v)::
            err = F_u x_u - F_v x_v
            (L_s x)_u += F_u^T err ;  (L_s x)_v -= F_v^T err
        ``x: (N, B, d)`` -> ``(N, B, d)``.
        """
        u_idx, v_idx = self.edge_indices[:, 0], self.edge_indices[:, 1]
        proj_u = torch.einsum("ekd,ebd->ebk", Fm[:, 0], x[u_idx])
        proj_v = torch.einsum("ekd,ebd->ebk", Fm[:, 1], x[v_idx])
        err = proj_u - proj_v                             # (E, B, k)
        grad = torch.zeros_like(x)
        grad.index_add_(0, u_idx, torch.einsum("ekd,ebk->ebd", Fm[:, 0], err))
        grad.index_add_(0, v_idx, torch.einsum("ekd,ebk->ebd", Fm[:, 1], -err))
        return grad

    def _cg_project(self, z_target, z_prev, Fm) -> torch.Tensor:
        """Unrolled conjugate-gradient z-update with hard ``ker(F)`` projection
        (``z_mode=project``).

        Solve ``(L_s + eps I) w = L_s z_target`` for ``w`` with ``T`` CG steps,
        warm-started at ``w0 = z_target - z_prev``, then return
        ``z = z_target - w``. The tiny Tikhonov ``eps`` regularizes the singular
        ``L_s`` (its kernel is the consensus subspace), so the result has
        ``F z -> 0`` (hard consensus). CG is batched: inner products reduce over
        agents and channel dims, keeping the batch axis.
        """
        eps = self.cfg.tikhonov_eps
        T = self.cfg.T

        def matvec(w):
            return self._laplacian_apply(w, Fm) + eps * w

        b = self._laplacian_apply(z_target, Fm)
        w0 = torch.zeros_like(z_target)   # zero-init: exact projection
                                          # (warm-start w0=z_target-z_prev
                                          # collapses to z=0 when z_prev=0;
                                          # zero-init is the range-space path)

        def bdot(a, c):
            return (a * c).sum(dim=(0, 2))                 # (B,)

        r = b - matvec(w0)
        p = r
        rTr = bdot(r, r)
        x = w0
        for _ in range(T):
            Ap = matvec(p)
            pTAp = bdot(p, Ap)
            alpha = rTr / (pTAp + 1e-8)
            x = x + alpha[None, :, None] * p
            r = r - alpha[None, :, None] * Ap
            rTr_new = bdot(r, r)
            beta = rTr_new / (rTr + 1e-8)
            p = r + beta[None, :, None] * p
            rTr = rTr_new
        return z_target - x

    def _diffusion_step(self, z, Fm):
        """One synchronous sheaf-diffusion step (legacy GD z-update).

        z: (N, B, d); Fm: (E, 2, k, d). Implements (per edge (i,j))::

            err  = F_ij z_i - F_ji z_j
            z_i -= eta * F_ij^T err
            z_j += eta * F_ji^T (-err)
        """
        u_idx, v_idx = self.edge_indices[:, 0], self.edge_indices[:, 1]
        z_u, z_v = z[u_idx], z[v_idx]
        F_u, F_v = Fm[:, 0], Fm[:, 1]
        proj_u = torch.einsum("ekd,ebd->ebk", F_u, z_u)   # (E, B, k)
        proj_v = torch.einsum("ekd,ebd->ebk", F_v, z_v)
        err = proj_u - proj_v                             # (E, B, k) sheaf error
        pull_u = torch.einsum("ekd,ebk->ebd", F_u, err)
        pull_v = torch.einsum("ekd,ebk->ebd", F_v, -err)
        grad = torch.zeros_like(z)
        grad.index_add_(0, u_idx, pull_u)
        grad.index_add_(0, v_idx, pull_v)
        return z - self.lr_z * grad

    def _edge_energy(self, z, Fm) -> torch.Tensor:
        """Per-edge sheaf prediction-error energy ``||F_ij z_i - F_ji z_j||^2``."""
        u_idx, v_idx = self.edge_indices[:, 0], self.edge_indices[:, 1]
        proj_u = torch.einsum("ekd,ebd->ebk", Fm[:, 0], z[u_idx])
        proj_v = torch.einsum("ekd,ebd->ebk", Fm[:, 1], z[v_idx])
        return (proj_u - proj_v).pow(2).sum(-1)           # (E, B)

    # ------------------------------------------------------------------
    # one ADMM round
    # ------------------------------------------------------------------

    def _run_round(self, x, z, u_states, enc_out):
        """One ADMM round. Returns updated (x, z, u, avg_disagree, energy, t_used)."""
        rho = self.rho
        # Step A: local solve (diagonal-prox for lasso, Woodbury for quadratic)
        x = self._x_update(z, u_states, enc_out, rho)

        # Step B: z-update (consensus). cg_project = hard ker(F) projection;
        # gd = legacy soft gradient-descent diffusion.
        z_target = x + u_states                            # ADMM z-target
        Fm = self.restriction()
        if self.cfg.z_solver == "cg_project":
            z_new = self._cg_project(z_target, z, Fm)      # single CG solve, T iters
            t_used = self.cfg.T
        else:
            z_new = z_target
            for _ in range(self.cfg.T):
                z_new = self._diffusion_step(z_new, Fm)
            t_used = self.cfg.T

        # Step C: dual update
        u_new = u_states + (x - z_new)

        with torch.no_grad():
            avg_disagree = u_new.pow(2).sum(-1).mean()     # scalar
            energy = self._edge_energy(z_new, Fm)          # (E, B)

        return x, z_new, u_new, avg_disagree, energy, t_used

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, patches: torch.Tensor, *, K: int | None = None
                ) -> tuple[torch.Tensor, dict]:
        """Run the ADMM solver for ``K`` rounds (default ``cfg.K``).

        ``patches: (N, B, patch_dim)``. Pass ``K=cfg.K_eval`` at inference to
        run more rounds than were used during training. Returns ``(logits, aux)``
        with the convergence log and per-round agent logits.
        """
        N, B = patches.shape[0], patches.shape[1]
        d = self.cfg.d
        device = patches.device
        K = self.cfg.K if K is None else K

        # encode -> local objective params {L, p}
        # append the reference-frame ("where") code to each agent's input
        if self.d_pos > 0 and self.pos_code.numel() > 0:
            pos = self.pos_code.unsqueeze(1).expand(-1, B, -1)   # (N, B, d_pos)
            enc_in = torch.cat([patches, pos], dim=-1)           # (N, B, patch_dim+d_pos)
        else:
            enc_in = patches
        enc_out = self.encoder(enc_in)                   # dict of objective params

        # init states
        x = torch.zeros(N, B, d, device=device)
        z = torch.zeros(N, B, d, device=device)
        u_states = torch.zeros(N, B, d, device=device)

        per_round_logits = []
        per_round_disagree = []
        per_round_energy = []
        per_round_t_used = []

        for _k in range(K):
            x, z, u_states, avg_disagree, energy, t_used = self._run_round(
                x, z, u_states, enc_out)
            # decode the configured readout variable: 'x' (local proposal)
            # or 'z' (consensus, legacy). With hard ker(F) projection the digit
            # signal lives in x; reading z loses per-agent info.
            readout = x if self.cfg.dec_readout == "x" else z
            logits_k, conf_k = self.decoder(readout)       # (N, B, C), (N, B, 1)
            per_round_logits.append(logits_k)
            per_round_disagree.append(avg_disagree.detach())
            per_round_energy.append(energy.detach())
            per_round_t_used.append(t_used)

        # final prediction: last round's per-agent vote, fused across columns.
        last_logits = per_round_logits[-1]                # (N, B, C)
        probs = F.softmax(last_logits, dim=-1)            # (N, B, C)
        if self.cfg.confidence_weighted:
            # confidence-weighted voting: weight each column by its learned
            # confidence, normalized across the population.
            w = conf_k.squeeze(-1) + 1e-6                 # (N, B)
            w = w / w.sum(0, keepdim=True)                # normalize over agents
            fused = (probs * w.unsqueeze(-1)).sum(0)      # (B, C)
        else:
            fused = probs.mean(0)                         # uniform vote (legacy)
        logits_out = torch.log(fused.clamp_min(1e-8))

        # differentiable disagreement signal for the loss. Computed outside the
        # no_grad block so grads flow.
        Fm_final = self.restriction()
        edge_energy_final = self._edge_energy(z, Fm_final)        # (E, B)
        disagreement_final = u_states.pow(2).sum(-1).mean()      # scalar

        conv_log = ConvergenceLog(
            avg_disagreement=torch.stack(per_round_disagree, dim=0),  # (K,)
            edge_energy=torch.stack(per_round_energy, dim=0),         # (K, E, B)
            edge_indices=self.edge_indices,
            t_used=torch.tensor(per_round_t_used, dtype=torch.float32),  # (K,)
        )

        aux = {
            "conv_log": conv_log,
            "per_round_logits": torch.stack(per_round_logits, dim=0),  # (K, N, B, C)
            "n_rounds": K,
            "mean_t_used": float(conv_log.t_used.mean()),  # avg inner steps/round
            # differentiable (grad-flowing) disagreement signals for the loss:
            "edge_energy_final": edge_energy_final,     # (E, B)
            "disagreement_final": disagreement_final,   # scalar
        }
        return logits_out, aux

    def predict(self, patches: torch.Tensor) -> torch.Tensor:
        return self.forward(patches, K=self.cfg.K_eval)[0]
