"""The full Sheaf-Forward-Backward model for MNIST.

Components
----------
1. a **shared MLP encoder** maps each agent's 4x4 patch (16 pixels) to a local
   target ``theta_i in R^d`` — the parameter of that agent's local objective
   ``0.5 * ||x_i - theta_i||^2``;
2. **per-edge learned sheaf restriction maps** ``F_ij in R^{c x d}`` (one per
   edge direction) project each agent's private state onto a shared ``c``-dim
   communication channel with a neighbor;
3. the **Forward-Backward dynamics** (:mod:`sheaf_fb.dynamics`) settle the
   agents to an equilibrium by alternating a gradient step toward the local
   target and a proximal step toward sheaf consensus with neighbors;
4. a **shared MLP decoder** maps each agent's equilibrium state to digit logits;
   the global prediction is the mean of the per-agent softmax.

Training uses **Equilibrium Propagation** (see :meth:`SheafFBModel.ep_step`):
a free phase and a nudged phase settle to two equilibria ``x_free`` and
``x_nudged``; the local learning signal is ``delta_i = (1/beta)(x_free_i -
x_nudged_i)``. Parameter gradients are estimated from the two equilibria only —
no backpropagation through the K rounds, no storing of intermediate rounds.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .dynamics import (
    consensus_grad,
    edge_residuals,
    energy,
    forward_backward,
    forward_backward_bptt,
)

# ---------------------------------------------------------------------------
# Shared encoder / decoder
# ---------------------------------------------------------------------------


class SharedEncoder(nn.Module):
    """Shared MLP encoder: ``patch (16 pixels) -> theta_i in R^d``.

    ``Linear(input_dim, hidden) -> ReLU -> Linear(hidden, d)``. ReLU is applied
    on the hidden layer only so ``theta_i`` may take either sign.
    """

    def __init__(self, input_dim: int, hidden_dim: int, d: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, d, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        return self.fc2(torch.relu(self.fc1(x)))


class SharedDecoder(nn.Module):
    """Shared MLP decoder: ``x_i (d) -> logits (num_classes)``.

    ``Linear(d, hidden) -> ReLU -> Linear(hidden, num_classes)``. Softmax is
    applied by the task / fusion layer, not here.
    """

    def __init__(self, d: int, hidden_dim: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Sheaf restriction maps
# ---------------------------------------------------------------------------


class SheafMaps(nn.Module):
    """Learned per-direction sheaf restriction maps ``F_ij in R^{c x d}``.

    Two sharing modes:

    * ``"per_edge"`` (spec default) — every edge direction ``(i->j)`` gets its
      own learned matrix. Stored as ``[E, 2, c, d]`` with ``F[e, 0] = F_{u->v}``
      and ``F[e, 1] = F_{v->u}`` for undirected edge ``e = (u, v)``.
    * ``"directional"`` — 4 shared base maps (one per compass direction for a
      4-connected grid), gathered per edge. More parameter-efficient and
      shares spatial structure across the grid.
    """

    def __init__(
        self, num_edges: int, d: int, c: int, sharing: str,
        edge_indices: torch.Tensor, node_positions: torch.Tensor,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.sharing = sharing
        self.d = d
        self.c = c
        self.edge_indices = edge_indices
        self.node_positions = node_positions

        if sharing == "per_edge":
            maps = torch.randn(num_edges, 2, c, d) * init_scale
            self.F = nn.Parameter(maps)
        elif sharing == "directional":
            maps = torch.randn(4, c, d) * init_scale
            self.F = nn.Parameter(maps)
        else:
            raise ValueError(f"unknown sheaf_sharing {sharing!r}")
        self.max_norm = 0.0  # set by the parent model after construction

    @staticmethod
    def _build_direction_index(edge_indices: torch.Tensor, node_positions: torch.Tensor) -> torch.Tensor:
        """Map each edge to a direction index for 4-way adjacency.

        0=N, 1=E, 2=S, 3=W (based on (dy, dx) from u to v).
        """
        u, v = edge_indices[:, 0], edge_indices[:, 1]
        dy = node_positions[v, 0] - node_positions[u, 0]
        dx = node_positions[v, 1] - node_positions[u, 1]
        # dy<0 -> N (0); dx>0 -> E (1); dy>0 -> S (2); dx<0 -> W (3)
        return torch.where(dy < 0, 0, torch.where(dx > 0, 1, torch.where(dy > 0, 2, 3)))

    def forward(self) -> torch.Tensor:
        """Return the effective per-edge maps ``[E, 2, c, d]``."""
        if self.sharing == "per_edge":
            return self.F
        # directional: gather the 4 base maps per edge and per endpoint direction
        u, v = self.edge_indices[:, 0], self.edge_indices[:, 1]
        dy = self.node_positions[v, 0] - self.node_positions[u, 0]
        dx = self.node_positions[v, 1] - self.node_positions[u, 1]
        dir_uv = torch.where(dy < 0, 0, torch.where(dx > 0, 1, torch.where(dy > 0, 2, 3)))
        dir_vu = torch.where(-dy < 0, 0, torch.where(-dx > 0, 1, torch.where(-dy > 0, 2, 3)))
        return torch.stack([self.F[dir_uv], self.F[dir_vu]], dim=1)

    @torch.no_grad()
    def project_norm(self, max_norm: float) -> None:
        """Project each map onto the Frobenius-norm ball of radius ``max_norm``.

        This is a projected-gradient safety step: it keeps the sheaf Laplacian's
        spectral radius bounded so the Forward-Backward dynamics stay stable
        (the iteration converges while ``eta * rho * max_degree * ||F||^2`` is
        below ~2). A no-op when ``max_norm <= 0``.
        """
        if max_norm <= 0:
            return
        F = self.F
        F_view = F.reshape(-1, self.c * self.d)
        norms = torch.linalg.vector_norm(F_view, dim=1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(max_norm / norms, max=1.0)
        F_view.mul_(scale)  # in-place via the view


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class SheafFBModel(nn.Module):
    """End-to-end Sheaf-Forward-Backward model for MNIST classification."""

    def __init__(
        self, cfg: ModelConfig, patch_dim: int, edge_indices: torch.Tensor,
        node_positions: torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        self.patch_dim = patch_dim
        self.edge_indices = edge_indices
        self.node_positions = node_positions
        self.num_edges = int(edge_indices.shape[0])

        self.encoder = SharedEncoder(patch_dim, cfg.enc_hidden_dim, cfg.d, bias=cfg.enc_bias)
        self.sheaf = SheafMaps(
            self.num_edges, cfg.d, cfg.c, cfg.sheaf_sharing,
            edge_indices, node_positions, init_scale=cfg.sheaf_init_scale)
        self.sheaf.max_norm = cfg.sheaf_max_norm
        self.decoder = SharedDecoder(cfg.d, cfg.dec_hidden_dim, cfg.num_classes, bias=cfg.dec_bias)

    # -- encoder / sheaf / decoder helpers ----------------------------------

    def _encode(self, patches: torch.Tensor) -> torch.Tensor:
        """``patches`` ``[N, B, ...]`` -> per-agent targets ``theta`` ``[N, B, d]``."""
        N, B = patches.shape[:2]
        flat = patches.reshape(N * B, *patches.shape[2:])
        theta = self.encoder(flat)  # [N*B, d]
        if self.cfg.theta_clip > 0:
            theta = theta.clamp(-self.cfg.theta_clip, self.cfg.theta_clip)
        return theta.reshape(N, B, self.cfg.d)

    def _sheaf_maps(self) -> torch.Tensor:
        return self.sheaf()

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` ``[N, B, d]`` -> per-agent logits ``[N, B, C]``."""
        N, B = x.shape[:2]
        logits = self.decoder(x.reshape(N * B, self.cfg.d))
        return logits.reshape(N, B, -1)

    # -- forward / inference -------------------------------------------------

    def settle(
        self, patches: torch.Tensor, *, nudged: bool, labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run Forward-Backward dynamics and return the equilibrium ``x`` ``[N, B, d]``.

        ``nudged=True`` uses ``beta = +cfg.beta`` (the +β phase); ``nudged=False``
        uses ``beta = 0`` (the free phase). For the −β phase use
        :meth:`settle_signed`.
        """
        return self.settle_signed(
            patches, labels=labels, beta=(self.cfg.beta if nudged else 0.0))

    def settle_signed(
        self, patches: torch.Tensor, *, labels: torch.Tensor | None, beta: float,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Settle with an explicit signed nudging strength ``beta``.

        ``beta = 0``  -> free phase; ``beta > 0`` -> +β phase; ``beta < 0`` ->
        −β phase (symmetric EP).

        ``x_init``: optional warm-start state (Fix 3). Nudged phases pass
        ``x_free`` here so they start from the free equilibrium and only need
        to track the small β-perturbation.
        """
        theta = self._encode(patches)
        F_uv = self._sheaf_maps()
        is_free = beta == 0
        return forward_backward(
            theta, self.edge_indices, F_uv,
            eta=self.cfg.eta, rho=self.cfg.rho,
            num_iters=self.cfg.K if is_free else self.cfg.K_nudge,
            decoder=self.decoder if beta != 0 else None,
            labels=labels if beta != 0 else None,
            beta=beta,
            loss_type=self.cfg.loss_type,
            warm_start=self.cfg.warm_start,
            x_init=x_init,
            converge_tol=self.cfg.converge_tol,
            per_agent_loss=self.cfg.per_agent_loss,
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Inference: settle (free phase) and decode -> per-agent logits ``[N, B, C]``."""
        x = self.settle(patches, nudged=False)
        return self._decode(x)

    def predict_global(self, patches: torch.Tensor) -> torch.Tensor:
        """Inference -> global probability vector ``p_global`` ``[B, C]``."""
        logits = self.forward(patches)
        return torch.softmax(logits, dim=-1).mean(0)

    # -- BPTT training path --------------------------------------------------

    def bptt_forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Run the free-phase dynamics with an autograd graph (BPTT).

        Runs all ``K`` Forward-Backward rounds, but only the last
        ``grad_window`` rounds retain the autograd graph (the first
        ``K - grad_window`` rounds are detached to save memory). Returns
        per-agent logits ``[N, B, C]``. Use :meth:`bptt_step` for training.
        """
        theta = self._encode(patches)
        F_uv = self._sheaf_maps()
        x = forward_backward_bptt(
            theta, self.edge_indices, F_uv,
            eta=self.cfg.eta, rho=self.cfg.rho, num_iters=self.cfg.K,
            warm_start=self.cfg.warm_start, grad_window=self.cfg.grad_window,
        )
        return self._decode(x)

    def bptt_step(
        self, patches: torch.Tensor, labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """One BPTT training step: unroll K rounds, backprop through the last ``grad_window``.

        The Forward-Backward dynamics run for all ``K`` rounds, but only the
        last ``grad_window`` rounds build an autograd graph (the first
        ``K - grad_window`` are detached to let the system settle cheaply).
        The averaged global CE loss is computed at the final state and
        backpropagated through the ``grad_window``-round trajectory.
        """
        logits = self.bptt_forward(patches)  # [N, B, C], graph retained
        probs = torch.softmax(logits, dim=-1)
        p_global = probs.mean(0)  # [B, C]
        if self.cfg.loss_type == "ce":
            loss = torch.nn.functional.cross_entropy(p_global, labels)
        else:
            y = torch.nn.functional.one_hot(labels, p_global.shape[-1]).float()
            loss = 0.5 * torch.sum((p_global - y) ** 2)

        optimizer.zero_grad()
        loss.backward()
        return {"loss": float(loss.detach())}

    # -- diagnostics --------------------------------------------------------

    @torch.no_grad()
    def settle_history(
        self, patches: torch.Tensor, *, nudged: bool = False,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward-only path returning the per-round trajectory (diagnostic)."""
        theta = self._encode(patches)
        F_uv = self._sheaf_maps()
        x = theta.clone() if self.cfg.warm_start else torch.zeros_like(theta)
        nudged = nudged and labels is not None
        beta = self.cfg.beta if nudged else 0.0
        N = theta.shape[0]
        xs, local_e, sheaf_e, total_e = [], [], [], []
        for _ in range(self.cfg.K):
            x = x - self.cfg.eta * (x - theta)
            if nudged:
                from .dynamics import _nudge_grad
                x = x - self.cfg.eta * beta * _nudge_grad(
                    x, self.decoder, N, labels, self.cfg.loss_type,
                    self.cfg.per_agent_loss)
            x = x - self.cfg.eta * consensus_grad(x, self.edge_indices, F_uv, self.cfg.rho)
            xs.append(x)
            err = edge_residuals(x, self.edge_indices, F_uv)
            local_e.append(0.5 * torch.sum((x - theta) ** 2).detach())
            sheaf_e.append(0.5 * self.cfg.rho * torch.sum(err ** 2).detach())
            total_e.append((local_e[-1] + sheaf_e[-1]).detach())
        return {
            "x": torch.stack(xs, dim=0),  # [K, N, B, d]
            "local_energy": torch.stack(local_e),  # [K]
            "sheaf_energy": torch.stack(sheaf_e),  # [K]
            "total_energy": torch.stack(total_e),  # [K]
        }

    # -- Equilibrium Propagation training step ------------------------------

    @torch.no_grad()
    def _inference_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Inference loss (averaged vote) for monitoring — not used for gradients."""
        logits = self._decode(x)
        p_global = torch.softmax(logits, dim=-1).mean(0)
        if self.cfg.loss_type == "ce":
            return torch.nn.functional.cross_entropy(p_global, labels)
        y = torch.nn.functional.one_hot(labels, p_global.shape[-1]).float()
        return 0.5 * torch.sum((p_global - y) ** 2)

    def _loss_at(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Nudging loss at state x (graph enabled w.r.t. decoder weights).

        When ``per_agent_loss=True`` (Fix 1) this is the **sum** of per-agent
        losses — each agent gets a full-strength learning signal. When False it
        is the classic averaged global loss.
        """
        from .dynamics import _nudge_loss
        return _nudge_loss(
            x, self.decoder, labels, self.cfg.loss_type, self.cfg.per_agent_loss)

    def ep_step(
        self, patches: torch.Tensor, labels: torch.Tensor, optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """One Equilibrium-Propagation training step.

        Two estimator variants (``cfg.ep_variant``):

        * ``"symmetric"`` (default, Laborieux et al. 2021) — **3 phases**:
          free (β=0), positive nudge (+β), negative nudge (−β). The gradient is
          the **symmetric finite difference**, which cancels the O(β) bias of
          classic EP and leaves only an O(β²) error:

              ∇θ L ≈ (1/(2β)) · [ ∂E(x_β⁺)/∂θ − ∂E(x_β⁻)/∂θ ]

          Because the decoder/readout weights ``w`` enter only through the loss
          (not the energy), their rule is the **average** of the loss gradient
          at the two nudged states:

              ∇w L ≈ ½ · [ ∂L(x_β⁺)/∂w + ∂L(x_β⁻)/∂w ]

          Both are obtained by backpropagating the single scalar

              S = (1/(2β))·[E(x_β⁺) − E(x_β⁻)] + ½·[L(x_β⁺) + L(x_β⁻)]

          whose autograd-Jacobian is exactly the two formulas above (the energy
          term only touches encoder + sheaf maps; the loss term only touches the
          decoder). Three equilibria are stored; no graph spans the K rounds.

        * ``"one_sided"`` — classic **2-phase** EP (free + +β):

              ∇θ L ≈ (1/β)·[∂E(x_β⁺)/∂θ − ∂E(x_free)/∂θ],
              ∇w L ≈ ∂L(x_β⁺)/∂w.

          Cheaper (2 settles) but has an O(β) bias that drifts/destabilizes
          training — kept for comparison only.
        """
        beta = self.cfg.beta
        variant = self.cfg.ep_variant

        # --- Settle the required phases (no graph through the K rounds) ---
        # Fix 3: nudged phases are warm-started from x_free so they only need to
        # track the small β-perturbation, not reconverge from scratch.
        with torch.no_grad():
            x_free = self.settle(patches, nudged=False)
            if self.cfg.nudge_warm_start:
                ws = x_free
            else:
                ws = None
            if variant == "symmetric":
                x_plus = self.settle_signed(patches, labels=labels, beta=beta, x_init=ws)
                x_minus = self.settle_signed(patches, labels=labels, beta=-beta, x_init=ws)
            elif variant == "one_sided":
                x_plus = self.settle_signed(patches, labels=labels, beta=beta, x_init=ws)
            else:
                raise ValueError(f"unknown ep_variant {variant!r}")

        x_free = x_free.detach()
        x_plus = x_plus.detach()
        if variant == "symmetric":
            x_minus = x_minus.detach()

        # --- EP gradient estimate: autograd only over the energy / loss
        #     *evaluated at the equilibria*, never through the K rounds. ---
        theta = self._encode(patches)  # graph w.r.t. encoder weights
        F_uv = self._sheaf_maps()  # graph w.r.t. sheaf-map weights

        if variant == "symmetric":
            e_plus = energy(x_plus, theta, self.edge_indices, F_uv, self.cfg.rho)
            e_minus = energy(x_minus, theta, self.edge_indices, F_uv, self.cfg.rho)
            l_plus = self._loss_at(x_plus, labels)
            l_minus = self._loss_at(x_minus, labels)
            ep_scalar = (e_plus - e_minus) / (2.0 * beta) + 0.5 * (l_plus + l_minus)
            # Report the inference loss (averaged vote) for monitoring.
            loss_free = self._inference_loss(x_free.detach(), labels)
            stats = {
                "loss": float(loss_free),
                "e_plus": float(e_plus.detach()),
                "e_minus": float(e_minus.detach()),
                "delta_norm": float(torch.linalg.vector_norm(x_plus - x_minus).detach()),
            }
        else:  # one_sided
            e_free = energy(x_free, theta, self.edge_indices, F_uv, self.cfg.rho)
            e_plus = energy(x_plus, theta, self.edge_indices, F_uv, self.cfg.rho)
            l_plus = self._loss_at(x_plus, labels)
            ep_scalar = (e_plus - e_free) / beta + l_plus
            loss_free = self._inference_loss(x_free.detach(), labels)
            stats = {
                "loss": float(loss_free),
                "e_free": float(e_free.detach()),
                "e_plus": float(e_plus.detach()),
                "delta_norm": float(torch.linalg.vector_norm(x_free - x_plus).detach()),
            }

        optimizer.zero_grad()
        ep_scalar.backward()
        return stats

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def project_sheaf(self) -> None:
        """Project the sheaf maps onto their norm ball (call after optimizer.step)."""
        self.sheaf.project_norm(self.sheaf.max_norm)
