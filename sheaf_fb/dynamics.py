"""Forward-Backward dynamics for the Sheaf-FB network (the coordination core).

Each agent ``i`` holds a single state vector ``x_i in R^d``. The total energy is

    E(x) = sum_i 0.5 * ||x_i - theta_i||^2
         + sum_{edges (i,j)} 0.5 * rho * ||F_ij x_i - F_ji x_j||^2

The first sum is the **local objective** (the agent's state should match the
target its encoder produced from its local patch). The second sum is the
**sheaf consensus** term (neighboring agents must agree on a learned ``c``-dim
projection of their states).

The forward dynamics are **Forward-Backward splitting** (proximal gradient) on
``E``: each of the ``K`` rounds performs, for every agent in parallel,

  STEP 1 (forward / gradient step on the local objective, which is smooth):
      x_i <- (1 - eta) * x_i + eta * theta_i

  STEP 2 (backward / proximal step on the sheaf consensus, which is treated as
      the non-smooth-ish part whose gradient is the sheaf pull):
      for each neighbor j:
          error_ij = F_ij x_i - F_ji x_j
          x_i <- x_i - eta * rho * F_ij^T error_ij

After ``K`` rounds the system reaches an equilibrium. With a nudge, the energy
becomes ``E(x) + beta * Loss`` and an extra term ``- eta * beta * dL/dx_i`` is
folded into the forward step; the rest of the dynamics are identical.

Everything operates on batched agent states ``x`` of shape ``[N, B, d]``; edges
are ``[E, 2]``; per-edge sheaf maps are ``[E, 2, c, d]`` with
``F[e, 0] = F_{u->v}`` and ``F[e, 1] = F_{v->u}`` for edge ``e = (u, v)``.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Sheaf consensus operators
# ---------------------------------------------------------------------------


def edge_residuals(
    x: torch.Tensor, edge_indices: torch.Tensor, F_uv: torch.Tensor,
) -> torch.Tensor:
    """Sheaf edge residuals ``F_ij x_i - F_ji x_j`` for every edge.

    Returns ``[E, B, c]``.
    """
    u, v = edge_indices[:, 0], edge_indices[:, 1]
    x_u = x[u]  # [E, B, d]
    x_v = x[v]  # [E, B, d]
    F_u = F_uv[:, 0]  # [E, c, d]  (map at endpoint u, projecting u -> channel)
    F_v = F_uv[:, 1]  # [E, c, d]  (map at endpoint v, projecting v -> channel)
    proj_u = torch.einsum("ecd,ebd->ebc", F_u, x_u)
    proj_v = torch.einsum("ecd,ebd->ebc", F_v, x_v)
    return proj_u - proj_v


def consensus_grad(
    x: torch.Tensor, edge_indices: torch.Tensor, F_uv: torch.Tensor, rho: float,
) -> torch.Tensor:
    """Gradient of the sheaf consensus term w.r.t. each agent's state.

    ``d/dx_i [ 0.5 * rho * sum_j ||F_ij x_i - F_ji x_j||^2 ]`` accumulated over
    all edges incident to ``i``. Returns ``[N, B, d]``.
    """
    u, v = edge_indices[:, 0], edge_indices[:, 1]
    err = edge_residuals(x, edge_indices, F_uv)  # [E, B, c]
    F_u = F_uv[:, 0]  # [E, c, d]
    F_v = F_uv[:, 1]  # [E, c, d]
    # adjoint of the projection at each endpoint: F^T err  -> [E, B, d]
    grad_u = torch.einsum("ecd,ebc->ebd", F_u, err)
    grad_v = torch.einsum("ecd,ebc->ebd", F_v, err)
    out = torch.zeros_like(x)
    out.index_add_(0, u, rho * grad_u)
    # d/dx_v of 0.5*rho*||F_uv x_u - F_vu x_v||^2 is -rho * F_vu^T err
    out.index_add_(0, v, -rho * grad_v)
    return out


def local_objective_grad(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Gradient of ``0.5 * ||x_i - theta_i||^2`` w.r.t. ``x_i`` -> ``x_i - theta_i``."""
    return x - theta


def energy(
    x: torch.Tensor, theta: torch.Tensor, edge_indices: torch.Tensor,
    F_uv: torch.Tensor, rho: float,
) -> torch.Tensor:
    """Total energy ``E(x)`` summed over all agents and edges (a scalar)."""
    local = 0.5 * torch.sum((x - theta) ** 2)
    err = edge_residuals(x, edge_indices, F_uv)
    sheaf = 0.5 * rho * torch.sum(err ** 2)
    return local + sheaf


# ---------------------------------------------------------------------------
# Forward-Backward dynamics
# ---------------------------------------------------------------------------


def _nudge_loss(
    x: torch.Tensor, decoder, labels: torch.Tensor, loss_type: str,
    per_agent: bool,
) -> torch.Tensor:
    """Compute the nudging loss at state ``x`` (graph enabled w.r.t. ``x``).

    * ``per_agent=True``  (Fix 1): ``L = sum_i loss(softmax(decoder(x_i)), y)``.
      Each agent gets a full-strength learning signal — no 1/N dilution.
    * ``per_agent=False``: ``L = loss(mean_i softmax(decoder(x_i)), y)``.
      The classic diluted version.
    """
    N, B, d = x.shape
    logits = decoder(x.reshape(N * B, d)).reshape(N, B, -1)  # [N, B, C]
    probs = torch.softmax(logits, dim=-1)
    if per_agent:
        if loss_type == "ce":
            lbl = labels.unsqueeze(0).expand(N, B)  # [N, B]
            return torch.nn.functional.cross_entropy(
                probs.reshape(N * B, -1), lbl.reshape(N * B), reduction="sum")
        y = torch.nn.functional.one_hot(labels, probs.shape[-1]).float()
        return 0.5 * torch.sum((probs - y.unsqueeze(0)) ** 2)
    p_global = probs.mean(0)  # [B, C]
    if loss_type == "ce":
        return torch.nn.functional.cross_entropy(p_global, labels)
    y = torch.nn.functional.one_hot(labels, probs.shape[-1]).float()
    return 0.5 * torch.sum((p_global - y) ** 2)


def _nudge_grad(
    x: torch.Tensor, decoder, num_agents: int, labels: torch.Tensor, loss_type: str,
    per_agent: bool = True,
) -> torch.Tensor:
    """Gradient of the nudging loss w.r.t. each agent's state.

    Computed with a local autodiff call (the graph does not span the K rounds),
    so this is not BPTT — it is a single-step Jacobian-transpose product.
    Returns ``[N, B, d]`` (detached).
    """
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        loss = _nudge_loss(x, decoder, labels, loss_type, per_agent)
        g = torch.autograd.grad(loss, x)[0]
    return g.detach()


def forward_backward_bptt(
    theta: torch.Tensor,
    edge_indices: torch.Tensor,
    F_uv: torch.Tensor,
    *,
    eta: float,
    rho: float,
    num_iters: int,
    warm_start: bool = True,
    grad_window: int | None = None,
) -> torch.Tensor:
    """Run Forward-Backward splitting with an autograd graph (BPTT).

    Identical dynamics to :func:`forward_backward` (free phase, no nudge), but
    the graph is **kept** so that gradients can be backpropagated through the
    trajectory (Backpropagation Through Time).

    **Gradient windowing** (``grad_window``): only the last ``grad_window``
    rounds build an autograd graph. The first ``num_iters - grad_window`` rounds
    are run under ``no_grad`` and the state is detached before the graph-building
    phase begins. This lets the system settle to near-equilibrium cheaply (no
    memory cost) and then backpropagates only through the final rounds near the
    equilibrium — saving memory while still training the dynamics. Set
    ``grad_window=None`` or ``grad_window=num_iters`` to backprop through all
    rounds.

    Returns the equilibrium state ``[N, B, d]`` (still in the autograd graph if
    ``grad_window > 0``).
    """
    if grad_window is None or grad_window > num_iters:
        grad_window = num_iters
    n_detached = num_iters - grad_window

    x = theta.clone() if warm_start else torch.zeros_like(theta)
    x_cap = 10.0

    # Phase 1: detached settling rounds (no graph, no memory).
    with torch.no_grad():
        for _ in range(n_detached):
            x = x - eta * local_objective_grad(x, theta)
            x = x - eta * consensus_grad(x, edge_indices, F_uv, rho)
            x = torch.clamp(x, -x_cap, x_cap)
        x = x.detach()

    # Phase 2: graph-building rounds (backprop through these).
    for _ in range(grad_window):
        x = x - eta * local_objective_grad(x, theta)
        x = x - eta * consensus_grad(x, edge_indices, F_uv, rho)
        x = torch.clamp(x, -x_cap, x_cap)

    return x  # NOT detached — graph retained for BPTT


def _energy_grad_norm(
    x: torch.Tensor, theta: torch.Tensor, edge_indices: torch.Tensor,
    F_uv: torch.Tensor, rho: float,
) -> torch.Tensor:
    """||dE/dx|| at state x — the convergence criterion."""
    g = local_objective_grad(x, theta) + consensus_grad(x, edge_indices, F_uv, rho)
    return torch.linalg.vector_norm(g)


def forward_backward(
    theta: torch.Tensor,
    edge_indices: torch.Tensor,
    F_uv: torch.Tensor,
    *,
    eta: float,
    rho: float,
    num_iters: int,
    decoder=None,
    labels: torch.Tensor | None = None,
    beta: float = 0.0,
    loss_type: str = "ce",
    warm_start: bool = True,
    x_init: torch.Tensor | None = None,
    converge_tol: float = 0.0,
    per_agent_loss: bool = True,
) -> torch.Tensor:
    """Run Forward-Backward splitting on ``E`` (+ nudge) to equilibrium.

    Returns the equilibrium state ``[N, B, d]`` (detached from any graph).

    **Adaptive stopping** (Fix 4): if ``converge_tol > 0``, the dynamics stop as
    soon as ``||dE/dx|| < converge_tol`` (or after ``num_iters`` rounds). This
    ensures the recorded state is a true equilibrium, eliminating the spurious
    energy-growth term from non-convergence.

    **Warm-start** (Fix 3): if ``x_init`` is provided, the dynamics start from
    that state instead of theta/zeros. The nudged phases pass ``x_free`` here so
    they only need to track the small perturbation, not reconverge from scratch.

    A nudging term ``beta * Loss`` is added to (or, for negative ``beta``,
    subtracted from) the energy: the forward step additionally applies
    ``- eta * beta * dL/dx_i``. With ``beta > 0`` this pulls the state toward
    lower loss (the +β phase); with ``beta < 0`` it pushes away (the −β phase
    of symmetric EP). ``beta == 0`` is the free phase. The nudging gradient is
    computed fresh each round via a single local autodiff call, so no graph is
    built across the ``K`` rounds (this is Equilibrium Propagation, not BPTT).
    """
    if x_init is not None:
        x = x_init.clone()
    else:
        x = theta.clone() if warm_start else torch.zeros_like(theta)
    nudged = beta != 0 and decoder is not None and labels is not None
    N = theta.shape[0]
    x_cap = 10.0

    for _ in range(num_iters):
        # STEP 1 - forward (gradient step on the smooth local objective).
        x = x - eta * local_objective_grad(x, theta)
        if nudged:
            x = x - eta * beta * _nudge_grad(
                x, decoder, N, labels, loss_type, per_agent_loss)

        # STEP 2 - backward (proximal step on the sheaf consensus term).
        x = x - eta * consensus_grad(x, edge_indices, F_uv, rho)
        x = torch.clamp(x, -x_cap, x_cap)

        # Adaptive convergence check (only in the free phase; nudged phases
        # have an extra loss-gradient term in dE/dx that doesn't converge to 0
        # in the same way, so we rely on warm-start + fewer rounds there).
        if converge_tol > 0 and not nudged:
            if float(_energy_grad_norm(x, theta, edge_indices, F_uv, rho)) < converge_tol:
                break

    return x.detach()
