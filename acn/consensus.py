"""ACN consensus core: primal / consensus / dual + motor / path integration / primer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from acn.networks import MotorSystem


@dataclass
class ConsensusState:
    x: torch.Tensor
    z: torch.Tensor
    u: torch.Tensor
    D: torch.Tensor
    Q: torch.Tensor
    history: list[dict] | None = None
    logits: torch.Tensor | None = None
    active: torch.Tensor | None = None
    relevance_logits: torch.Tensor | None = None
    motor: torch.Tensor | None = None


def make_state(
    B: int, N: int, E: int, d: int,
    device: torch.device | str,
    dtype: torch.dtype,
    D_init: torch.Tensor | None = None,
    z_init: torch.Tensor | None = None,
) -> ConsensusState:
    if z_init is not None:
        z = z_init.to(device=device, dtype=dtype).clone()
    else:
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
    @staticmethod
    def primal(
        A: torch.Tensor, b: torch.Tensor, z: torch.Tensor, u: torch.Tensor, rho: float
    ) -> torch.Tensor:
        """x_i = (A_i + rho I)^{-1} (b_i + rho (z_i - u_i)).

        Uses Cholesky + triangular solve instead of MagMA batched LU to avoid
        illegal memory access on large batch sizes (> 8k). Chunked forward so
        autograd backprop is also chunked.
        """
        B, N, d, _ = A.shape
        I = torch.eye(d, device=A.device, dtype=A.dtype)
        M = A + rho * I
        rhs = b + rho * (z - u)

        # Flatten B,N so we have a single batch of B*N matrices
        M_f = M.reshape(B * N, d, d)
        rhs_f = rhs.reshape(B * N, d, 1)
        chunk_size = 2048  # MagMA safe limit for this kernel
        B_total = M_f.shape[0]

        if B_total <= chunk_size:
            L = torch.linalg.cholesky(M_f)
            y = torch.linalg.solve_triangular(L, rhs_f, upper=False)
            x = torch.linalg.solve_triangular(
                L.transpose(-1, -2), y, upper=True
            )
            return x.squeeze(-1).reshape(B, N, d)

        x_parts = []
        for i in range(0, B_total, chunk_size):
            end = min(i + chunk_size, B_total)
            L_chunk = torch.linalg.cholesky(M_f[i:end])
            y_chunk = torch.linalg.solve_triangular(L_chunk, rhs_f[i:end], upper=False)
            x_chunk = torch.linalg.solve_triangular(
                L_chunk.transpose(-1, -2), y_chunk, upper=True
            )
            x_parts.append(x_chunk)
        x = torch.cat(x_parts, dim=0)
        return x.squeeze(-1).reshape(B, N, d)

    @staticmethod
    def flux(
        z: torch.Tensor, r_ij: torch.Tensor, r_ji: torch.Tensor,
        ei: torch.Tensor, ej: torch.Tensor,
    ) -> torch.Tensor:
        z_ei = z[:, ei]
        z_ej = z[:, ej]
        if r_ij.dim() == 2:
            r_ij = r_ij.unsqueeze(0)
            r_ji = r_ji.unsqueeze(0)
        diff = r_ij * z_ei - r_ji * z_ej
        return diff

    @staticmethod
    def consensus_step(
        z: torch.Tensor, x: torch.Tensor, u: torch.Tensor,
        r_ij: torch.Tensor, r_ji: torch.Tensor,
        D: torch.Tensor, ei: torch.Tensor, ej: torch.Tensor,
        eta_z: float, rho: float,
    ) -> torch.Tensor:
        diff = ACNCore.flux(z, r_ij, r_ji, ei, ej)
        if r_ij.dim() == 2:
            r_ij_b = r_ij.unsqueeze(0)
            r_ji_b = r_ji.unsqueeze(0)
        else:
            r_ij_b, r_ji_b = r_ij, r_ji
        grad_i = D.unsqueeze(-1) * (r_ij_b * diff)
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
        edges: torch.Tensor, D_init: torch.Tensor,
        num_nodes: int,
        *,
        rounds: int,
        diffusion_steps: int,
        rho: float, eta_z: float,
        record: bool = False,
        detach_after: int | None = None,
        active: torch.Tensor,
        motor_system: MotorSystem | None = None,
        z_init: torch.Tensor | None = None,
    ) -> ConsensusState:
        if edges.dim() == 2 and edges.shape[0] == 2:
            ei, ej = edges[0], edges[1]
        elif edges.dim() == 2 and edges.shape[1] == 2:
            ei, ej = edges[:, 0], edges[:, 1]
        else:
            raise ValueError(f"bad edges shape {tuple(edges.shape)}")

        B, N, d = b.shape
        device, dtype = b.device, b.dtype
        state = make_state(B, N, edges.shape[-1], d, device, dtype, D_init=D_init, z_init=z_init)
        state.active = active
        history: list[dict] | None = [] if record else None

        col_mask = active.unsqueeze(-1)        # (B, N, 1)
        edge_mask = active[:, ei] * active[:, ej]  # (B, E)
        state.D = state.D * edge_mask

        for k in range(rounds):
            if detach_after is not None and k >= detach_after:
                state.x = state.x.detach().requires_grad_(False)
                state.z = state.z.detach().requires_grad_(False)
                state.u = state.u.detach().requires_grad_(False)

            # 1. primal
            state.x = ACNCore.primal(A, b * col_mask, state.z, state.u, rho)

            # measure disagreement
            diff = ACNCore.flux(state.x, r_ij, r_ji, ei, ej)
            Q = diff.pow(2).sum(-1)
            state.Q = Q * edge_mask

            # 2. z consensus step (diffusion across the graph). For all-pairs
            #    topology a single step fully mixes the active vote — no loop.
            if diffusion_steps <= 1:
                state.z = ACNCore.consensus_step(
                    state.z, state.x, state.u, r_ij, r_ji,
                    state.D, ei, ej, eta_z, rho,
                )
                state.z = state.z * col_mask
            else:
                for _ in range(diffusion_steps):
                    state.z = ACNCore.consensus_step(
                        state.z, state.x, state.u, r_ij, r_ji,
                        state.D, ei, ej, eta_z, rho,
                    )
                    state.z = state.z * col_mask

            # 3. Motor / efference copy / path integration
            if motor_system is not None:
                motor, shift_z, shift_u = motor_system(state.z)
                state.motor = motor
                # Path integration: update latent position
                state.z = state.z + col_mask * shift_z
                # Efference copy: add motor projection to dual
                #   u = u + (x - z) + shift_u
                state.u = state.u + col_mask * (state.x - state.z + shift_u)
            else:
                # standard dual
                state.u = state.u + col_mask * (state.x - state.z)

            if history is not None:
                hist = {
                    "x": state.x.detach(), "z": state.z.detach(),
                    "u": state.u.detach(), "D": state.D.detach(),
                    "Q": state.Q.detach(), "active": state.active.detach(),
                }
                if state.motor is not None:
                    hist["motor"] = state.motor.detach()
                history.append(hist)

        state.history = history
        return state

    # ------------------------------------------------------------------ #
    # Unified hierarchy loop (BPTT with detach_after)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _admm_round(
        state: ConsensusState,
        A: torch.Tensor, b: torch.Tensor,
        r_ij: torch.Tensor, r_ji: torch.Tensor,
        ei: torch.Tensor, ej: torch.Tensor,
        col_mask: torch.Tensor, edge_mask: torch.Tensor,
        diffusion_steps: int,
        rho: float, eta_z: float,
        motor_system: "MotorSystem | None",
    ) -> None:
        """One ADMM round in-place: primal -> consensus -> dual (+ motor)."""
        state.x = ACNCore.primal(A, b * col_mask, state.z, state.u, rho)
        diff = ACNCore.flux(state.x, r_ij, r_ji, ei, ej)
        Q = diff.pow(2).sum(-1)
        state.Q = Q * edge_mask
        if diffusion_steps <= 1:
            state.z = ACNCore.consensus_step(
                state.z, state.x, state.u, r_ij, r_ji,
                state.D, ei, ej, eta_z, rho,
            )
            state.z = state.z * col_mask
        else:
            for _ in range(diffusion_steps):
                state.z = ACNCore.consensus_step(
                    state.z, state.x, state.u, r_ij, r_ji,
                    state.D, ei, ej, eta_z, rho,
                )
                state.z = state.z * col_mask
        if motor_system is not None:
            motor, shift_z, shift_u = motor_system(state.z)
            state.motor = motor
            state.z = state.z + col_mask * shift_z
            state.u = state.u + col_mask * (state.x - state.z + shift_u)
        else:
            state.u = state.u + col_mask * (state.x - state.z)

    @staticmethod
    def run_hierarchical(
        A_b, b_b, r_ij_b, r_ji_b, edges_b, D_init_b, active_b,
        A_t, b_t, r_ij_t, r_ji_t, edges_t, D_init_t, active_t,
        decode_down, encode_up, decode_down_z=None,
        *,
        rounds, bottom_diffusion_steps, top_diffusion_steps,
        rho, eta_z, pc_eta_top, pc_eta_bottom,
        record=False, detach_after=None,
        bottom_motor_system=None, bottom_z_init=None, top_z_init=None,
        bottom_decoder=None, num_classes=10, vote_temperature=1.0,
        use_vote_input=False,
    ):
        """Unified hierarchy loop with BPTT (detach_after truncates gradient).

        Each round, the top layer reads the bottom layer's aggregate. If
        `use_vote_input`, that aggregate is the VOTE DISTRIBUTION (mean of
        softmax(bottom_decoder(z_b)) over active bottom columns, num_classes-dim)
        - letting the top layer see disagreement. Otherwise it's the legacy
        average z_b (d-dim).
        """
        ei_b, ej_b = (edges_b[0], edges_b[1]) if edges_b.shape[0] == 2 else (edges_b[:, 0], edges_b[:, 1])
        ei_t, ej_t = (edges_t[0], edges_t[1]) if edges_t.shape[0] == 2 else (edges_t[:, 0], edges_t[:, 1])

        B, N_b, d_b = b_b.shape
        _, N_t, d_t = b_t.shape
        device, dtype = b_b.device, b_b.dtype

        s_b = make_state(B, N_b, edges_b.shape[-1], d_b, device, dtype,
                         D_init=D_init_b, z_init=bottom_z_init)
        s_b.active = active_b
        s_t = make_state(B, N_t, edges_t.shape[-1], d_t, device, dtype,
                         D_init=D_init_t, z_init=top_z_init)
        s_t.active = active_t

        col_mask_b = active_b.unsqueeze(-1)
        col_mask_t = active_t.unsqueeze(-1)
        edge_mask_b = active_b[:, ei_b] * active_b[:, ej_b]
        edge_mask_t = active_t[:, ei_t] * active_t[:, ej_t]
        s_b.D = s_b.D * edge_mask_b
        s_t.D = s_t.D * edge_mask_t

        hist_b = [] if record else None
        hist_t = [] if record else None

        def _fuse_mean(z, active):
            w = active.unsqueeze(-1)
            return (w * z).sum(dim=1) / w.sum(dim=1).clamp(min=1e-6)

        from acn.networks import vote_fuse   # local import to avoid cycle

        def _bottom_aggregate(z, active):
            # The signal the top layer reads from the bottom each round.
            if use_vote_input and bottom_decoder is not None:
                return vote_fuse(z, active, bottom_decoder, num_classes, vote_temperature)
            return _fuse_mean(z, active)

        for k in range(rounds):
            if detach_after is not None and k >= detach_after:
                s_b.x = s_b.x.detach().requires_grad_(False)
                s_b.z = s_b.z.detach().requires_grad_(False)
                s_b.u = s_b.u.detach().requires_grad_(False)
                s_t.x = s_t.x.detach().requires_grad_(False)
                s_t.z = s_t.z.detach().requires_grad_(False)
                s_t.u = s_t.u.detach().requires_grad_(False)

            ACNCore._admm_round(s_b, A_b, b_b, r_ij_b, r_ji_b, ei_b, ej_b,
                                col_mask_b, edge_mask_b, bottom_diffusion_steps,
                                rho, eta_z, bottom_motor_system)

            z_global = _bottom_aggregate(s_b.z, s_b.active)
            ACNCore._admm_round(s_t, A_t, b_t, r_ij_t, r_ji_t, ei_t, ej_t,
                                col_mask_t, edge_mask_t, top_diffusion_steps,
                                rho, eta_z, None)

            top_mean = _fuse_mean(s_t.z, s_t.active)
            prediction_down = decode_down(top_mean)          # predicted bottom aggregate (vote or z)
            error_up = z_global - prediction_down
            correction_top = encode_up(error_up)
            s_t.z = s_t.z + col_mask_t * (pc_eta_top * correction_top).unsqueeze(1)
            # Top-down nudge on the bottom z. When the inter-layer signal is
            # the vote (dim != latent_dim), use the separate decode_down_z map
            # so the bottom columns get guidance in their own (latent) space.
            if decode_down_z is not None:
                z_nudge = decode_down_z(top_mean)            # (B, latent_dim)
            else:
                z_nudge = prediction_down                    # (B, latent_dim) — legacy
            s_b.z = s_b.z + col_mask_b * (pc_eta_bottom * z_nudge).unsqueeze(1)

            if hist_b is not None:
                hist_b.append({"z": s_b.z.detach(), "active": s_b.active.detach(),
                               "Q": s_b.Q.detach(), "D": s_b.D.detach()})
            if hist_t is not None:
                hist_t.append({"z": s_t.z.detach(), "active": s_t.active.detach(),
                               "Q": s_t.Q.detach(), "D": s_t.D.detach()})

        s_b.history = hist_b
        s_t.history = hist_t
        return s_b, s_t
