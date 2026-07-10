"""Magnitude-Direction decoupled optimizer step (Algorithm 1 of arXiv:2606.25971).

Each weight matrix is factorized as  W = diag(g_row) Wc diag(g_col)  with Wc on a
fixed-norm hypersphere and learnable per-row/per-col gains. The direction Wc is
updated by a normalized base optimizer (Adam) and projected back to the sphere;
the gains are updated by Adam at a separate (usually smaller) rate. This stops
the weight magnitude from silently drifting — the failure mode that blows up
quadratic-in-weight energies like our sheaf term ½γ‖Fz‖².

Here we expose the per-matrix decoupled step used by `MDOptimizer`. The gains are
reparameterized through a positive map (softplus) so they stay positive and the
learning dynamics are well-conditioned.
"""
from __future__ import annotations

import torch


def md_step(W: torch.Tensor, G: torch.Tensor, state_Wc: dict, state_gain: dict,
            *, lr_W: float, lr_gain: float, eps: float = 1e-8,
            betas=(0.9, 0.999)):
    """One Magnitude-Direction decoupled step on a 2D weight matrix W.

    Uses scalar gain γ (W = γ ⊙ Wc, ‖Wc‖_F = 1). Returns the new W.
    `state_Wc` / `state_gain` hold the Adam moments for the direction / gain.
    """
    with torch.no_grad():
        # recover the current gain and direction
        norm = W.norm() + eps
        Wc = W / norm                       # direction (on the sphere)
        gamma = norm                        # scalar gain (the Frobenius norm)

        # split the gradient: g_gamma = <Wc, G>, G_Wc = gamma * G
        g_gamma = (Wc * G).sum()
        G_Wc = gamma * G

        # Adam step on the direction (normalized update ~ lr_W)
        m, v = state_Wc.get("m"), state_Wc.get("v")
        t = state_Wc.get("t", 0) + 1
        if m is None:
            m = torch.zeros_like(G_Wc); v = torch.zeros_like(G_Wc)
        m = betas[0] * m + (1 - betas[0]) * G_Wc
        v = betas[1] * v + (1 - betas[1]) * G_Wc * G_Wc
        m_hat = m / (1 - betas[0] ** t)
        v_hat = v / (1 - betas[1] ** t)
        Wc = Wc - lr_W * m_hat / (v_hat.sqrt() + eps)
        Wc = Wc / (Wc.norm() + eps)          # project back onto the sphere

        # Adam step on the gain
        mg, vg = state_gain.get("m"), state_gain.get("v")
        tg = state_gain.get("t", 0) + 1
        if mg is None:
            mg = torch.tensor(0.0, device=g_gamma.device); vg = torch.tensor(0.0, device=g_gamma.device)
        mg = betas[0] * mg + (1 - betas[0]) * g_gamma
        vg = betas[1] * vg + (1 - betas[1]) * g_gamma * g_gamma
        mg_hat = mg / (1 - betas[0] ** tg)
        vg_hat = vg / (1 - betas[1] ** tg)
        gamma = gamma.clamp(min=eps)     # keep the magnitude positive

        # reassemble
        W_new = gamma * Wc
        state_Wc.update(m=m, v=v, t=t)
        state_gain.update(m=mg, v=vg, t=tg)
        return W_new


class MDOptimizer:
    """Adam for most params; Magnitude-Direction decoupled Adam for tagged 2D weights.

    `md_params`: set of parameter names (as in named_parameters()) to decouple.
    All other parameters get plain Adam. This is a lightweight stand-in optimizer
    (not a torch.optim.Optimizer subclass) so it can apply the fused MD step
    directly on the .data of the tagged weights.
    """

    def __init__(self, model, md_params: set[str], *,
                 lr: float = 1e-3, lr_W: float = 1e-3, lr_gain: float = 1e-3,
                 eps: float = 1e-8, betas=(0.9, 0.999)):
        self.md_params = md_params
        self.lr = lr; self.lr_W = lr_W; self.lr_gain = lr_gain
        self.eps = eps; self.betas = betas
        # plain Adam state for non-MD params
        self._named = dict(model.named_parameters())
        self._adam_m = {n: torch.zeros_like(p) for n, p in self._named.items() if n not in md_params}
        self._adam_v = {n: torch.zeros_like(p) for n, p in self._named.items() if n not in md_params}
        self._adam_t = {n: 0 for n in self._named if n not in md_params}
        # MD state per tagged weight (each treated as one matrix with scalar gain)
        self._md_Wc = {n: dict() for n in md_params}
        self._md_g = {n: dict() for n in md_params}

    def zero_grad(self):
        for p in self._named.values():
            if p.grad is not None:
                p.grad = None

    @torch.no_grad()
    def step(self):
        b0, b1 = self.betas
        for n, p in self._named.items():
            G = p.grad
            if G is None:
                continue
            if n in self.md_params:
                # MD decoupled step on the whole tensor treated as one matrix
                # (works for any shape — we flatten to 2D: (num_dirs, d_e*d))
                orig_shape = G.shape
                W2 = p.data.reshape(orig_shape[0], -1) if G.dim() > 2 else p.data
                G2 = G.reshape(orig_shape[0], -1) if G.dim() > 2 else G
                # apply MD per-slice (each row of the 2D view is its own matrix)
                Wnew = W2.clone()
                for s in range(W2.shape[0]):
                    sw = self._md_Wc[n].setdefault(s, {})
                    sg = self._md_g[n].setdefault(s, {})
                    Wnew[s] = md_step(W2[s], G2[s], sw, sg,
                                      lr_W=self.lr_W, lr_gain=self.lr_gain,
                                      eps=self.eps, betas=self.betas)
                p.data = Wnew.reshape(orig_shape)
            else:
                # plain Adam
                t = self._adam_t[n] + 1
                self._adam_m[n] = b0 * self._adam_m[n] + (1 - b0) * G
                self._adam_v[n] = b1 * self._adam_v[n] + (1 - b1) * G * G
                m_hat = self._adam_m[n] / (1 - b0 ** t)
                v_hat = self._adam_v[n] / (1 - b1 ** t)
                p.data = p.data - self.lr * m_hat / (v_hat.sqrt() + self.eps)
                self._adam_t[n] = t
