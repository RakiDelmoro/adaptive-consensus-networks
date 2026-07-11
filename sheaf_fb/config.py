"""Configuration for the Sheaf-Forward-Backward network (Equilibrium Propagation).

A frozen dataclass tree holding the MNIST architecture / Forward-Backward
dynamics / Equilibrium-Propagation training hyperparameters.

Key differences from the sibling ``acn`` (Sheaf-ADMM + BPTT) package:

* **One state variable per agent** (``x_i``) instead of ADMM's three
  (``x_i, z_i, u_i``).
* **Forward-Backward (proximal gradient) dynamics** — a gradient step on the
  local objective followed by a proximal step on the sheaf consensus term —
  run for ``K`` rounds to settle to an equilibrium.
* **Equilibrium Propagation (EP)** training: two settles (free + nudged), the
  difference of equilibria is the local learning signal. No BPTT, no unrolling,
  no storing of intermediate rounds.
* **Per-edge learned sheaf restriction maps** ``F_ij`` of shape ``[c, d]`` that
  project each agent's private ``d``-dim state onto a shared ``c``-dim
  communication channel with a neighbor.

The grid is a 7x7 arrangement of agents, each owning a non-overlapping 4x4
patch of a 28x28 MNIST image (49 patches of 16 pixels), with 4-connected
adjacency (84 undirected edges).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Image decomposition geometry (determines the number of agents).

    28x28, patch 4, stride 4 -> centers at 2,6,...,26 -> 7x7 = 49 agents,
    4-connected grid (84 undirected edges).
    """

    image_size: int = 28
    patch_size: int = 4
    stride: int = 4
    connectivity: int = 4  # 4-way grid adjacency (up/down/left/right)
    num_classes: int = 10
    val_frac: float = 0.1
    train_subset: int | None = None
    data_root: str = "./data"


@dataclass(frozen=True)
class ModelConfig:
    """Sheaf-FB architecture: shared encoder, per-edge sheaf maps, shared decoder."""

    num_classes: int = 10

    # --- stalk dims ---
    d: int = 64  # vertex stalk (agent private state dim)
    c: int = 8  # edge stalk (shared communication channel dim)

    # --- encoder: MLP 16 -> hidden -> d, shared across all agents ---
    enc_hidden_dim: int = 64
    enc_bias: bool = True
    # Clamp the encoder output (local target theta_i) to [-theta_clip, theta_clip].
    # Bounds the local objective and the sheaf energy so the Forward-Backward
    # dynamics stay well-conditioned even as the encoder weights grow under EP's
    # 1/(2*beta) gradient amplification. 0.0 = no clamp.
    theta_clip: float = 3.0

    # --- decoder: MLP d -> hidden -> num_classes, shared across all agents ---
    dec_hidden_dim: int = 64
    dec_bias: bool = True

    # --- sheaf maps F_ij: one learned [c, d] matrix per edge direction ---
    # "per_edge"   -> every (i,j) edge direction gets its own F_ij  (spec default)
    # "directional"-> 4 shared base maps (N/E/S/W), gathered per edge
    sheaf_sharing: str = "per_edge"
    sheaf_init_scale: float = 0.002  # std of Gaussian init (||F|| ≈ sqrt(c*d)*scale)
    # Frobenius-norm bound per sheaf map. CRITICAL for EP: the Forward-Backward
    # dynamics converge only while ``eta * (1 + rho*lambda_LF) < 2``, where
    # ``lambda_LF`` scales as ``max_degree * ||F||^2``. With cap=0.05, degree=4,
    # eta=0.5: ``0.5 * (1 + 4*0.05^2) = 0.505 << 2``. If the cap is too large the
    # dynamics oscillate and EP gets garbage (no true equilibrium). Set to 0.0
    # to disable the projection.
    sheaf_max_norm: float = 0.05

    # --- Forward-Backward dynamics ---
    eta: float = 0.5  # step size (converges while eta*(1+rho*lambda_LF) < 2)
    rho: float = 1.0  # sheaf consensus penalty strength
    K: int = 50  # number of Forward-Backward rounds (same for train and eval)
    K_nudge: int = 50  # rounds for nudged phases (same as free for EP)
    # BPTT gradient window: only backpropagate through the last ``grad_window``
    # rounds of the K-round trajectory. The first ``K - grad_window`` rounds are
    # run under no_grad (detached) to let the system settle before building the
    # graph. This saves memory while still training the dynamics near the
    # equilibrium. Set to K (or None) to backprop through all rounds.
    grad_window: int = 20
    # Convergence tolerance: stop early when ||dE/dx|| < tol. 0.0 = always run K.
    converge_tol: float = 0.0
    warm_start: bool = True  # init x_i = theta_i (else x_i = 0)
    # Warm-start nudged phases from the free equilibrium. The nudged
    # equilibrium is close to the free one, so starting from x_free needs far
    # fewer rounds to track the perturbation.
    nudge_warm_start: bool = True

    # --- Equilibrium Propagation ---
    beta: float = 0.1  # nudging strength
    # EP gradient estimator variant:
    #   "symmetric" (default) -> 3-phase symmetric-difference EP (Laborieux et
    #     al. 2021): free + (+beta) + (-beta) settles; the +/-beta pair gives an
    #     unbiased (O(beta^2)) gradient estimate. Robust, scales to deep nets.
    #   "one_sided"           -> classic 2-phase EP (free + +beta); O(beta)
    #     biased estimate. Cheaper (2 settles) but prone to drift/instability.
    ep_variant: str = "symmetric"
    # Nudge / training loss: "ce" (cross-entropy on logits) or "mse" (spec:
    # 0.5 * ||softmax_mean - onehot||^2). Both are minimized by the same FB
    # dynamics; "ce" trains better, "mse" matches the spec literally.
    loss_type: str = "ce"
    # Per-agent loss aggregation for the nudge (Fix 1):
    #   True  -> L = sum_i loss(softmax(decoder(x_i)), y)  — each agent gets a
    #           FULL-STRENGTH learning signal (no 1/N dilution). Inference still
    #           uses the averaged vote.
    #   False -> L = loss(mean_i softmax(decoder(x_i)), y)  — the diluted version.
    per_agent_loss: bool = True


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1.0e-3
    # Training mode:
    #   "bptt" -> Backpropagation Through Time: unroll the K Forward-Backward
    #            rounds with a full autograd graph and backprop through all of
    #            them. Standard gradient-based training; uses more memory but
    #            gives exact gradients. Use this to verify the architecture.
    #   "ep"   -> Equilibrium Propagation: two/three settles (free + nudged),
    #            difference of equilibria as the local learning signal. No graph
    #            through the K rounds. Memory-efficient and biologically
    #            plausible, but harder to tune (see EP-specific config below).
    training_mode: str = "bptt"
    # Scale applied to the energy parameters (encoder + sheaf maps) relative to
    # ``lr``. EP's energy-param gradient carries a 1/(2*beta) amplification that
    # the decoder gradient does not, so the energy params need a much smaller
    # effective step to stay stable. 0.1 -> energy params see lr*0.1.
    # In BPTT mode this is typically 1.0 (all params get the same lr).
    energy_lr_scale: float = 1.0
    weight_decay: float = 1.0e-3
    grad_clip: float = 1.0  # global-norm clip
    val_interval: int = 1
    # Consensus visualization: generate a GIF every N epochs
    viz: bool = True
    viz_interval: int = 5
    viz_samples: int = 4
    exit_on_nan: bool = True
    save_dir: str = "results/runs"
    # Re-evaluate the free-phase equilibrium for accuracy reporting.
    eval_K: int | None = None  # None -> use model.K


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "sheaf_fb_mnist"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def override(self, **changes: Any) -> ExperimentConfig:
        return _apply_overrides(self, changes)

    def to_dict(self) -> dict[str, Any]:
        def _unroll(o: Any) -> Any:
            if isinstance(o, tuple):
                return list(o)
            if hasattr(o, "__dataclass_fields__"):
                return {k: _unroll(getattr(o, k)) for k in o.__dataclass_fields__}
            return o

        return _unroll(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_yaml())


# --- presets -----------------------------------------------------------------

_PRESETS: dict[str, ExperimentConfig] = {
    "mnist": ExperimentConfig(
        name="sheaf_fb_mnist",
        data=DataConfig(
            image_size=28, patch_size=4, stride=4, connectivity=4, num_classes=10,
            val_frac=0.1,
        ),
        model=ModelConfig(
            num_classes=10,
            d=64, c=8,
            enc_hidden_dim=64, dec_hidden_dim=64,
            sheaf_sharing="per_edge", sheaf_init_scale=0.02,
            eta=0.5, rho=1.0, K=50, K_nudge=10, grad_window=20, converge_tol=0.0,
            warm_start=True, nudge_warm_start=True,
            beta=0.1, loss_type="ce",
            sheaf_max_norm=1.0,
            ep_variant="symmetric",
            theta_clip=3.0,
            per_agent_loss=True,
        ),
        train=TrainConfig(
            seed=42, epochs=20, batch_size=128, lr=1.0e-3, training_mode="bptt",
            energy_lr_scale=1.0, weight_decay=1.0e-3, grad_clip=1.0, val_interval=1,
        ),
    ),
    # EP preset: Equilibrium Propagation from scratch. The key differences from
    # the BPTT preset are: (1) ``sheaf_max_norm=0.05`` — tight cap keeps the
    # Forward-Backward spectral radius near 1 so the dynamics truly converge to
    # an equilibrium (EP requires this); (2) ``training_mode="ep"``; (3)
    # ``per_agent_loss=True`` for a full-strength nudge signal; (4) symmetric
    # 3-phase EP (free + +beta + -beta) for an unbiased O(beta^2) gradient.
    "mnist_ep": ExperimentConfig(
        name="sheaf_fb_mnist_ep",
        data=DataConfig(
            image_size=28, patch_size=4, stride=4, connectivity=4, num_classes=10,
            val_frac=0.1,
        ),
        model=ModelConfig(
            num_classes=10,
            d=64, c=8,
            enc_hidden_dim=64, dec_hidden_dim=64,
            sheaf_sharing="per_edge", sheaf_init_scale=0.002,
            eta=0.5, rho=1.0, K=50, K_nudge=50, grad_window=20, converge_tol=0.0,
            warm_start=True, nudge_warm_start=True,
            beta=0.1, loss_type="ce",
            sheaf_max_norm=0.05,
            ep_variant="symmetric",
            theta_clip=3.0,
            per_agent_loss=True,
        ),
        train=TrainConfig(
            seed=42, epochs=20, batch_size=128, lr=1.0e-3, training_mode="ep",
            energy_lr_scale=0.1, weight_decay=1.0e-3, grad_clip=1.0, val_interval=1,
        ),
    ),
    # Small-scale smoke-test preset: same code path, tiny dims, few rounds.
    "poc": ExperimentConfig(
        name="sheaf_fb_poc",
        data=DataConfig(
            image_size=28, patch_size=4, stride=4, connectivity=4, num_classes=10,
            val_frac=0.2, train_subset=2000,
        ),
        model=ModelConfig(
            num_classes=10,
            d=16, c=4,
            enc_hidden_dim=32, dec_hidden_dim=32,
            sheaf_sharing="per_edge", sheaf_init_scale=0.02,
            eta=0.5, rho=1.0, K=50, K_nudge=5, grad_window=20, converge_tol=0.0,
            warm_start=True, nudge_warm_start=True,
            beta=0.1, loss_type="ce",
            sheaf_max_norm=1.0,
            ep_variant="symmetric",
            theta_clip=3.0,
            per_agent_loss=True,
        ),
        train=TrainConfig(
            seed=0, epochs=2, batch_size=64, lr=1.0e-2, training_mode="bptt",
            energy_lr_scale=1.0, weight_decay=1.0e-3,
            grad_clip=1.0, val_interval=1,
        ),
    ),
}


def get_preset(name: str) -> ExperimentConfig:
    if name not in _PRESETS:
        raise KeyError(f"unknown preset {name!r}; known: {list(_PRESETS)}")
    return replace(_PRESETS[name])


def _apply_overrides(cfg: ExperimentConfig, changes: dict[str, Any]) -> ExperimentConfig:
    grouped: dict[str, dict[str, Any]] = {}
    for key, val in changes.items():
        if "." in key:
            section, leaf = key.split(".", 1)
            grouped.setdefault(section, {})[leaf] = val
        else:
            grouped[key] = val
    kwargs: dict[str, Any] = {}
    for section, vals in grouped.items():
        sub = getattr(cfg, section)
        if hasattr(sub, "__dataclass_fields__"):
            kwargs[section] = replace(sub, **vals)
        else:
            kwargs[section] = vals
    return replace(cfg, **kwargs)
