"""Configuration dataclasses driving every ACN run.

One :class:`ExperimentConfig` fully determines a run. Configs can be constructed
in code, loaded from YAML, or derived via :meth:`ExperimentConfig.override`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Dataset + decomposition granularity."""

    dataset: str = "mnist"      # "digits" (8x8) | "mnist" (28x28)
    image_size: int = 28
    patch_size: int = 7
    stride: int = 3
    normalize: bool = True
    val_frac: float = 0.1
    # cap on training samples (None = all)
    train_subset: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    """ACN architecture + consensus dynamics (Thousand-Brains sparse-column)."""

    latent_dim: int = 32           # d
    num_classes: int = 10
    # encoder/decoder
    enc_hidden: tuple[int, ...] = (64, 64)
    dec_hidden: tuple[int, ...] = (64, 64)
    # consensus
    rounds: int = 20                # K ADMM rounds
    diffusion_steps: int = 1       # T diffusion steps per round
    # ADMM hyperparams (frozen, non-learnable)
    rho: float = 1.0
    eta_z: float = 0.1
    # conductance (Physarum) dynamics — frozen rates, flux-mode only
    eta_D: float = 1.0
    gamma_D: float = 0.5
    D_init: float = 1.0
    D_init_mode: str = "overlap"   # "overlap" | "dense"
    D_clip: float = 1.0
    D_prune_eps: float = 0.05
    # === edge topology ===
    # "spatial"  : edges only between overlapping patches (the original;
    #              consensus is local — distant active columns are NOT wired).
    # "allpairs" : complete graph over all columns; the gate's edge_mask keeps
    #              only active-active pairs live per input, so every active
    #              column negotiates with every other active column regardless
    #              of image position (global consensus). N=64 → E=2016 (~3.7x
    #              the spatial edge count). The Physarum grow/prune rule runs
    #              on ALL edges — watch for over-coupling (all D racing to
    #              D_clip, vote agreement → ~100%). Fall back to "spatial" if
    #              ensemble diversity collapses. See BLUEPRINT §5 (all-pairs).
    topology_mode: str = "spatial"
    # SPD floor for A_i = L L^T + eps * I
    A_eps: float = 1e-2
    # backprop memory: detach graph after this many rounds (None = full unroll)
    detach_after: int | None = 4
    # auxiliary per-neuron loss: each mini network individually predicts the label
    # (makes columns competent local predictors so consensus has something to fuse)
    local_loss_weight: float = 0.5
    # === learned sparse column gate (the adaptive mechanism) ===
    # The encoder outputs a per-column relevance logit; the gate is sigmoid(logit).
    # The active count is DISCOVERED per input driven by this sparsity penalty:
    # column_sparsity_weight * mean(active). Higher = fewer columns active.
    # Must be gentle (~1e-3) so CE dominates first; too strong (e.g. 0.05)
    # collapses the gate to 0 before learning starts. The relevance head is
    # bias-init high so gates start near 1 (active) and sparsity prunes DOWN.
    # See BLUEPRINT.md §3, LOG_2026-07-05.
    column_sparsity_weight: float = 0.001
    # === router z-loss (ST-MoE 2022) — keeps the gate from saturating ===
    # Penalizes the squared log-sum-exp of the per-column relevance logits:
    #   z_loss = mean_batch( [ logsumexp(s) ]^2 )
    # This makes large logits quadratically expensive, so the gate can't drift
    # to "all columns fire" (the saturation failure seen when training on full
    # data). Unlike the linear sparsity penalty above, it acts on the *logits*
    # (the gate's inputs), not the active count — so per-input adaptivity is
    # preserved (an "8" can still fire more columns than a "1"). Weight 0.001
    # is the ST-MoE value; scale up if the gate still creeps. Biologically
    # analogous to homeostatic plasticity (keep total input bounded).
    column_z_loss_weight: float = 0.001
    # === confidence-weighted fusion (adaptive robustness) ===
    # Fuse mode controls how active columns' votes are combined:
    #   "sparse"     : equal weight among active columns (the original)
    #   "confident"  : weight each active column by exp(−H_norm_i / tau), where
    #                  H_norm_i is the normalized entropy of the column's own
    #                  per-class scorecard (0 = sharp, 1 = flat). Columns whose
    #                  patch is uninformative (flat scorecard — noise/occlusion)
    #                  get suppressed; sharp columns vote at full strength.
    #                  Uses only the column's own logits, so it works even for
    #                  isolated active columns (no active-neighbor dependency,
    #                  which broke the previous ‖u_i‖-based version under the
    #                  sparse spatial graph). Confidence is computed at runtime,
    #                  per input, per column.
    fuse_mode: str = "confident"
    # temperature on the normalized entropy for confident fusion. Smaller =
    # more aggressive suppression of flat-scorecard columns. tau=1.0 → a
    # perfectly flat column gets confidence ≈0.37; tau=0.5 → ≈0.14.
    fuse_tau: float = 1.0


@dataclass(frozen=True)
class TrainConfig:
    """Optimization + logging."""

    seed: int = 0
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    # sparsity regularization weight on mean(D) (wire conductance, secondary)
    lambda_sparse: float = 0.1
    grad_clip: float | None = 1.0
    log_every: int = 10
    save_dir: str = "results/runs"
    snapshot_every: int = 5
    device: str = "cuda"
    # generate the consensus GIF from the best checkpoint after training
    viz_after_train: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level config: data + model + train + a name."""

    name: str = "model_result"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def override(self, **changes: Any) -> "ExperimentConfig":
        """Return a new config with nested overrides (dotted keys, e.g. model.rounds=5)."""
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


# --------------------------------------------------------------------------- #
# presets
# --------------------------------------------------------------------------- #

_PRESETS: dict[str, ExperimentConfig] = {
    # Minimal test config (8x8 MNIST, 9 columns). Used by the test suite and for
    # fast sanity checks. Same learned-gate mechanism as the real config, just
    # smaller. Gates start near 1 (bias-init); sparsity prunes during training.
    "poc": ExperimentConfig(
        name="poc",
        data=DataConfig(
            dataset="digits", image_size=8, patch_size=4, stride=2, val_frac=0.2,
            train_subset=12000,
        ),
        model=ModelConfig(
            latent_dim=8, num_classes=10,
            enc_hidden=(16,), dec_hidden=(16,),
            rounds=3, diffusion_steps=1,
            rho=1.0, eta_z=0.1, eta_D=0.01, gamma_D=0.001,
            D_init=1.0, D_init_mode="dense",
            detach_after=None,
            local_loss_weight=0.0,
            column_sparsity_weight=0.001,
        ),
        train=TrainConfig(
            seed=0, epochs=50, batch_size=128, lr=1e-3,
            lambda_sparse=1e-3, device="cuda", snapshot_every=5,
            viz_after_train=False,
        ),
    ),
    # The real config: learned sparse column gate on native 28x28 MNIST. The
    # encoder learns a per-column relevance logit; the active count is DISCOVERED
    # per input, driven by the sparsity penalty. No fixed quota. A "1" recruits
    # few columns; an "8" recruits many. See BLUEPRINT.md §3, LOG_2026-07-05.
    "model_result": ExperimentConfig(
        name="model_result",
        data=DataConfig(
            dataset="mnist", image_size=28, patch_size=7, stride=3, val_frac=0.1,
            # train_subset=None -> use the full 60,000 MNIST training images.
        ),
        model=ModelConfig(
            latent_dim=32, num_classes=10,
            enc_hidden=(64, 64), dec_hidden=(64, 64),
            rounds=20, diffusion_steps=1,
            rho=1.0, eta_z=0.1,
            # conductance rates frozen; fast timescale (gamma_D=0.5 -> 2-round
            # settling, 20 rounds = 10 time constants). Steady state
            # D* = (eta_D/gamma_D)*phi; ratio=2 -> avg D*=2 (clips to 1), weak
            # links -> D*<0.05 -> pruned. See BLUEPRINT.md §3-4.
            eta_D=1.0, gamma_D=0.5,
            D_init=1.0, D_init_mode="overlap", D_prune_eps=0.05,
            detach_after=4, local_loss_weight=0.5,
            column_sparsity_weight=0.001,  # gentle; CE dominates first
        ),
        train=TrainConfig(
            seed=0, epochs=50, batch_size=128, lr=1e-3,
            lambda_sparse=0.1, device="cuda", snapshot_every=5,
            viz_after_train=True,
        ),
    ),
}


def get_preset(name: str) -> ExperimentConfig:
    """Return a named preset config (a copy; safe to mutate via override)."""
    if name not in _PRESETS:
        raise KeyError(f"unknown preset {name!r}; known: {list(_PRESETS)}")
    return replace(_PRESETS[name])


def _apply_overrides(cfg: ExperimentConfig, changes: dict[str, Any]) -> ExperimentConfig:
    """Apply dotted-path overrides immutably."""
    grouped: dict[str, dict[str, Any]] = {}
    for key, val in changes.items():
        if "." in key:
            section, leaf = key.split(".", 1)
            grouped.setdefault(section, {})[leaf] = val
        else:
            grouped[key] = val  # type: ignore[assignment]
    kwargs: dict[str, Any] = {}
    for section, vals in grouped.items():
        sub = getattr(cfg, section)
        if hasattr(sub, "__dataclass_fields__"):
            kwargs[section] = replace(sub, **vals)
        else:
            kwargs[section] = vals
    return replace(cfg, **kwargs)
