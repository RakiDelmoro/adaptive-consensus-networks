"""Configuration for PCN-ACN (predictive-coding consensus network, EP-trained)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    image_size: int = 28
    patch_size: int = 5
    stride: int = 2
    val_frac: float = 0.1
    train_subset: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    # ── layer dims ──
    latent_dim: int = 32          # d  (per-column latent dim)
    num_classes: int = 10

    # ── modules ──
    enc_hidden: tuple[int, ...] = (256, 256)
    lateral_hidden: tuple[int, ...] = (256,)

    # ── gating (all-active by default) ──
    topk_k: int = 1000            # >= N -> all columns active
    topk_tau: float = 0.5

    # ── energy strengths (κ = lateral consensus, the key knob) ──
    kappa: float = 0.1            # lateral consensus (prediction-error coupling)
    lam_col: float = 1.0          # per-column prediction error
    lam_output: float = 1.0       # per-column output prediction error

    # ── settle (gradient descent on the energy) ──
    k_max: int = 20               # settle rounds
    alpha: float = 0.5            # gradient-descent step size

    # ── equilibrium propagation ──
    beta: float = 0.1             # nudge strength


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    save_dir: str = "results/runs"
    snapshot_every: int = 5
    robustness_n_samples: int = 1000
    use_md_optimizer: bool = True   # MD-decoupled weights (stability)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "acn"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def override(self, **changes: Any) -> "ExperimentConfig":
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


_PRESETS: dict[str, ExperimentConfig] = {
    "poc": ExperimentConfig(
        name="acn_poc",
        data=DataConfig(
            image_size=8, patch_size=4, stride=2, val_frac=0.2,
            train_subset=12000,
        ),
        model=ModelConfig(
            latent_dim=8, num_classes=10,
            enc_hidden=(16,), lateral_hidden=(16,),
            topk_k=1000,
            kappa=0.1, k_max=20, alpha=0.5, beta=0.1,
        ),
        train=TrainConfig(seed=0, epochs=30, batch_size=128, lr=3e-3,
                          snapshot_every=5),
    ),
    "mnist": ExperimentConfig(
        name="acn",
        data=DataConfig(
            image_size=28, patch_size=5, stride=2, val_frac=0.1,
        ),
        model=ModelConfig(
            latent_dim=128, num_classes=10,
            enc_hidden=(256, 256), lateral_hidden=(256,),
            topk_k=1000,
            kappa=0.1, k_max=20, alpha=0.5, beta=2.0,
            lam_output=4.0,
        ),
        train=TrainConfig(seed=0, epochs=50, batch_size=128, lr=0.005,
                          snapshot_every=5, use_md_optimizer=False),
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
            if hasattr(vals, "__dataclass_fields__"):
                kwargs[section] = vals
            else:
                kwargs[section] = replace(sub, **vals)
        else:
            kwargs[section] = vals
    return replace(cfg, **kwargs)
