"""Configuration dataclasses for ACN."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "mnist"
    image_size: int = 28
    patch_size: int = 7
    stride: int = 3
    normalize: bool = True
    val_frac: float = 0.1
    train_subset: int | None = None

    decomposition_mode: str = "multi"
    multi_scale_specs: tuple[tuple[int, int], ...] = ((4, 4), (6, 6), (8, 8))
    pool_to: int = 4


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int = 32
    num_classes: int = 10
    enc_hidden: tuple[int, ...] = (64, 64)
    dec_hidden: tuple[int, ...] = (64, 64)
    rounds: int = 20
    diffusion_steps: int = 1
    rho: float = 1.0
    eta_z: float = 0.1

    D_init: float = 1.0
    # Bottom layer topology: ALL-PAIRS — every active column is wired to every
    # other active column, regardless of image position. The consensus vote is
    # a long-range phenomenon (active columns voting on the SAME object must
    # reconcile directly, not hop-by-hop through strangers), so the active↔active
    # wires (kept live by the edge_mask) realize "active columns talk to each
    # other wherever they are" — the Thousand-Brains voting principle. With
    # all-pairs, ONE diffusion step fully mixes the active vote, so
    # bottom_diffusion_steps=1 (no loop needed). The top (abstract) layer is
    # also all-pairs (8 conceptual columns).
    topology_mode: str = "allpairs"
    A_eps: float = 1e-2
    detach_after: int | None = 4
    local_loss_weight: float = 0.5

    column_sparsity_weight: float = 0.001
    column_z_loss_weight: float = 0.001

    # ────────────────────────────────────────────
    # Thousand-Brains reference-frame extensions
    # ────────────────────────────────────────────
    use_positional_encoding: bool = True
    pos_embed_dim: int = 32

    # Motor / efference copy / path integration
    use_motor: bool = True
    motor_dim: int = 2

    # Top-down prior for the abstract layer (the feedforward-sweep aggregate
    # reaches the top via a learned projection)
    use_abstract_topdown_primer: bool = True

    # Hard top-k gate
    topk_k: int = 8
    topk_tau: float = 0.5

    # Abstract Layer 2
    use_abstract_layer: bool = True
    abstract_num_columns: int = 8
    abstract_dim: int = 16
    abstract_rounds: int = 5
    abstract_topk: int = 4
    abstract_dec_hidden: tuple[int, ...] = (32, 32)
    abstract_enc_hidden: tuple[int, ...] = (32, 32)

    # Top-down abstract primer into bottom layer
    use_abstract_topdown_primer: bool = True

    # ────────────────────────────────────────────
    # unified hierarchy + predictive-coding fusion
    # ONE outer loop steps both layers together; a single linear (Kalman-style)
    # predictive-coding exchange reconciles them each round (predictions down,
    # errors up). The top layer's compressed belief is the verdict; the bottom's
    # readout becomes a (small) residual blend. See ACNCore.run_hierarchical.
    # ────────────────────────────────────────────
    hierarchy_rounds: int = 20
    bottom_diffusion_steps: int = 1   # all-pairs: 1 hop fully mixes the active vote (no loop)
    top_diffusion_steps: int = 1     # all-pairs top: same
    pc_eta_top: float = 0.5          # gain on the error that reaches the top
    pc_eta_bottom: float = 0.1       # gain on the prediction that nudges the bottom


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    log_every: int = 10
    save_dir: str = "results/runs"
    snapshot_every: int = 5
    device: str = "cuda"
    viz_after_train: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "acn_default"
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
        name="poc_result",
        data=DataConfig(
            dataset="digits", image_size=8, patch_size=4, stride=2, val_frac=0.2,
            train_subset=12000,
        ),
        model=ModelConfig(
            latent_dim=8, num_classes=10,
            enc_hidden=(16,), dec_hidden=(16,),
            rounds=3, diffusion_steps=1,
            rho=1.0, eta_z=0.1,
            D_init=1.0,
            detach_after=None,
            local_loss_weight=0.0,
            column_sparsity_weight=0.001,
            topology_mode="allpairs",
            use_motor=True, motor_dim=2,
            use_abstract_layer=True,
            abstract_num_columns=8, abstract_dim=8, abstract_rounds=2,
            topk_k=3,
            hierarchy_rounds=3, bottom_diffusion_steps=1, top_diffusion_steps=1,
            pc_eta_top=0.5, pc_eta_bottom=0.1,
        ),
        train=TrainConfig(
            seed=0, epochs=50, batch_size=128, lr=1e-3, device="cuda", snapshot_every=5,
            viz_after_train=False,
        ),
    ),
    "mnist": ExperimentConfig(
        name="acn_result",
        data=DataConfig(
            dataset="mnist", image_size=28, patch_size=7, stride=3, val_frac=0.1,
            decomposition_mode="multi",
            multi_scale_specs=((4, 4), (6, 6), (8, 8)),
            pool_to=4,
        ),
        model=ModelConfig(
            latent_dim=32, num_classes=10,
            enc_hidden=(64, 64), dec_hidden=(64, 64),
            rounds=20, diffusion_steps=1,
            rho=1.0, eta_z=0.1,
            D_init=1.0,
            detach_after=8,
            local_loss_weight=0.5,
            column_sparsity_weight=0.001,
            column_z_loss_weight=0.001,
            topology_mode="allpairs",
            use_motor=True, motor_dim=2,
            use_abstract_layer=True,
            abstract_num_columns=8, abstract_dim=16, abstract_rounds=5,
            topk_k=8, topk_tau=0.5,
            hierarchy_rounds=20, bottom_diffusion_steps=1, top_diffusion_steps=1,
            pc_eta_top=0.5, pc_eta_bottom=0.1,
        ),
        train=TrainConfig(
            seed=0, epochs=50, batch_size=128, lr=1e-3, device="cuda",
            snapshot_every=5, viz_after_train=True,
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
            grouped[key] = val  # type: ignore[assignment]
    kwargs: dict[str, Any] = {}
    for section, vals in grouped.items():
        sub = getattr(cfg, section)
        if hasattr(sub, "__dataclass_fields__"):
            kwargs[section] = replace(sub, **vals)
        else:
            kwargs[section] = vals
    return replace(cfg, **kwargs)
