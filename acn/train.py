"""Training loop for ACN."""

from __future__ import annotations

from tqdm import tqdm
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from acn.config import ExperimentConfig, get_preset
from acn.model import AdaptiveConsensusNetwork
from acn.robustness import evaluate_robustness, ROBUSTNESS_VERSIONS
from acn.visualize import make_spotlight_gif, make_robustness_spotlight_gif

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(cfg: ExperimentConfig):
    dc = cfg.data
    if dc.dataset == "mnist":
        t = transforms.Compose([transforms.ToTensor()])
        ds = datasets.MNIST(root="./data", train=True, download=True, transform=t)
        val_size = int(len(ds) * dc.val_frac)
        train_size = len(ds) - val_size
        if dc.train_subset is not None and dc.train_subset < train_size:
            train_size = dc.train_subset
            val_size = len(ds) - train_size
        train_ds, val_ds = random_split(
            ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.train.seed),
        )
        test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=t)
    elif dc.dataset == "digits":
        t = transforms.Compose([transforms.ToTensor()])
        ds = datasets.MNIST(root="./data", train=True, download=True, transform=t)
        val_size = int(len(ds) * dc.val_frac)
        train_size = len(ds) - val_size
        if dc.train_subset is not None and dc.train_subset < train_size:
            train_size = dc.train_subset
            val_size = len(ds) - train_size
        train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.train.seed))
        test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=t)
    else:
        raise ValueError(f"unknown dataset {dc.dataset!r}")

    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, val_dl, test_dl


def evaluate(model: nn.Module, loader: DataLoader, loss_fn, cfg: ExperimentConfig):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred, _ = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * x.size(0)
            correct += (pred.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def compute_z_loss(s: torch.Tensor) -> torch.Tensor:
    """ST-MoE z-loss: penalizes large log-sum-exp of logits."""
    lse = torch.logsumexp(s, dim=-1)
    return (lse ** 2).mean()


def _load_test_tensors(test_dl: DataLoader):
    """Collect the full test set as two tensors (X, y) on CPU.

    Done once so robustness evaluation and the spotlight GIFs can reuse the
    same fixed samples at every snapshot interval.
    """
    xs, ys = [], []
    for x, y in test_dl:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def train(cfg: ExperimentConfig):
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)

    save_dir = Path(cfg.train.save_dir) / cfg.name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.yaml").write_text(cfg.to_yaml())

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    log = logging.getLogger(__name__)
    log.info(f"Running {cfg.name} on {device}")

    train_dl, val_dl, test_dl = get_dataloaders(cfg)

    model = AdaptiveConsensusNetwork(
        image_size=cfg.data.image_size,
        patch_size=cfg.data.patch_size,
        stride=cfg.data.stride,
        model_cfg=cfg.model,
        data_cfg=cfg.data,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    best_path = None
    metrics_log: list[dict] = []
    robustness_log: list[dict] = []
    start = time.time()

    # Fixed test-set tensors for robustness evaluation + spotlight GIFs.
    # We pull the full test set once so the same 1000 samples (and the same
    # one-per-digit spotlight rows) are reused at every snapshot interval —
    # making robustness accuracy and the GIFs comparable across epochs.
    Xte, yte = _load_test_tensors(test_dl)
    n_robust = min(cfg.train.robustness_n_samples, len(Xte))
    rob_X, rob_y = Xte[:n_robust], yte[:n_robust]
    log.info(f"robustness eval set: {n_robust} test samples × {len(ROBUSTNESS_VERSIONS)} versions")

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_sparsity = 0.0
        epoch_motor = 0.0

        pbar = tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch:02d}", leave=True)
        for batch_idx, (x, y) in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            pred, (state_bottom, state_abstract) = model(x)

            ce = loss_fn(pred, y)
            z_loss_bottom = 0.0
            z_loss_abstract = 0.0
            if state_bottom.relevance_logits is not None:
                z_loss_bottom = compute_z_loss(state_bottom.relevance_logits)
            if state_abstract is not None and state_abstract.relevance_logits is not None:
                z_loss_abstract = compute_z_loss(state_abstract.relevance_logits)

            local_loss = torch.zeros(1, device=device)
            if cfg.model.local_loss_weight > 0 and state_bottom.logits is not None:
                per_node = loss_fn(state_bottom.logits.view(-1, cfg.model.num_classes), y.unsqueeze(1).expand(-1, state_bottom.logits.shape[1]).reshape(-1))
                local_loss = per_node * state_bottom.active.view(-1).mean()

            active_frac = state_bottom.active.mean()
            sparsity_loss = cfg.model.column_sparsity_weight * active_frac
            sparsity_loss_abstract = 0.0
            if state_abstract is not None:
                sparsity_loss_abstract = cfg.model.column_sparsity_weight * state_abstract.active.mean()

            total_loss = (
                ce
                + cfg.model.column_z_loss_weight * z_loss_bottom
                + cfg.model.column_z_loss_weight * z_loss_abstract
                + cfg.model.local_loss_weight * local_loss
                + sparsity_loss
                + sparsity_loss_abstract
            )
            total_loss.backward()
            if cfg.train.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()

            epoch_loss += total_loss.item() * x.size(0)
            epoch_correct += (pred.argmax(dim=1) == y).sum().item()
            epoch_total += x.size(0)
            epoch_sparsity += active_frac.item() * x.size(0)
            if state_bottom.motor is not None:
                epoch_motor += state_bottom.motor.abs().mean().detach().item() * x.size(0)

            pbar.set_postfix({
                "loss": f"{total_loss.item():.3f}",
                "acc": f"{((pred.argmax(dim=1)==y).sum().item()/x.size(0)):.3f}",
                "sp": f"{active_frac.item():.3f}",
                "m": f"{state_bottom.motor.abs().mean().detach().item():.3f}" if state_bottom.motor is not None else "-",
            })

        pbar.close()
        epoch_loss /= epoch_total
        epoch_acc = epoch_correct / epoch_total
        epoch_sparsity /= epoch_total
        epoch_motor /= epoch_total

        val_loss, val_acc = evaluate(model, val_dl, loss_fn, cfg)
        log.info(
            f"Epoch {epoch:02d} | "
            f"train_loss={epoch_loss:.4f} train_acc={epoch_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"sparsity={epoch_sparsity:.3f} motor_norm={epoch_motor:.4f}"
        )
        metrics_log.append({
            "epoch": epoch, "train_loss": epoch_loss, "train_acc": epoch_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "sparsity": epoch_sparsity,
        })

        if val_acc > best_val:
            best_val = val_acc
            best_path = save_dir / "model_best.pt"
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, best_path)
            log.info(f"  -> saved best (val_acc={val_acc:.4f})")

        if epoch % cfg.train.snapshot_every == 0:
            ck = save_dir / f"model_e{epoch:03d}.pt"
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, ck)

            # ── robustness evaluation on the fixed 1000-sample set ──
            rob = evaluate_robustness(model, rob_X, rob_y, seed=42)
            rob_line = "  ".join(f"{k}={v:.3f}" for k, v in rob.items())
            log.info(f"  [snapshot e{epoch:02d}] robustness: {rob_line}")
            robustness_log.append({"epoch": epoch, **rob})
            (save_dir / "robustness_log.json").write_text(
                json.dumps(robustness_log, indent=2) + "\n")

            # ── spotlight GIFs for this checkpoint ──
            try:
                make_spotlight_gif(
                    model, rob_X, rob_y,
                    path=save_dir / f"model_e{epoch:03d}_spotlight.gif",
                    n_samples=min(cfg.train.viz_n_samples, n_robust),
                    duration_ms=200, linger_frames=15,
                )
                make_robustness_spotlight_gif(
                    model, rob_X, rob_y,
                    path=save_dir / f"model_e{epoch:03d}_robustness_spotlight.gif",
                    sample_idx=cfg.train.viz_sample_idx,
                    duration_ms=250, linger_frames=20,
                )
            except Exception as e:  # viz must never break training
                log.info(f"  [snapshot e{epoch:02d}] spotlight GIF skipped: {e}")

        (save_dir / "metrics.json").write_text(json.dumps(metrics_log, indent=2) + "\n")

    if best_path is not None:
        ck = torch.load(best_path, map_location=device)
        model.load_state_dict(ck["model"])
    test_loss, test_acc = evaluate(model, test_dl, loss_fn, cfg)
    log.info(f"Test loss={test_loss:.4f} acc={test_acc:.4f}")

    summary = {
        "best_val_acc": best_val, "test_acc": test_acc,
        "test_loss": test_loss, "config": cfg.to_dict(),
        "elapsed_seconds": time.time() - start,
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return model, summary


if __name__ == "__main__":
    import sys, argparse
    import yaml

    # parse overrides from CLI as dotted keys
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", nargs="?", default="mnist")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    cfg = get_preset(args.preset)
    changes = {}
    for ov in args.overrides:
        if "=" in ov:
            k, v = ov.split("=", 1)
            try:
                v = yaml.safe_load(v)
            except Exception:
                pass
            changes[k] = v

    if changes:
        cfg = cfg.override(**changes)
    model, summary = train(cfg)
