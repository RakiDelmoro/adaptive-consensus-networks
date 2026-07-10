"""Training loop for ACN-v2 (equilibrium propagation)."""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from acn.config import ExperimentConfig, get_preset
from acn.robustness import evaluate_robustness
from acn.model import ACNv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(cfg: ExperimentConfig):
    dc = cfg.data
    t = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=t)
    val_size = int(len(ds) * dc.val_frac)
    train_size = len(ds) - val_size
    if dc.train_subset is not None and dc.train_subset < train_size:
        train_size = dc.train_subset
        val_size = len(ds) - train_size
    train_ds, val_ds = random_split(
        ds, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.train.seed))
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=t)
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)
    return train_dl, val_dl, test_dl


@torch.no_grad()
def evaluate(model: nn.Module, loader, cfg: ExperimentConfig):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred, _ = model(x)
        total_loss += loss_fn(pred, y).item() * x.size(0)
        correct += (pred.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def _load_test_tensors(test_dl):
    xs, ys = [], []
    for x, y in test_dl:
        xs.append(x); ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def train(cfg: ExperimentConfig):
    random.seed(cfg.train.seed); np.random.seed(cfg.train.seed); torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)

    save_dir = Path(cfg.train.save_dir) / cfg.name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.yaml").write_text(cfg.to_yaml())

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    log = logging.getLogger(__name__)
    log.info(f"Running {cfg.name} on {device}  (equilibrium propagation, no BPTT)")

    train_dl, val_dl, test_dl = get_dataloaders(cfg)

    model = ACNv2(
        image_size=cfg.data.image_size, patch_size=cfg.data.patch_size,
        stride=cfg.data.stride, model_cfg=cfg.model, data_cfg=cfg.data,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)
    if cfg.train.use_md_optimizer:
        # MD-decoupled weights: keeps each weight matrix's norm fixed (no silent
        # magnitude drift) — the stability fix from arXiv:2606.25971. Applied to
        # all 2D weight matrices; 1D params (biases) use plain Adam.
        from acn.md_optimizer import MDOptimizer
        md_params = {n for n, p in model.named_parameters() if p.dim() >= 2}
        opt = MDOptimizer(model, md_params=md_params,
                          lr=cfg.train.lr, lr_W=cfg.train.lr, lr_gain=cfg.train.lr * 0.1)
        log.info(f"  using MD-decoupled optimizer on {len(md_params)} weight matrices")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                               weight_decay=cfg.train.weight_decay)
    best_val = 0.0
    best_path = None
    metrics_log: list[dict] = []
    robustness_log: list[dict] = []
    start = time.time()

    Xte, yte = _load_test_tensors(test_dl)
    n_rob = min(cfg.train.robustness_n_samples, len(Xte))
    rob_X, rob_y = Xte[:n_rob], yte[:n_rob]

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_total = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch:02d}", leave=True)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            # ── equilibrium propagation: the loss IS the energy contrast ──
            eq_loss = model.eqprop_loss(x, y)
            eq_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # lightweight monitoring: use the contrast value (no extra forward pass).
            epoch_loss += float(eq_loss) * x.size(0)
            epoch_total += x.size(0)
            pbar.set_postfix({
                "eq": f"{float(eq_loss):.3f}",
            })
        pbar.close()

        val_loss, val_acc = evaluate(model, val_dl, cfg)
        log.info(
            f"Epoch {epoch:02d} | train_eq={epoch_loss/epoch_total:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        metrics_log.append({"epoch": epoch, "train_eq": epoch_loss/epoch_total,
                            "val_loss": val_loss, "val_acc": val_acc})

        # ── per-epoch consensus visualization: 4 test samples ──
        try:
            from acn.viz_epoch import make_epoch_consensus_gif
            model.eval()
            # pick 4 fixed test samples (digits 0,1,2,3) for consistency across epochs
            viz_indices = []
            for d in range(4):
                matches = (yte == d).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    viz_indices.append(matches[0].item())
            viz_path = save_dir / f"consensus_e{epoch:03d}.gif"
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            make_epoch_consensus_gif(model, Xte, yte, viz_indices, viz_path,
                                      cfg.model.k_max, cfg.model.alpha, epoch, device)
            log.info(f"  [viz] saved consensus_e{epoch:03d}.gif")
            model.train()
        except Exception as e:
            log.info(f"  [viz] skipped: {e}")

        if val_acc > best_val:
            best_val = val_acc
            best_path = save_dir / "model_best.pt"
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, best_path)
            log.info(f"  -> saved best (val_acc={val_acc:.4f})")

        if epoch % cfg.train.snapshot_every == 0:
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch},
                       save_dir / f"model_e{epoch:03d}.pt")
            rob = evaluate_robustness(model, rob_X, rob_y, seed=42)
            rob_line = "  ".join(f"{k}={v:.3f}" for k, v in rob.items())
            log.info(f"  [snapshot e{epoch:02d}] robustness: {rob_line}")
            robustness_log.append({"epoch": epoch, **rob})
            (save_dir / "robustness_log.json").write_text(
                json.dumps(robustness_log, indent=2) + "\n")

        (save_dir / "metrics.json").write_text(json.dumps(metrics_log, indent=2) + "\n")

    if best_path is not None:
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    test_loss, test_acc = evaluate(model, test_dl, cfg)
    log.info(f"Test loss={test_loss:.4f} acc={test_acc:.4f}")
    summary = {"best_val_acc": best_val, "test_acc": test_acc, "test_loss": test_loss,
               "config": cfg.to_dict(), "elapsed_seconds": time.time() - start}
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return model, summary


if __name__ == "__main__":
    import argparse, yaml
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
    train(cfg)
