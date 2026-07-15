"""Training loop for the Lean Sheaf-ADMM consensus network on MNIST.

Adam + grad-clip, cross-entropy on the fused output plus a per-round CE
auxiliary (stabilizes training — every round's fused vote is a valid
prediction), per-batch tqdm progress, per-epoch val/test, best-val
checkpointing, and a per-epoch consensus-evolution GIF. Train and eval use the
**exact same** forward pass (fixed ``K`` rounds, full communication).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from tqdm import tqdm

from .config import ModelConfig, TrainConfig
from .graph import build_grid, patchify
from .model import AdaptiveConsensusModel
from .viz import make_consensus_gif

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_loaders(data_root: str, batch_size: int, val_frac: float = 0.1, seed: int = 0):
    transform = transforms.Compose([transforms.ToTensor()])
    train_full = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    n_val = int(len(train_full) * val_frac)
    n_train = len(train_full) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        train_full, [n_train, n_val], generator=gen
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=512, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


def make_model(model_cfg: ModelConfig) -> AdaptiveConsensusModel:
    ei, _npos, _gh, _gw, n = build_grid(
        model_cfg.image_size, model_cfg.patch_size, model_cfg.stride, model_cfg.connectivity)
    return AdaptiveConsensusModel(
        model_cfg, patch_dim=model_cfg.patch_size ** 2, edge_indices=ei, n_agents=n)


def _patches(images: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
    """``(B,1,28,28)`` -> ``(N, B, patch_dim)``."""
    return patchify(images, cfg.patch_size, cfg.stride).permute(1, 0, 2)


# ---------------------------------------------------------------------------
# Loss / eval
# ---------------------------------------------------------------------------

def compute_loss(model: AdaptiveConsensusModel, logits, aux, labels
                 ) -> tuple[torch.Tensor, dict]:
    """Task CE + per-round CE auxiliary + disagreement regularizers.

    Every round's fused vote is a valid prediction (per-round CE auxiliary).
    The ADMM disagreement (sheaf edge-energy / dual disagreement) is used as an
    extra learning signal.
    """
    l_task = F.cross_entropy(logits, labels)
    loss = l_task

    if model.cfg.round_aux_weight > 0:
        pr = aux["per_round_logits"]                      # (K, N, B, C)
        pr_probs = F.softmax(pr, dim=-1).mean(1)          # (K, B, C)
        pr_logits = torch.log(pr_probs.clamp_min(1e-8))   # (K, B, C)
        K, Bb, C = pr_logits.shape
        pr_flat = pr_logits.reshape(K * Bb, C)
        tgt_flat = labels.unsqueeze(0).expand(K, -1).reshape(K * Bb)
        w = torch.linspace(1.0 / K, 1.0, K, device=pr_logits.device)
        w = w.unsqueeze(-1).expand(K, Bb).reshape(K * Bb)
        l_aux = (w * F.cross_entropy(pr_flat, tgt_flat, reduction="none")).mean()
        loss = loss + model.cfg.round_aux_weight * l_aux

    l_energy = torch.zeros((), device=logits.device)
    l_disagree = torch.zeros((), device=logits.device)
    if model.cfg.edge_energy_weight > 0 and "edge_energy_final" in aux:
        l_energy = aux["edge_energy_final"].mean()
        loss = loss + model.cfg.edge_energy_weight * l_energy
    if model.cfg.disagreement_weight > 0 and "disagreement_final" in aux:
        l_disagree = aux["disagreement_final"]
        loss = loss + model.cfg.disagreement_weight * l_disagree

    diag = {"l_task": float(l_task),
            "l_energy": float(l_energy),
            "l_disagree": float(l_disagree)}
    return loss, diag


@torch.no_grad()
def evaluate(model: AdaptiveConsensusModel, loader, model_cfg: ModelConfig
             ) -> tuple[float, float, dict]:
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    agg = {"avg_disagreement": 0.0}
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        patches = _patches(images, model_cfg)
        logits, aux = model(patches, K=model_cfg.K_eval)
        loss, _diag = compute_loss(model, logits, aux, labels)
        loss_sum += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
        agg["avg_disagreement"] += float(aux["conv_log"].avg_disagreement.mean()) * images.size(0)
    n = max(total, 1)
    return correct / n, loss_sum / n, {k: v / n for k, v in agg.items()}


# ---------------------------------------------------------------------------
# Train driver
# ---------------------------------------------------------------------------

def train(model_cfg: ModelConfig, train_cfg: TrainConfig, *, verbose: bool = True
          ) -> tuple[AdaptiveConsensusModel, dict]:
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    model = make_model(model_cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Device: {DEVICE} | params: {n_params:,}")
        print(f"Model: d={model_cfg.d} k={model_cfg.k} K={model_cfg.K} K_eval={model_cfg.K_eval} "
              f"T={model_cfg.T} rho={model_cfg.rho_init} lr_z={model_cfg.lr_z_init} "
              f"agents={model.n_agents} edges={model.num_edges}")

    train_loader, val_loader, test_loader = get_loaders(
        train_cfg.data_root, train_cfg.batch_size, train_cfg.val_frac, train_cfg.seed)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr,
                           weight_decay=train_cfg.weight_decay)

    save_dir = Path(train_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = save_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    best_val = 0.0
    best_state = None
    history = []

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, run_correct, run_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch:2d}/{train_cfg.epochs}",
                    dynamic_ncols=True, leave=False)
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            patches = _patches(images, model_cfg)

            optimizer.zero_grad()
            logits, aux = model(patches)
            loss, _diag = compute_loss(model, logits, aux, labels)
            loss.backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

            bs = images.size(0)
            run_loss += loss.item() * bs
            run_correct += (logits.argmax(1) == labels).sum().item()
            run_total += bs
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{(logits.argmax(1) == labels).float().mean().item():.3f}",
            )
        pbar.close()

        train_loss = run_loss / max(run_total, 1)
        train_acc = run_correct / max(run_total, 1)
        val_acc, val_loss, val_diag = evaluate(model, val_loader, model_cfg)
        dt = time.time() - t0
        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"
        if verbose:
            print(
                f"epoch {epoch:2d}/{train_cfg.epochs} "
                f"loss={train_loss:.4f} acc={train_acc:.4f} "
                f"val_acc={val_acc:.4f} val_loss={val_loss:.4f} "
                f"u={val_diag['avg_disagreement']:.4f} "
                f"[{dt:.1f}s]{marker}"
            )

        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
               "val_acc": val_acc, "val_loss": val_loss, "time": dt, **val_diag}

        if train_cfg.viz and (train_cfg.viz_interval == 0
                              or epoch % train_cfg.viz_interval == 0
                              or epoch == train_cfg.epochs):
            vp = viz_dir / f"epoch_{epoch:03d}.gif"
            make_consensus_gif(model, test_loader, vp, model_cfg=model_cfg,
                               epoch=epoch, device=DEVICE,
                               n_samples=train_cfg.viz_samples)
            row["viz"] = str(vp)
            if verbose:
                print(f"  viz: {vp}")

        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, test_loss, test_diag = evaluate(model, test_loader, model_cfg)
    if verbose:
        print(f"\nBEST val_acc={best_val:.4f}  TEST acc={test_acc:.4f} loss={test_loss:.4f} "
              f"u={test_diag['avg_disagreement']:.4f}")

    save_path = save_dir / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict(), "model_cfg": model_cfg,
                "test_acc": test_acc}, save_path)
    if verbose:
        print(f"saved -> {save_path}")

    summary = {"history": history, "best_val": best_val, "test_acc": test_acc,
               "test_diag": test_diag, "params": n_params}
    return model, summary
