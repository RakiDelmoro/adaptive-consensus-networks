"""Training loop for the Sheaf-FB network (Equilibrium Propagation).

Each step runs two settles (free + nudged) and applies the EP gradient estimate
(see :meth:`sheaf_fb.model.SheafFBModel.ep_step`). Train and eval use the
**exact same** forward pass — same ``K`` Forward-Backward rounds, same uniform
population vote. No BPTT, no unrolling, no storage of intermediate rounds.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from acn.data import get_mnist_loaders

from .config import ExperimentConfig, TrainConfig
from .model import SheafFBModel
from .task import MNISTTaskFB
from .viz import make_consensus_gif

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_optimizer(model: SheafFBModel, tc: TrainConfig) -> torch.optim.Optimizer:
    """Two parameter groups: energy params (encoder + sheaf) get a smaller lr.

    EP's energy-parameter gradient ``(1/(2*beta)) * [dE+/dtheta - dE-/dtheta]``
    carries a ``1/(2*beta)`` amplification that the decoder (readout) gradient
    does not. To keep both stable at the same base ``lr``, the energy params get
    ``lr * energy_lr_scale``.
    """
    energy_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (energy_params if name.startswith(("encoder.", "sheaf.")) else other_params).append(p)
    groups = [
        {"params": energy_params, "lr": tc.lr * tc.energy_lr_scale},
        {"params": other_params, "lr": tc.lr},
    ]
    return torch.optim.AdamW(groups, weight_decay=tc.weight_decay)


def evaluate(model: SheafFBModel, task: MNISTTaskFB, loader, *, max_batches: int | None = None) -> dict[str, float]:
    """Eval with the same free-phase forward pass and uniform vote as training."""
    model.eval()
    agg, n = {}, 0
    with torch.no_grad():
        for images, labels in loader:
            fwd, targets, _ = task.prepare(images, labels)
            logits = model(fwd["patches"])  # [N, B, C]
            m = task.evaluate(logits, targets)
            for kk, v in m.items():
                agg[kk] = agg.get(kk, 0.0) + v
            n += 1
            if max_batches is not None and n >= max_batches:
                break
    return {kk: v / max(n, 1) for kk, v in agg.items()}


def train(cfg: ExperimentConfig, verbose: bool = True) -> tuple[SheafFBModel, dict]:
    tc = cfg.train
    dc = cfg.data
    mc = cfg.model
    torch.manual_seed(tc.seed)

    train_dl, val_dl, test_dl = get_mnist_loaders(
        data_root=dc.data_root, batch_size=tc.batch_size, val_frac=dc.val_frac,
        train_subset=dc.train_subset, seed=tc.seed)

    task = MNISTTaskFB(dc, device)
    patch_dim = 1 * dc.patch_size * dc.patch_size
    model = SheafFBModel(
        mc, patch_dim=patch_dim,
        edge_indices=task.edge_indices, node_positions=task.node_positions,
    ).to(device)
    n_params = model.num_parameters()

    save_dir = Path(tc.save_dir) / cfg.name
    save_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[init] device={device}  agents(N)={task.N}  "
              f"edges(E)={task.edge_indices.shape[0]}  params={n_params:,}")
        print(f"[init] d={mc.d} c={mc.c} K={mc.K} eta={mc.eta} rho={mc.rho} "
              f"beta={mc.beta} lr={tc.lr} mode={tc.training_mode} loss={mc.loss_type}")

    optimizer = _make_optimizer(model, tc)

    history = []
    best_acc = 0.0

    for epoch in range(tc.epochs):
        model.train()
        t0 = time.time()
        running, nb = 0.0, 0
        pbar = tqdm(train_dl, desc=f"epoch {epoch:3d}", leave=False,
                    dynamic_ncols=True, disable=not verbose)
        for images, labels in pbar:
            fwd, targets, _ = task.prepare(images, labels)
            if tc.training_mode == "bptt":
                stats = model.bptt_step(fwd["patches"], targets["labels"], optimizer)
            else:
                stats = model.ep_step(fwd["patches"], targets["labels"], optimizer)
            if tc.grad_clip and tc.grad_clip > 0:
                # Clip per parameter group (not globally): with per-agent loss
                # the decoder gradient is ~N× larger than the energy-param
                # gradient, so a global clip would suppress the energy params.
                for g in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(g["params"], tc.grad_clip)
            optimizer.step()
            model.project_sheaf()

            running += stats["loss"]
            nb += 1
            if tc.training_mode == "bptt":
                pbar.set_postfix(loss=f"{stats['loss']:.4f}")
            else:
                pbar.set_postfix(
                    loss=f"{stats['loss']:.4f}",
                    delta=f"{stats['delta_norm']:.2e}",
                    e_plus=f"{stats.get('e_plus', stats.get('e_free', 0.0)):.1f}")
            if tc.exit_on_nan and not math.isfinite(stats["loss"]):
                pbar.close()
                print(f"[epoch {epoch}] non-finite loss — stopping.")
                return model, {"history": history, "best_acc": best_acc}
        pbar.close()

        train_loss = running / max(nb, 1)

        if epoch % tc.val_interval == 0 or epoch == tc.epochs - 1:
            val = evaluate(model, task, val_dl)
            test = evaluate(model, task, test_dl)
            best_acc = max(best_acc, test["acc"])
            row = {
                "epoch": epoch, "train_loss": train_loss,
                "val_acc": val["acc"], "test_acc": test["acc"],
                "time": time.time() - t0,
            }
            history.append(row)
            if verbose:
                print(f"[epoch {epoch:3d}] loss={train_loss:.4f}  "
                      f"val_acc={val['acc']*100:.2f}%  test_acc={test['acc']*100:.2f}%  "
                      f"({row['time']:.1f}s)")

            # Consensus visualization
            if tc.viz and (tc.viz_interval == 0 or epoch % tc.viz_interval == 0
                           or epoch == tc.epochs - 1):
                viz_dir = save_dir / "viz"
                viz_dir.mkdir(parents=True, exist_ok=True)
                viz_path = viz_dir / f"epoch_{epoch:03d}.gif"
                make_consensus_gif(
                    model, task, test_dl, viz_path,
                    epoch=epoch, device=device, n_samples=tc.viz_samples)
                if verbose:
                    print(f"  viz: {viz_path}")
        elif verbose:
            print(f"[epoch {epoch:3d}] loss={train_loss:.4f}  ({time.time() - t0:.1f}s)")

    summary = {
        "history": history, "best_acc": best_acc, "params": n_params,
        "num_agents": task.N, "num_edges": int(task.edge_indices.shape[0]),
        "config": cfg.to_dict(),
    }
    cfg.save(save_dir / "config.yaml")
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    torch.save({"model_state": model.state_dict(), "config": cfg.to_dict()},
               save_dir / "checkpoint.pt")
    if verbose:
        print(f"[done] best_test_acc={best_acc*100:.2f}%  saved to {save_dir}")
    return model, summary
