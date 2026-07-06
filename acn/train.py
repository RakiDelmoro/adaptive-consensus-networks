"""ACN training loop: data loading, loss, logging, snapshots.

Run with ``python -m acn.train --config poc``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from acn.config import ExperimentConfig, get_preset
from acn.inspect import snapshot, link_summary, node_summary, StateSnapshot
from acn.model import AdaptiveConsensusNetwork


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_cuda_fast_math() -> None:
    """Maximize GPU throughput for ACN's small-tile matmuls/solves.

    - cudnn.benchmark: pick best conv/matmul algos (helps fixed shapes per epoch)
    - TF32 on Ampere+: ~8x faster matmuls/solves with negligible accuracy impact
      at d<=32 / batch<=512; we restore precision for gradcheck-style tests via
      the float64 path in tests/ which does not call this.
    - matmul precision 'high' allows TF32 for torch.linalg on torch>=2.0.
    """
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def _load_mnist_tensors(size: int, normalize: bool, seed: int, val_frac: float,
                        train_subset: int | None = None):
    """Download MNIST once and return (train, val) image/label tensors.

    `size` downsamples MNIST via bilinear resize (e.g. 8 for the small-scale POC,
    28 for full MNIST). Images are (N,1,size,size) float32 in [0,1] (or standardized).
    """
    from torchvision import datasets, transforms

    root = str(Path("data").resolve())
    tfms = [transforms.ToTensor(), transforms.Resize((size, size), antialias=True)]
    tfm = transforms.Compose(tfms)
    tr = datasets.MNIST(root, train=True, download=True, transform=tfm)
    te = datasets.MNIST(root, train=False, download=True, transform=tfm)
    Xtr = torch.stack([tr[i][0] for i in range(len(tr))])       # (60000,1,size,size)
    ytr = torch.tensor([tr[i][1] for i in range(len(tr))], dtype=torch.long)
    Xte = torch.stack([te[i][0] for i in range(len(te))])       # (10000,1,size,size)
    yte = torch.tensor([te[i][1] for i in range(len(te))], dtype=torch.long)

    if normalize:
        mu, sd = Xtr.mean(), Xtr.std() + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd

    # We use the official MNIST test set as the eval split. `val_frac` is reserved
    # for future early-stopping and does not reduce the training set here.
    if train_subset is not None and train_subset > 0:
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(Xtr), generator=gen)[:train_subset]
        Xtr, ytr = Xtr[perm], ytr[perm]
    return (Xtr, ytr), (Xte, yte)


def load_digits(cfg):
    """Small-scale POC: MNIST resized to 8x8 (torch-only, no sklearn)."""
    return _load_mnist_tensors(
        size=cfg.data.image_size, normalize=cfg.data.normalize,
        seed=cfg.train.seed, val_frac=cfg.data.val_frac,
    )


def load_mnist(cfg):
    """Full MNIST at native resolution (torch-only)."""
    return _load_mnist_tensors(
        size=cfg.data.image_size, normalize=cfg.data.normalize,
        seed=cfg.train.seed, val_frac=cfg.data.val_frac,
    )


def make_loaders(cfg: ExperimentConfig, device):
    if cfg.data.dataset not in ("digits", "mnist"):
        raise ValueError(cfg.data.dataset)
    # Both presets use torchvision MNIST; "digits" is MNIST resized to 8x8,
    # "mnist" is native 28x28. Resolution comes from cfg.data.image_size.
    (Xtr, ytr), (Xte, yte) = _load_mnist_tensors(
        size=cfg.data.image_size, normalize=cfg.data.normalize,
        seed=cfg.train.seed, val_frac=cfg.data.val_frac,
        train_subset=cfg.data.train_subset,
    )
    Xtr, ytr = Xtr.to(device, non_blocking=True), ytr.to(device, non_blocking=True)
    Xte, yte = Xte.to(device, non_blocking=True), yte.to(device, non_blocking=True)
    return (Xtr, ytr), (Xte, yte)


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #


def acn_loss(pred: torch.Tensor, labels: torch.Tensor, D: torch.Tensor, lam: float) -> tuple[torch.Tensor, dict]:
    ce = F.cross_entropy(pred, labels)
    sparse = lam * D.mean() if D.numel() > 0 else torch.zeros((), device=pred.device)
    loss = ce + sparse
    return loss, {"ce": float(ce.detach()), "sparse": float(sparse.detach()),
                  "loss": float(loss.detach())}


def acn_loss_gpu(pred, labels, D, lam, logits_per_node, local_weight, active,
                 gate_sparsity_weight, relevance_logits, z_loss_weight):
    """The ACN training loss.

    loss = CE(fused, label)                         # the team must predict the label
         + lam * mean(D)                            # wire-conductance sparsity (secondary)
         + local_weight * per_column_CE             # each column must predict the label
         + gate_sparsity_weight * mean(active)      # fire fewer columns (linear, weak)
         + z_loss_weight * (logsumexp(s))^2         # ST-MoE router z-loss: keep gate
                                                      # logits bounded so the gate can't
                                                      # saturate to "all columns fire".

    `active` is the learned gate (B, N) in [0,1]; `logits_per_node` is the per-column
    decoder output (B, N, C); `relevance_logits` is the gate's raw input s (B, N).
    See BLUEPRINT.md §3, LOG_2026-07-05, and ST-MoE (arXiv 2202.08906).
    """
    ce = F.cross_entropy(pred, labels)                                   # (B,)
    sparse = lam * D.mean() if D.numel() > 0 else torch.zeros((), device=pred.device)
    logp = F.log_softmax(logits_per_node, dim=-1)                        # (B, N, C)
    target = labels.view(-1, 1, 1).expand(-1, logits_per_node.shape[1], 1)  # (B, N, 1)
    local = -logp.gather(-1, target).squeeze(-1).mean()                  # mean over B*N
    gate_sparsity = active.mean()                                        # mean fraction active
    # ST-MoE router z-loss: penalize the squared log-sum-exp of the gate logits.
    # logsumexp(s) is a smooth "max" over columns — large when any logit is large.
    # Squaring it makes large logits quadratically costly, preventing the gate
    # from drifting to saturation (all columns firing on every input).
    z_loss = (torch.logsumexp(relevance_logits, dim=-1).mean()) ** 2
    loss = (ce + sparse + local_weight * local
            + gate_sparsity_weight * gate_sparsity
            + z_loss_weight * z_loss)
    return loss, {"ce": ce.detach(), "sparse": sparse.detach(), "loss": loss.detach(),
                  "local": local.detach(), "gate_sparsity": gate_sparsity.detach(),
                  "z_loss": z_loss.detach()}


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #


def iterate_minibatches(n: int, bs: int, shuffle: bool, seed, device=None):
    """Yield GPU-resident index slices (no CPU sync).

    When `device` is a CUDA device, the permutation is generated on-GPU and the
    returned index tensors already live on `device`, so indexing data tensors on
    the same device never crosses the bus.
    """
    if device is not None and device.type == "cuda":
        gen = torch.Generator(device=device).manual_seed(int(seed) % (2**31 - 1))
        idx = torch.randperm(n, generator=gen, device=device) if shuffle else torch.arange(n, device=device)
        for s in range(0, n, bs):
            yield idx[s:s + bs]
    else:
        idx = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)
        for s in range(0, n, bs):
            yield torch.as_tensor(idx[s:s + bs], dtype=torch.long)


@torch.no_grad()
def evaluate(model, X, y, bs=256, device=None):
    """Eval accumulating on GPU; a single sync at the end."""
    model.eval()
    correct = torch.zeros((), device=X.device, dtype=torch.float32)
    loss_sum = torch.zeros((), device=X.device, dtype=torch.float32)
    total = 0
    for idx in iterate_minibatches(len(X), bs, shuffle=False, seed=0, device=device):
        xb, yb = X[idx], y[idx]
        pred, state = model(xb)
        l, _ = acn_loss(pred, yb, state.D, lam=0.0)
        loss_sum += l.detach() * len(idx)
        correct += (pred.argmax(-1) == yb).sum()
        total += len(idx)
    acc = (correct / total).item()
    loss = (loss_sum / total).item()
    return {"acc": acc, "loss": loss}


def train(cfg: ExperimentConfig) -> dict:
    set_seed(cfg.train.seed)
    enable_cuda_fast_math()
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    if cfg.train.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    (Xtr, ytr), (Xte, yte) = make_loaders(cfg, device)
    print(f"data: train={len(Xtr)} eval={len(Xte)} image={tuple(Xtr.shape[1:])}")

    model = AdaptiveConsensusNetwork(
        image_size=cfg.data.image_size,
        patch_size=cfg.data.patch_size,
        stride=cfg.data.stride,
        model_cfg=cfg.model,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: params={n_params}, edges={None} (built on first forward)")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    run_dir = Path(cfg.train.save_dir) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")

    D_history: list[np.ndarray] = []
    metrics_log = []
    best_acc = 0.0

    for epoch in range(cfg.train.epochs):
        model.train()
        # accumulators stay on device; single sync at end of epoch
        ep_loss = torch.zeros((), device=device, dtype=torch.float32)
        ep_ce = torch.zeros((), device=device, dtype=torch.float32)
        ep_sparse = torch.zeros((), device=device, dtype=torch.float32)
        ep_local = torch.zeros((), device=device, dtype=torch.float32)
        ep_gate = torch.zeros((), device=device, dtype=torch.float32)
        ep_z = torch.zeros((), device=device, dtype=torch.float32)
        seen = 0
        lw = cfg.model.local_loss_weight
        gsw = cfg.model.column_sparsity_weight
        zw = cfg.model.column_z_loss_weight
        n_batches = (len(Xtr) + cfg.train.batch_size - 1) // cfg.train.batch_size
        pbar = tqdm(
            iterate_minibatches(len(Xtr), cfg.train.batch_size, shuffle=True,
                                seed=cfg.train.seed + epoch, device=device),
            total=n_batches, desc=f"epoch {epoch:3d}", leave=False, dynamic_ncols=True,
        )
        for idx in pbar:
            xb, yb = Xtr[idx], ytr[idx]
            pred, state = model(xb)
            loss, comps = acn_loss_gpu(pred, yb, state.D, cfg.train.lambda_sparse,
                                       logits_per_node=state.logits,
                                       local_weight=lw, active=state.active,
                                       gate_sparsity_weight=gsw,
                                       relevance_logits=state.relevance_logits,
                                       z_loss_weight=zw)
            opt.zero_grad()
            loss.backward()
            if cfg.train.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            nb = xb.shape[0]
            ep_loss += comps["loss"] * nb
            ep_ce += comps["ce"] * nb
            ep_sparse += comps["sparse"] * nb
            ep_local += comps["local"] * nb
            ep_gate += comps["gate_sparsity"] * nb
            ep_z += comps["z_loss"] * nb
            seen += nb
            pbar.set_postfix(loss=f"{float(comps['loss']):.3f}", ce=f"{float(comps['ce']):.3f}")
        pbar.close()

        eval_m = evaluate(model, Xte, yte, device=device)
        rec = {
            "epoch": epoch,
            "train_loss": (ep_loss / seen).item(), "train_ce": (ep_ce / seen).item(),
            "train_sparse": (ep_sparse / seen).item(), "train_local": (ep_local / seen).item(),
            "train_gate_sparsity": (ep_gate / seen).item(),
            "train_z_loss": (ep_z / seen).item(),
            **{f"eval_{k}": v for k, v in eval_m.items()},
        }
        metrics_log.append(rec)
        if eval_m["acc"] > best_acc:
            best_acc = eval_m["acc"]
            torch.save(model.state_dict(), run_dir / "model_best.pt")
        if cfg.train.log_every and (epoch % cfg.train.log_every == 0 or epoch == cfg.train.epochs - 1):
            print(
                f"epoch {epoch:3d} | loss {rec['train_loss']:.4f} ce {rec['train_ce']:.4f} "
                f"local {rec['train_local']:.4f} sparse {rec['train_sparse']:.4f} "
                f"gate {rec['train_gate_sparsity']:.4f} zloss {rec['train_z_loss']:.4f} | "
                f"eval acc {eval_m['acc']:.4f} loss {eval_m['loss']:.4f}"
            )

        if cfg.train.snapshot_every and (epoch % cfg.train.snapshot_every == 0 or epoch == cfg.train.epochs - 1):
            snap, _ = snapshot(model, Xte[:64], record=True)
            D_history.append(snap.D.mean(axis=0))
            np.savez(run_dir / f"D_epoch{epoch:03d}.npz", D=snap.D.mean(axis=0))

    # final snapshot + summaries
    snap, _ = snapshot(model, Xte[:128], record=True)
    snap.save(run_dir / "final_state.npz")
    ls = link_summary(snap, prune_eps=cfg.model.D_prune_eps)
    ns = node_summary(snap)
    final = {
        "best_acc": best_acc,
        "final_acc": eval_m["acc"],
        "n_params": n_params,
        "link": ls.__dict__, "node": ns.__dict__,
        "epochs": cfg.train.epochs,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_log, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(final, indent=2))
    if D_history:
        np.savez(run_dir / "D_history.npz", history=np.stack(D_history))
    print(f"done. best_acc={best_acc:.4f} active_link_frac={ls.active_link_frac:.3f}")

    # automatically generate the consensus-agreement GIF from the best checkpoint
    if cfg.train.viz_after_train:
        print("\n=== generating consensus visualization ===")
        # import lazily so the training module doesn't hard-depend on the viz script
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "viz_consensus", Path(__file__).resolve().parent.parent / "scripts" / "viz_consensus.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["viz_consensus"] = mod
        spec.loader.exec_module(mod)
        gif_path = mod.run_viz(cfg, device=str(device))
        print(f"consensus GIF: {gif_path}")

    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="model_result", help="preset name or path to YAML")
    p.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. model.rounds=5")
    args = p.parse_args()

    cfg = get_preset(args.config)
    for ov in args.override:
        k, v = ov.split("=", 1)
        # try to coerce to number
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        cfg = cfg.override(**{k: v})

    print(f"=== experiment: {cfg.name} ===")
    print(cfg.to_yaml())
    train(cfg)


if __name__ == "__main__":
    main()
