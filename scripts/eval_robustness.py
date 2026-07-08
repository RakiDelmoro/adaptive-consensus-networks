"""Distribution-shift evaluation for ACN: how robust is the model to corrupted inputs?

Runs the trained ACN model on the MNIST test set under several corruption conditions
(clean, rotation, occlusion, noise, shift) and reports accuracy for each.

    python scripts/eval_robustness.py
    python scripts/eval_robustness.py --n-eval 2000      # faster subset
    python scripts/eval_robustness.py --override model.topk_k=12
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from acn.config import get_preset
from acn.model import AdaptiveConsensusNetwork


# ─── fast math ─────────────────────────────────────────────────────── #
def enable_cuda_fast_math() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except AttributeError:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# ─── corruption functions ──────────────────────────────────────────── #
def rotate_batch(X, degrees):
    n = len(X)
    ang = torch.empty(n, device=X.device).uniform_(-degrees, degrees)
    theta = torch.zeros(n, 2, 3, device=X.device)
    rad = ang * (np.pi / 180.0)
    theta[:, 0, 0] = torch.cos(rad); theta[:, 0, 1] = -torch.sin(rad)
    theta[:, 1, 0] = torch.sin(rad);  theta[:, 1, 1] = torch.cos(rad)
    grid = F.affine_grid(theta, X.shape, align_corners=False)
    return F.grid_sample(X, grid, align_corners=False, padding_mode="zeros")


def occlude_batch(X, n_boxes=3, box_size=7):
    out = X.clone()
    H, W = X.shape[-2], X.shape[-1]
    for _ in range(n_boxes):
        r = torch.randint(0, H - box_size + 1, (len(X),), device=X.device)
        c = torch.randint(0, W - box_size + 1, (len(X),), device=X.device)
        for b in range(len(X)):
            out[b, :, r[b]:r[b]+box_size, c[b]:c[b]+box_size] = 0.0
    return out


def noise_batch(X, sigma=0.3):
    return (X + sigma * torch.randn_like(X)).clamp(0, 1)


def shift_batch(X, frac=0.2):
    n = len(X)
    theta = torch.zeros(n, 2, 3, device=X.device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = torch.empty(n, device=X.device).uniform_(-frac, frac)
    theta[:, 1, 2] = torch.empty(n, device=X.device).uniform_(-frac, frac)
    grid = F.affine_grid(theta, X.shape, align_corners=False)
    return F.grid_sample(X, grid, align_corners=False, padding_mode="zeros")


# ─── evaluation ────────────────────────────────────────────────────── #
@torch.no_grad()
def evaluate_condition(model, X, y, bs=256):
    correct, total = 0, 0
    for i in range(0, len(X), bs):
        xb, yb = X[i:i+bs], y[i:i+bs]
        pred = model(xb)[0].argmax(-1)
        correct += (pred == yb).sum().item()
        total += len(yb)
    return correct / total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="mnist")
    p.add_argument("--n-eval", type=int, default=10000)
    p.add_argument("--rotations", nargs="+", type=float, default=[15, 30, 45])
    p.add_argument("--noise-sigma", type=float, default=0.3)
    p.add_argument("--shift-frac", type=float, default=0.2)
    p.add_argument("--occlude-boxes", type=int, default=3)
    p.add_argument("--occlude-size", type=int, default=7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-viz", dest="viz", action="store_true", default=False)
    p.add_argument("--override", nargs="*", default=[])
    args = p.parse_args()

    enable_cuda_fast_math()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = get_preset(args.config)
    for ov in args.override:
        k, v = ov.split("=", 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        cfg = cfg.override(**{k: v})

    # load raw MNIST test set
    print(f"loading MNIST test set...")
    ds = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    Xte = torch.stack([ds[i][0] for i in range(len(ds))]).to(dev)
    yte = torch.tensor([ds[i][1] for i in range(len(ds))]).to(dev)

    if args.n_eval < len(Xte):
        Xte = Xte[:args.n_eval]; yte = yte[:args.n_eval]
    print(f"evaluating on {len(Xte)} test samples\n")

    # load model — ACN needs data_cfg for multi-scale decomposition
    print("loading ACN model checkpoint...")
    model = AdaptiveConsensusNetwork(
        cfg.data.image_size, cfg.data.patch_size, cfg.data.stride,
        cfg.model, data_cfg=cfg.data
    ).to(dev)
    ckpt = Path(cfg.train.save_dir) / cfg.name / "model_best.pt"
    print(f"checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location=dev)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    # ACN training uses raw [0,1] images (no normalization) — match that
    def norm(X): return X

    # build conditions
    conditions = [("clean", lambda x: x)]
    for deg in args.rotations:
        conditions.append((f"rot{int(deg)}", lambda x, d=deg: rotate_batch(x, d)))
    conditions.append((f"shift{int(args.shift_frac*100)}",
                       lambda x: shift_batch(x, args.shift_frac)))
    conditions.append((f"occ{args.occlude_boxes}x{args.occlude_size}",
                       lambda x: occlude_batch(x, args.occlude_boxes, args.occlude_size)))
    conditions.append((f"noise{int(args.noise_sigma*100)}",
                       lambda x: noise_batch(x, args.noise_sigma)))

    cond_names = [c[0] for c in conditions]
    header = f"{'condition':<14} " + " ".join(f"{c:>10}" for c in cond_names)
    print(header)
    print("-" * len(header))

    torch.manual_seed(0)
    accs = []
    for name, fn in conditions:
        torch.manual_seed(0)
        Xc = norm(fn(Xte))
        acc = evaluate_condition(model, Xc, yte)
        accs.append(acc)

    print(f"{'accuracy':<14} " + " ".join(f"{a*100:>9.1f}%" for a in accs))
    print()
    base = accs[0]
    deltas = [accs[i] - base for i in range(len(accs))]
    print(f"{'Δ vs clean':<14} " + " ".join(
        f"{'+' if d>=0 else ''}{d*100:>8.1f}" for d in deltas))
    print()
    print(f"clean accuracy: {accs[0]*100:.2f}%")
    print(f"avg corrupted:  {np.mean(accs[1:])*100:.2f}%")
    worst_idx = 1 + accs[1:].index(min(accs[1:]))
    worst_name = cond_names[worst_idx]
    print(f"worst case:     {min(accs[1:])*100:.2f}%  ({worst_name})")
    print()
    print("small Δ = model degrades gracefully (robust); large negative Δ = brittle")


if __name__ == "__main__":
    main()
