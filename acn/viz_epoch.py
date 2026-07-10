"""Per-epoch consensus visualization: 4 test samples, each showing the input
image + a grid of colored columns (one per column) evolving through the settle
rounds. Generated automatically at the end of every training epoch.

Each column box is colored by the digit it currently predicts:
  0=blue 1=red 2=green 3=orange 4=purple 5=brown 6=pink 7=gray 8=yellow-green 9=cyan
Brightness = confidence. As rounds progress, columns converge from disagreement
to consensus — the "agents reaching agreement" made visible.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIGIT_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


DIGIT_RGB = [_hex_to_rgb(c) for c in DIGIT_COLORS]


@torch.no_grad()
def _decode_per_column(model, h1, active):
    """The trained per-column readout: D(h1_i) -> softmax probs, one per column.

    With the per-column readout this is the real, in-distribution decode (not
    an OOD isolation). Returns (B, N, C) softmax probabilities.
    """
    logits = model.column_decoder(h1)              # (B, N, C)
    return F.softmax(logits, dim=-1)


@torch.no_grad()
def _fused_prediction(model, h1, active):
    """Global prediction = mean of per-column softmaxes over active columns (B, C)."""
    probs = _decode_per_column(model, h1, active)             # (B, N, C)
    w = active.unsqueeze(-1)
    return (w * probs).sum(1) / w.sum(1).clamp(min=1e-6)      # (B, C)


@torch.no_grad()
def _record_settle(model, state0, ctx, sc, k_max, alpha):
    """Run the settle, recording per-column probs + the fused prediction at each round."""
    from acn.energy import state_grad
    s = state0.clone_detach()
    records = [(_decode_per_column(model, s.h1, ctx.active)[0].cpu(),
                _fused_prediction(model, s.h1, ctx.active)[0].cpu())]
    for _ in range(k_max):
        grads, _ = state_grad(s, ctx, sc, beta=0.0, target=None)
        s.h1 = s.h1 - alpha * grads["h1"] * ctx.active.unsqueeze(-1)
        s.llogits = s.llogits - alpha * grads["llogits"]
        records.append((_decode_per_column(model, s.h1, ctx.active)[0].cpu(),
                        _fused_prediction(model, s.h1, ctx.active)[0].cpu()))
    return records


def _column_grid_shape(N):
    n_cols = int(math.isqrt(N))
    n_rows = (N + n_cols - 1) // n_cols
    return n_rows, n_cols


def make_epoch_consensus_gif(model, test_x, test_y, indices, path, k_max, alpha,
                             epoch, device):
    """Generate a single GIF with 4 test samples (2x2 layout).

    Each panel shows: [left] the input image, [right] the grid of column boxes
    colored by their predicted digit, evolving through the settle rounds.
    """
    n_samples = len(indices)
    n_rows_grid, n_cols_grid = _column_grid_shape(model._N)

    # collect per-sample frame data
    all_frames = []   # list of length n_samples; each is list of ((N, C) probs, (C,) fused)
    labels = []
    images = []
    actives = []
    for idx in indices:
        x = test_x[idx:idx+1].to(device)
        y = test_y[idx:idx+1].to(device)
        ctx, sc, state0 = model._prepare(x)
        frames = _record_settle(model, state0, ctx, sc, k_max, alpha)
        all_frames.append(frames)
        labels.append(test_y[idx].item())
        images.append(test_x[idx, 0].cpu().numpy())
        actives.append(ctx.active[0].cpu())

    # layout: 2x2 grid of samples; each sample = [image | column grid]
    # right side: a persistent color legend (digit -> color)
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.25], hspace=0.25, wspace=0.05)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    ax_leg = fig.add_subplot(gs[:, 2])  # legend spans both rows

    # draw the legend once (it doesn't change between frames)
    for d in range(10):
        y_pos = 9 - d  # top = 9, bottom = 0
        rect = plt.Rectangle((0, y_pos), 0.9, 0.9,
                             facecolor=DIGIT_COLORS[d], edgecolor="white", lw=1)
        ax_leg.add_patch(rect)
        ax_leg.text(1.1, y_pos + 0.45, str(d), ha="left", va="center",
                    fontsize=12, fontweight="bold", color="white")
    ax_leg.set_xlim(-0.1, 2.5)
    ax_leg.set_ylim(-0.5, 10.5)
    ax_leg.set_aspect("equal")
    ax_leg.axis("off")
    ax_leg.set_title("digit\n→ color", fontsize=9, fontweight="bold")

    def render_frame(frame_idx):
        for s in range(min(n_samples, 4)):
            ax = axes[s]
            ax.clear()
            # remove any inset axes left from the previous frame (matplotlib keeps
            # them; ax.clear() does NOT remove child inset axes, so they'd stack up)
            for child in list(ax.child_axes):
                child.remove()
            # left: the input digit image
            ax_in = ax.inset_axes([0.0, 0.0, 0.35, 1.0])
            ax_in.imshow(images[s], cmap="gray")
            ax_in.set_title(f"input (label {labels[s]})", fontsize=9)
            ax_in.axis("off")
            # right: the column grid
            ax_g = ax.inset_axes([0.40, 0.0, 0.60, 1.0])
            probs, fused = all_frames[s][frame_idx]   # (N, C), (C,)
            act = actives[s]
            fused_pred = int(fused.argmax().item())
            fused_conf = float(fused[fused_pred].item())
            correct = (fused_pred == labels[s])
            for ci in range(len(probs)):
                r, c = ci // n_cols_grid, ci % n_cols_grid
                if act[ci] < 0.5:
                    rect = plt.Rectangle((c, n_rows_grid - 1 - r), 0.9, 0.9,
                                         facecolor="#222222", edgecolor="#444", lw=0.3)
                    ax_g.add_patch(rect)
                    continue
                pred = probs[ci].argmax().item()
                conf = probs[ci][pred].item()
                rgb = DIGIT_RGB[pred]
                bright = 0.3 + 0.7 * conf
                color = (rgb[0]*bright, rgb[1]*bright, rgb[2]*bright)
                rect = plt.Rectangle((c, n_rows_grid-1-r), 0.9, 0.9,
                                     facecolor=color, edgecolor="white", lw=0.5)
                ax_g.add_patch(rect)
                # no digit text — color only (brightness = confidence)
            ax_g.set_xlim(-0.1, n_cols_grid+0.1)
            ax_g.set_ylim(-0.1, n_rows_grid+0.1)
            ax_g.set_aspect("equal")
            ax_g.axis("off")
            ax_g.set_title(f"columns @ round {frame_idx}  |  fused → {fused_pred} ({fused_conf:.0%}) {'✓' if correct else '✗'}"
                           + ("  FINAL" if frame_idx >= k_max else ""), fontsize=8)
            ax.axis("off")
        fig.suptitle(f"Epoch {epoch} — consensus over {k_max} settle rounds",
                     fontsize=13, fontweight="bold")

    # Save frames manually with PIL so we control per-frame duration and the
    # linger at the final frame (PillowWriter collapses identical frames; PIL doesn't).
    from PIL import Image as PILImage
    frame_imgs = []
    for fi in range(k_max + 1):
        render_frame(fi)
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        w, h = fig.canvas.get_width_height()
        frame_imgs.append(PILImage.frombytes("RGB", (w, h), buf))
    # linger: append the final frame 5 more times with a longer duration
    frame_imgs.append(frame_imgs[-1])  # duplicate; PIL handles durations per-frame
    frame_imgs.append(frame_imgs[-1])
    frame_imgs.append(frame_imgs[-1])
    frame_imgs.append(frame_imgs[-1])
    frame_imgs.append(frame_imgs[-1])
    # durations: 400ms per round frame, 2000ms (2s) per linger frame
    durations = [400] * (k_max + 1) + [2000] * 5
    frame_imgs[0].save(str(path), save_all=True, append_images=frame_imgs[1:],
                       duration=durations, loop=0, optimize=False)
    plt.close(fig)
