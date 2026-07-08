"""ACN — Decision Confidence Visualization.

A per-sample animated GIF showing the model's DECISION CONFIDENCE as a simple
2x5 grid of boxes (digits 0-9):

  1. Input digit  (ground truth + ✓/✗ tag)
  2. A 2x5 grid where each cell = one digit (0-4 on top, 5-9 on bottom):
       * blank (black) if the model's softmax confidence for that digit is < 50%
         (it's not confident enough to be part of the decision)
       * colored with that digit's label color if confidence >= 50%, with
         brightness rising from dim (at 50%) to full bright (at 100%) — so the
         STRENGTH OF THE COLOR IS THE CONFIDENCE.
     The model's predicted digit (argmax) always gets a green border so the
     decision is visible even when nothing crosses 50%. The percentage is
     printed in each lit cell.

As the consensus rounds animate, a digit crosses 50% and its cell brightens —
you watch the decision form. The bottom/top column grids and the weighted-vote
bars are intentionally NOT shown; the 2x5 confidence grid is the whole story.

Two entry points (signatures unchanged so scripts/viz_spotlight.py still works):
  * :func:`make_spotlight_gif`            — one row per digit class (0-9), batched 4/batch
  * :func:`make_robustness_spotlight_gif` — one digit under 8 corruptions, batched 4/batch
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import PillowWriter


class _PlayOncePillowWriter(PillowWriter):
    """PillowWriter that writes loop=1 (play once, then hold the last frame)
    instead of the default loop=0 (infinite loop).

    So the GIF plays the consensus rounds once, lingers on the final converged
    state, and stops there — it does NOT restart. Reopening the file replays it.
    """
    def finish(self):
        self._frames[0].save(
            self.outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=1)

# 10 distinct colors for digits 0-9
DIGIT_COLORS = [
    "#ef4444", "#3b82f6", "#22c55e", "#f97316", "#a855f7",
    "#06b6d4", "#eab308", "#ec4899", "#78350f", "#6b7280",
]
DIGIT_COLORS_RGB = np.array([
    [0.937, 0.247, 0.247],  # 0 red
    [0.231, 0.510, 0.965],  # 1 blue
    [0.133, 0.773, 0.369],  # 2 green
    [0.976, 0.451, 0.094],  # 3 orange
    [0.659, 0.333, 0.890],  # 4 purple
    [0.024, 0.714, 0.831],  # 5 cyan
    [0.918, 0.702, 0.031],  # 6 yellow
    [0.925, 0.282, 0.600],  # 7 pink
    [0.471, 0.208, 0.059],  # 8 brown
    [0.420, 0.439, 0.502],  # 9 gray
], dtype=np.float32)


# ════════════════════════════════════════════════════════════════════
# V3-specific snapshot: extract everything the GIF needs (both layers)
# ════════════════════════════════════════════════════════════════════

class V3Snapshot:
    """Holds all interpretable state from a ACN forward pass, both layers.

    Decodes per-round per-column logits for BOTH the bottom layer and the
    abstract layer so the grids can animate over consensus rounds.
    """
    def __init__(self, model, images, record=True):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            pred, (state_bottom, state_abstract) = model(images, record=record)

        device = next(model.parameters()).device

        self.pred = pred.cpu().numpy()                       # (B, 10)
        self.pred_class = pred.argmax(-1).cpu().numpy()      # (B,)

        # ── bottom layer ──
        self.active = state_bottom.active.cpu().numpy()      # (B, N)
        self.N = state_bottom.active.shape[1]

        # per-round bottom history (decoded into logits + argmax + active)
        self.bottom_logits_round = None      # (R, B, N, C)
        self.bottom_choices_round = None     # (R, B, N)
        self.bottom_active_round = None      # (R, B, N)
        self.bottom_fused_round = None       # (R, B, C)
        if state_bottom.history:
            logits_list = []
            active_list = []
            for h in state_bottom.history:
                z_t = h["z"].to(device).float()
                lg = model.bottom_decoder(z_t).detach().cpu().numpy()  # (B, N, C)
                logits_list.append(lg)
                active_list.append(h["active"].cpu().numpy())          # (B, N)
            self.bottom_logits_round = np.stack(logits_list)           # (R, B, N, C)
            self.bottom_active_round = np.stack(active_list)           # (R, B, N)
            self.bottom_choices_round = self.bottom_logits_round.argmax(-1)  # (R, B, N)
            self.bottom_fused_round = _sparse_fuse_np(
                self.bottom_logits_round, self.bottom_active_round)    # (R, B, C)

        # final bottom (for static fallback if no history)
        self.logits_bottom = model.bottom_decoder(state_bottom.z).detach().cpu().numpy()
        self.bottom_choices = self.logits_bottom.argmax(-1)            # (B, N)
        self.bottom_fused = _sparse_fuse_np(
            self.logits_bottom[None], self.active[None])[0]            # (B, C)

        # ── abstract layer ──
        self.has_abstract = state_abstract is not None
        if self.has_abstract:
            self.active2 = state_abstract.active.cpu().numpy()         # (B, N2)
            self.N2 = state_abstract.active.shape[1]
            self.logits_abstract = model.abstract_decoder(
                state_abstract.z).detach().cpu().numpy()               # (B, N2, C)
            self.abstract_choices = self.logits_abstract.argmax(-1)    # (B, N2)
            self.abstract_fused = _sparse_fuse_np(
                self.logits_abstract[None], self.active2[None])[0]     # (B, C)

            self.abstract_logits_round = None
            self.abstract_choices_round = None
            self.abstract_active_round = None
            self.abstract_fused_round = None
            if state_abstract.history:
                logits_list = []
                active_list = []
                for h in state_abstract.history:
                    z_t = h["z"].to(device).float()
                    lg = model.abstract_decoder(z_t).detach().cpu().numpy()
                    logits_list.append(lg)
                    active_list.append(h["active"].cpu().numpy())
                self.abstract_logits_round = np.stack(logits_list)     # (R, B, N2, C)
                self.abstract_active_round = np.stack(active_list)     # (R, B, N2)
                self.abstract_choices_round = self.abstract_logits_round.argmax(-1)
                self.abstract_fused_round = _sparse_fuse_np(
                    self.abstract_logits_round, self.abstract_active_round)  # (R, B, C)
        else:
            self.active2 = None
            self.N2 = 0
            self.logits_abstract = None
            self.abstract_choices = None
            self.abstract_fused = np.zeros_like(self.bottom_fused)
            self.abstract_logits_round = None
            self.abstract_choices_round = None
            self.abstract_active_round = None
            self.abstract_fused_round = None

        # combined fused per round. In the unified predictive-coding hierarchy both
        # layers run the SAME number of rounds (hierarchy_rounds) in lockstep, so
        # the per-round histories are already aligned. Stored as a diagnostic sum
        # (bottom mean + top mean); the model's actual readout is the
        # confidence-weighted cooperative vote of the two layers' readouts.
        if self.bottom_fused_round is not None:
            abs_aligned = (self.abstract_fused_round if self.abstract_fused_round is not None
                           else np.zeros_like(self.bottom_fused_round))
            self.fused_round = abs_aligned + self.bottom_fused_round   # (R, B, C) diagnostic
        else:
            self.fused_round = None

        # final combined fused
        self.fused_final = self.bottom_fused + self.abstract_fused      # (B, C)

        # roster / sizes (for the input-panel spotlight overlay, if wanted)
        b_topo = model._bottom_topo
        self.roster = b_topo.roster
        self.coords = b_topo.coords.cpu().numpy()
        if self.roster is not None:
            self.patch_sizes = np.array([s.size for s in self.roster])
        else:
            self.patch_sizes = np.full(self.N, model.patch_size)

        if was_training:
            model.train()


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def _sparse_fuse_np(logits, active):
    """Numpy sparse fuse: mean over active columns.

    logits: (..., N, C), active: (..., N) -> (..., C)
    """
    w = active[..., None]                          # (..., N, 1)
    summed = (w * logits).sum(axis=-2)             # (..., C)
    count = w.sum(axis=-2).clip(min=1e-6)          # (..., 1)
    return summed / count


def _confidence(pred_logits):
    """Softmax confidence of the predicted class."""
    p = np.exp(pred_logits - pred_logits.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    return p.max(axis=-1)
def _weighted_vote_data(snap, s: int, round_idx: int):
    """Collect the cooperative readout data for one sample at one round.

    The model's verdict is a CONFIDENCE-WEIGHTED vote of the two layers:
        pred = w_top * pred_top + w_bot * pred_bottom
    where w_top/w_bot are the normalized logit margins (top1-top2) of each
    layer's fused readout. This helper returns exactly those pieces so the
    panel can show: each layer's per-digit mean (its vote), each layer's
    confidence weight (a gauge), and the weighted combination (the verdict,
    guaranteed to match snap.pred_class).
    """
    # per-layer fused readouts at this round
    if snap.bottom_fused_round is not None and round_idx < len(snap.bottom_fused_round):
        pred_bottom = snap.bottom_fused_round[round_idx, s]    # (C,)
    else:
        pred_bottom = snap.bottom_fused[s]
    if snap.has_abstract and snap.abstract_fused_round is not None \
            and round_idx < len(snap.abstract_fused_round):
        pred_top = snap.abstract_fused_round[round_idx, s]     # (C,)
    else:
        pred_top = snap.abstract_fused[s]

    # confidence weights = logit margin (top1 - top2), normalized to sum 1
    def _margin(v):
        top2 = np.sort(v)[-2:]
        return max(float(top2[-1] - top2[-2]), 1e-3)
    w_top_raw = _margin(pred_top) if snap.has_abstract else 0.0
    w_bot_raw = _margin(pred_bottom)
    w_sum = w_top_raw + w_bot_raw + 1e-6
    w_top = w_top_raw / w_sum
    w_bot = w_bot_raw / w_sum

    # the real verdict = confidence-weighted combination
    fused = w_top * pred_top + w_bot * pred_bottom            # (C,)
    winner = int(fused.argmax())
    return {"pred_top": pred_top, "pred_bottom": pred_bottom,
            "w_top": w_top, "w_bot": w_bot,
            "fused": fused, "winner": winner}
# ════════════════════════════════════════════════════════════════════
# Figure builder shared by both GIFs
# ════════════════════════════════════════════════════════════════════
def _softmax_np(logits):
    """Softmax over the last axis (numpy). logits: (..., C) -> (..., C)."""
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _confidence_grid_rgb(probs, floor=0.30):
    """Render the 10 digit probabilities as a 2x5 RGB confidence grid.

    Each cell = one digit (row-major: 0..4 on top, 5..9 on bottom).
      * prob < 0.5  -> BLANK (black): the model isn't confident enough in that
        digit for it to be part of the decision.
      * prob >= 0.5 -> the digit's label COLOR, brightness rising from `floor`
        (dim, at 50%) to 1.0 (full bright, at 100%). So a just-over-50% cell is
        a dim version of its color and a 95% cell is full-bright — the strength
        of the color IS the confidence.
    """
    grid = np.zeros((2, 5, 3), dtype=np.float32)
    for d in range(10):
        r, c = d // 5, d % 5
        p = float(probs[d])
        if p >= 0.5:
            b = floor + (1.0 - floor) * ((p - 0.5) / 0.5)
            grid[r, c] = DIGIT_COLORS_RGB[d] * b
    return grid


def _build_figure(nsamp, row_titles, has_abstract):
    """Create the figure + artists for the confidence-grid GIF.

    Layout per row: [input digit | 2x5 confidence grid]. Each of the 10 cells
    is a digit 0-9: blank if the model's softmax confidence for it is < 50%,
    otherwise colored with that digit's label color, dim near 50% and bright
    near 100%. The model's predicted digit (argmax) always gets a green border
    so the decision is visible even when nothing crosses 50%.
    """
    n_panels = 2   # input | confidence grid
    width_ratios = [1.0, 3.0]
    fig = plt.figure(figsize=(9, 2.6 * nsamp + 0.6))
    gs = fig.add_gridspec(nsamp, n_panels, width_ratios=width_ratios)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.04,
                        wspace=0.12, hspace=0.30)

    artists = {"fig": fig, "gs": gs, "has_abstract": has_abstract}

    # ── input digit axes ──
    artists["img_axes"] = []
    artists["img_ims"] = []
    for r in range(nsamp):
        ax = fig.add_subplot(gs[r, 0])
        im = ax.imshow(np.zeros((28, 28)), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(row_titles[r], fontsize=11, pad=4)
        artists["img_axes"].append(ax)
        artists["img_ims"].append(im)

    # ── 2x5 confidence grid axes ──
    artists["conf_axes"] = []
    artists["conf_ims"] = []           # per-sample: imshow (2,5,3) background
    artists["conf_pct_texts"] = []     # per-sample: 10 text (percentage in each cell)
    artists["conf_winner_rects"] = []  # per-sample: green border on the argmax cell
    for r in range(nsamp):
        ax = fig.add_subplot(gs[r, 1])
        im = ax.imshow(np.zeros((2, 5, 3)), interpolation="nearest", vmin=0, vmax=1)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(1.5, -0.5)          # row 0 (digits 0-4) on top
        ax.set_xticks([]); ax.set_yticks([])
        # white cell borders (static) + static digit-number labels
        for d in range(10):
            rr, cc = d // 5, d % 5
            ax.add_patch(plt.Rectangle((cc - 0.5, rr - 0.5), 1, 1, fill=False,
                                       edgecolor="white", linewidth=1.2, zorder=2))
            ax.text(cc, rr, str(d), ha="center", va="center", fontsize=12,
                    color="#71717a", fontweight="bold", zorder=3)
        # percentage text per cell (updated each frame; blank if < 50%)
        pct_texts = []
        for d in range(10):
            rr, cc = d // 5, d % 5
            t = ax.text(cc, rr + 0.30, "", ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold", zorder=4)
            pct_texts.append(t)
        # green winner border (moved each frame to the argmax cell)
        win_rect = mpatches.Rectangle((-0.5, -0.5), 1, 1, fill=False,
                                      edgecolor="#16a34a", linewidth=3.0, zorder=5)
        ax.add_patch(win_rect)
        artists["conf_axes"].append(ax)
        artists["conf_ims"].append(im)
        artists["conf_pct_texts"].append(pct_texts)
        artists["conf_winner_rects"].append(win_rect)
    artists["conf_axes"][0].set_title(
        "Decision confidence  (blank < 50%  \u00b7  color strength = confidence)",
        fontsize=10, pad=6, color="#22d3ee")

    return artists


def _draw_round(artists, snap, images_np, labels_np, round_idx, n_rounds,
                row_titles):
    """Update all artists for one animation frame (one consensus round).

    Each sample: input digit + the 2x5 confidence grid for the model's readout
    at THIS round (softmax of the confidence-weighted fused readout). As
    consensus proceeds, a digit crosses 50% and its cell brightens.
    """
    nsamp = len(images_np)
    for s in range(nsamp):
        artists["img_ims"][s].set_data(images_np[s])

        # the model's real (confidence-weighted) readout at this round -> softmax
        vd = _weighted_vote_data(snap, s, round_idx)
        probs = _softmax_np(vd["fused"])                  # (10,)
        grid = _confidence_grid_rgb(probs)                # (2, 5, 3)
        artists["conf_ims"][s].set_data(grid)

        # percentage text in each lit cell (>= 50%); blank otherwise
        for d in range(10):
            p = float(probs[d])
            artists["conf_pct_texts"][s][d].set_text(f"{p:.0%}" if p >= 0.5 else "")
        # green border on the model's predicted digit (argmax), always visible
        w = vd["winner"]
        wr, wc = w // 5, w % 5
        artists["conf_winner_rects"][s].set_xy((wc - 0.5, wr - 0.5))

    r = min(round_idx + 1, n_rounds)
    artists["fig"].suptitle(
        f"ACN \u2014 Decision Confidence  |  Round {r}/{n_rounds}",
        fontsize=13, color="#22d3ee")


# ════════════════════════════════════════════════════════════════════
# Batched GIF rendering (4 samples per batch so the 3-stack weighted-vote
# panel has enough vertical room — no y-axis overlap)
# ════════════════════════════════════════════════════════════════════

def _render_batch_frames(model, batch_imgs, batch_labels, row_title_fn,
                          linger_frames, suptitle_fn):
    """Render one batch (≤4 samples) to a list of PIL.Image frames.

    Each frame is one consensus round; `linger_frames` copies of the final
    frame are appended so the batch holds on its converged state.

    `row_title_fn(local_idx, pred_class, batch_label) -> str` builds each row's
    title from the PER-BATCH snapshot's prediction, so the ✓/✗ tag always
    matches the grid's winner (the gate is stochastic, so the tag must come
    from the same snapshot as the grid).
    """
    dev = next(model.parameters()).device
    batch_imgs = batch_imgs.to(dev)
    snap = V3Snapshot(model, batch_imgs, record=True)
    nsamp = len(batch_imgs)
    images_np = batch_imgs.cpu().numpy().squeeze()
    labels_np = batch_labels.cpu().numpy() if torch.is_tensor(batch_labels) \
        else np.asarray(batch_labels)
    if images_np.ndim == 2:               # single sample -> add a dim
        images_np = images_np[None]

    # row titles from THIS batch's snapshot (tag consistent with the grid)
    row_titles = [row_title_fn(j, int(snap.pred_class[j]), int(labels_np[j]))
                  for j in range(nsamp)]

    n_rounds = len(snap.bottom_choices_round) if snap.bottom_choices_round is not None else 1
    artists = _build_figure(nsamp, row_titles, snap.has_abstract)

    frames = []
    for k in range(n_rounds):
        _draw_round(artists, snap, images_np, labels_np, k, n_rounds, row_titles)
        artists["fig"].suptitle(suptitle_fn(k, n_rounds), fontsize=13, color="#22d3ee")
        artists["fig"].canvas.draw()
        w, h = artists["fig"].canvas.get_width_height()
        buf = artists["fig"].canvas.buffer_rgba()
        frames.append(Image.frombytes("RGBA", (w, h), bytes(buf)).convert("RGB"))
    for _ in range(linger_frames):
        frames.append(frames[-1].copy())
    plt.close(artists["fig"])
    return frames


def _save_frames_as_gif(frames, path, duration_ms):
    """Save a list of PIL.Image frames to a play-once GIF (loop=1)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(path), save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=1)
# ════════════════════════════════════════════════════════════════════
# Main GIF builder: per-digit spotlight
# ════════════════════════════════════════════════════════════════════

def make_spotlight_gif(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    path: str | Path = "spotlight.gif",
    n_samples: int = 10,
    duration_ms: int = 200,
    linger_frames: int = 15,
    batch_size: int = 4,
) -> None:
    """Create the Thousand-Brains Grid GIF (one row per digit class), BATCHED.

    Samples are split into batches of `batch_size` (default 4) so each batch
    figure has only 4 rows — giving the 3-stack weighted-vote panel enough
    vertical room (no y-axis label overlap). The GIF plays batch 1 → linger →
    batch 2 → linger → ... then holds on the last batch's final frame (loop=1).

    Each row: input digit | bottom grid (confidence-dimmed) | top grid |
    weighted vote (top + bottom → fused, with confidence gauges).
    """
    # pick one sample per digit class if possible
    labels_np_all = labels.cpu().numpy()
    sample_indices = []
    for d in range(10):
        idxs = np.where(labels_np_all == d)[0]
        if len(idxs) > 0:
            sample_indices.append(int(idxs[0]))
        if len(sample_indices) >= n_samples:
            break
    if len(sample_indices) < n_samples:
        remaining = [i for i in range(len(labels_np_all)) if i not in sample_indices]
        sample_indices.extend(remaining[:n_samples - len(sample_indices)])
    sample_indices = sample_indices[:n_samples]

    dev = next(model.parameters()).device
    all_imgs = images[sample_indices].to(dev)
    all_labels = labels[sample_indices]

    # reproducible gate
    torch.manual_seed(42)

    # split into batches of `batch_size`
    batches = [sample_indices[i:i + batch_size] for i in range(0, len(sample_indices), batch_size)]
    n_batches = len(batches)

    def row_title_fn(j, pred, lbl):
        tag = "\u2713" if pred == lbl else "\u2717"
        return f"digit {lbl}  {tag}"

    all_frames = []
    for bi, idxs in enumerate(batches):
        batch_imgs = images[idxs]
        batch_labels = labels[idxs]

        def suptitle(k, R, _bi=bi):
            return (f"ACN \u2014 Decision Confidence  |  "
                    f"batch {_bi + 1}/{n_batches}  |  Round {min(k + 1, R)}/{R}")
        frames = _render_batch_frames(
            model, batch_imgs, batch_labels, row_title_fn,
            linger_frames, suptitle)
        all_frames.extend(frames)

    _save_frames_as_gif(all_frames, path, duration_ms)
    print(f"Spotlight GIF saved: {path}  ({n_batches} batches × ≤{batch_size} samples, "
          f"{len(all_frames)} frames)")


# ════════════════════════════════════════════════════════════════════
# Robustness Spotlight: one digit under 7 corruptions
# ════════════════════════════════════════════════════════════════════

def rotate_batch(X, degrees):
    n = len(X)
    ang = torch.empty(n, device=X.device).uniform_(-degrees, degrees)
    theta = torch.zeros(n, 2, 3, device=X.device)
    rad = ang * (np.pi / 180.0)
    theta[:, 0, 0] = torch.cos(rad); theta[:, 0, 1] = -torch.sin(rad)
    theta[:, 1, 0] = torch.sin(rad); theta[:, 1, 1] = torch.cos(rad)
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
    theta[:, 0, 0] = 1.0; theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = torch.empty(n, device=X.device).uniform_(-frac, frac)
    theta[:, 1, 2] = torch.empty(n, device=X.device).uniform_(-frac, frac)
    grid = F.affine_grid(theta, X.shape, align_corners=False)
    return F.grid_sample(X, grid, align_corners=False, padding_mode="zeros")


def make_robustness_spotlight_gif(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    path: str | Path = "robustness_spotlight.gif",
    sample_idx: int = 0,
    duration_ms: int = 250,
    linger_frames: int = 20,
    batch_size: int = 4,
) -> None:
    """Create the Robustness Grid GIF: one digit under 8 corruptions, BATCHED.

    8 conditions (clean + 7 corruptions) split into 2 batches of 4 so each
    batch figure has only 4 rows (the 3-stack weighted-vote panel gets enough
    vertical room). The GIF plays batch 1 (rotations) → linger → batch 2
    (shifts/occlude/noise) → linger, then holds (loop=1).
    """
    dev = next(model.parameters()).device
    img = images[sample_idx:sample_idx+1].to(dev)
    label = int(labels[sample_idx])

    torch.manual_seed(42)
    conditions = [
        ("clean",   img),
        ("rot15",   rotate_batch(img, 15)),
        ("rot30",   rotate_batch(img, 30)),
        ("rot45",   rotate_batch(img, 45)),
        ("shift20", shift_batch(img, 0.2)),
        ("shift30", shift_batch(img, 0.3)),
        ("occ3×7",  occlude_batch(img, 3, 7)),
        ("noise30", noise_batch(img, 0.3)),
    ]
    cond_names = [c[0] for c in conditions]
    all_imgs = torch.cat([c[1] for c in conditions], dim=0)   # (8, 1, 28, 28)
    all_labels = torch.full((len(conditions),), label, dtype=torch.long)

    # reproducible gate
    torch.manual_seed(42)

    # split the 8 conditions into batches of `batch_size`
    n = len(conditions)
    batches = [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]
    n_batches = len(batches)

    all_frames = []
    for bi, idxs in enumerate(batches):
        batch_imgs = all_imgs[idxs]
        batch_labels = all_labels[idxs]
        cond_names_batch = [cond_names[ci] for ci in idxs]

        def _rtf(j, pred, lbl, _names=cond_names_batch):
            tag = "\u2713" if pred == lbl else "\u2717"
            return f"{_names[j]}  {tag}"

        def suptitle(k, R, _bi=bi):
            return (f"ACN — Robustness Confidence  |  GT: {label}  |  "
                    f"batch {_bi + 1}/{n_batches}  |  Round {min(k + 1, R)}/{R}")
        frames = _render_batch_frames(
            model, batch_imgs, batch_labels, _rtf,
            linger_frames, suptitle)
        all_frames.extend(frames)

    _save_frames_as_gif(all_frames, path, duration_ms)
    print(f"Robustness Grid GIF saved: {path}  ({n_batches} batches × ≤{batch_size}, "
          f"{len(all_frames)} frames)")
