"""ACN — Thousand-Brains Spotlight Visualization.

A per-sample animated GIF showing the full two-layer decision story as it
forms over consensus rounds.  For each sample row we show FOUR panels:

  1. Input digit            — the raw image with the ground-truth label and a
                              ✓/✗ tag (tag taken from the same snapshot as the
                              grids, so it always matches the winner).
  2. Bottom-layer columns   — every bottom-layer column drawn on the image
                              grid at its spatial patch location, colored by
                              the digit that column currently votes for.
                              Brightness = that column's confidence in its
                              choice; inactive (gated-off) columns are dim
                              slate.  So you watch the bottom "thousand brains"
                              settle on a digit across the image.
  3. Top-layer columns      — every abstract-layer column (a small 2×4 grid)
                              colored by its vote the same way.  The top layer
                              is coarser/fewer columns, so it tells a
                              complementary, higher-level story.
  4. Decision = mix         — the informative readout: the bottom layer's
                              fused vote and the top layer's fused vote are
                              shown as two stacked colored bars, each labeled
                              with its confidence WEIGHT (w_bot / w_top).  A
                              "⊕ weighted vote" arrow points to a third bar —
                              the FINAL decision = w_top·top + w_bot·bottom —
                              whose winning digit is outlined in green and
                              labeled with its percentage.  This makes it
                              visually obvious that the last decision
                              confidence is the MIX of the two layers.

At the very bottom of the figure a horizontal DIGIT COLOR LEGEND shows the
ten digit→color mappings so every panel is readable at a glance.

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

# color for an inactive (gated-off) column
_INACTIVE_RGB = np.array([0.10, 0.11, 0.14], dtype=np.float32)
# brightness floor so a just-active column is still visible
_FLOOR = 0.35


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

        # combined fused per round (diagnostic sum; the model's real readout is
        # the confidence-weighted cooperative vote, recomputed per frame).
        if self.bottom_fused_round is not None:
            abs_aligned = (self.abstract_fused_round if self.abstract_fused_round is not None
                           else np.zeros_like(self.bottom_fused_round))
            self.fused_round = abs_aligned + self.bottom_fused_round   # (R, B, C) diagnostic
        else:
            self.fused_round = None

        # final combined fused
        self.fused_final = self.bottom_fused + self.abstract_fused      # (B, C)

        # roster / sizes (for the bottom spatial-grid layout)
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


def _softmax_np(logits):
    """Softmax over the last axis (numpy). logits: (..., C) -> (..., C)."""
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _column_probs_softmax(logits_n_c):
    """Per-column softmax. logits_n_c: (N, C) -> (N, C) probs."""
    return _softmax_np(logits_n_c)


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
    mix panel can show: each layer's per-digit vote (a stacked color bar),
    each layer's confidence weight, and the weighted combination (the verdict,
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


def _cell_color(choice, active):
    """RGB color for one column cell given its digit choice and active flag.

    Active -> solid digit color (no confidence dimming; the color IS the vote).
    Inactive -> dark slate (column is present but not voting this round).
    """
    if not active:
        return _INACTIVE_RGB
    return DIGIT_COLORS_RGB[int(choice)]


def _top_grid_shape(n2):
    """Pick a (rows, cols) grid shape for n2 abstract columns."""
    if n2 <= 4:
        return 1, n2
    if n2 <= 9:
        return 2, int(np.ceil(n2 / 2))
    return int(np.ceil(np.sqrt(n2))), int(np.ceil(n2 / np.ceil(np.sqrt(n2))))


# ════════════════════════════════════════════════════════════════════
# Figure builder shared by both GIFs
# ════════════════════════════════════════════════════════════════════

def _build_figure(nsamp, row_titles, snap):
    """Create the figure + artists for the three-panel spotlight GIF.

    Layout per row (3 panels):
        [input digit | bottom columns grid | top columns grid]
    Plus a VERTICAL digit-color legend pinned to the right side of the figure.

    All columns are shown (even inactive). Active columns are colored by their
    predicted digit (solid color, no confidence dimming). Inactive columns are
    dark slate (present but not voting).
    """
    has_abstract = snap.has_abstract
    n_bottom = snap.N
    coords = snap.coords                       # (N, 2) row,col in image px
    sizes = snap.patch_sizes                   # (N,)
    # draw largest patches first so smaller (finer) columns land on top
    draw_order = np.argsort(-sizes, kind="stable")

    n_top = snap.N2 if has_abstract else 0
    top_rows, top_cols = _top_grid_shape(n_top) if n_top > 0 else (1, 1)

    n_panels = 3
    width_ratios = [1.0, 1.6, 0.9 + 0.18 * top_cols]
    # taller top margin for small batches so the suptitle doesn't crash into
    # the panel titles. Reserve a fixed ~0.7 inch header regardless of nsamp.
    fig_h = 2.95 * nsamp + 1.2
    fig = plt.figure(figsize=(12.5, fig_h), facecolor="white")
    top_margin = 1.0 - 0.7 / fig_h          # fraction of fig reserved at top
    gs = fig.add_gridspec(nsamp, n_panels, width_ratios=width_ratios)
    # panels use right=0.78 so the legend (starting at 0.80) has clear separation
    fig.subplots_adjust(left=0.03, right=0.78, top=top_margin, bottom=0.04,
                        wspace=0.18, hspace=0.32)

    artists = {"fig": fig, "gs": gs, "has_abstract": has_abstract,
               "n_bottom": n_bottom, "n_top": n_top,
               "draw_order": draw_order, "coords": coords, "sizes": sizes,
               "top_rows": top_rows, "top_cols": top_cols}

    # ── panel 0: input digit ──
    artists["img_axes"] = []
    artists["img_ims"] = []
    for r in range(nsamp):
        ax = fig.add_subplot(gs[r, 0])
        im = ax.imshow(np.zeros((28, 28)), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(row_titles[r], fontsize=11, pad=4)
        artists["img_axes"].append(ax)
        artists["img_ims"].append(im)

    # ── panel 1: bottom-layer columns on the image grid ──
    artists["bot_axes"] = []
    artists["bot_rects"] = []        # per sample: list[N] of Rectangle (in draw_order)
    for r in range(nsamp):
        ax = fig.add_subplot(gs[r, 1])
        ax.set_xlim(-0.5, 28.5)
        ax.set_ylim(28.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        rects = []
        for idx in draw_order:
            idx = int(idx)
            row, col = int(coords[idx, 0]), int(coords[idx, 1])
            sz = int(sizes[idx])
            rect = mpatches.Rectangle((col, row), sz, sz,
                                      facecolor=_INACTIVE_RGB,
                                      edgecolor="#e4e4e7", linewidth=0.4,
                                      alpha=0.9, zorder=3)
            ax.add_patch(rect)
            rects.append(rect)
        artists["bot_rects"].append(rects)
        artists["bot_axes"].append(ax)
    artists["bot_axes"][0].set_title(
        "Bottom-layer columns\n(color = predicted digit)",
        fontsize=10, pad=6, color="black")

    # ── panel 2: top-layer columns in a small grid ──
    artists["top_axes"] = []
    artists["top_rects"] = []        # per sample: list[N2] of Rectangle
    for r in range(nsamp):
        ax = fig.add_subplot(gs[r, 2])
        ax.set_xlim(-0.5, top_cols - 0.5)
        ax.set_ylim(top_rows - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        rects = []
        for i in range(n_top):
            rr, cc = i // top_cols, i % top_cols
            rect = mpatches.Rectangle((cc - 0.5, rr - 0.5), 1, 1,
                                      facecolor=_INACTIVE_RGB,
                                      edgecolor="#e4e4e7", linewidth=1.0,
                                      alpha=0.95, zorder=3)
            ax.add_patch(rect)
            rects.append(rect)
        if not has_abstract:
            ax.text(0, 0, "no abstract\nlayer", ha="center", va="center",
                    fontsize=10, color="#71717a")
        artists["top_rects"].append(rects)
        artists["top_axes"].append(ax)
    artists["top_axes"][0].set_title(
        "Top-layer columns\n(higher-level vote)", fontsize=10, pad=6,
        color="black")

    # ── right-side VERTICAL digit-color legend ──
    # Exactly 10 swatches (digits 0-9), each colored with its digit color
    # and labeled with the digit number centered on the swatch. Nothing else.
    legend_ax = fig.add_axes([0.82, 0.04, 0.16, top_margin - 0.06])
    legend_ax.set_xlim(-0.5, 1.2)
    legend_ax.set_ylim(-0.5, 9.5)
    legend_ax.set_xticks([]); legend_ax.set_yticks([])
    legend_ax.set_facecolor("white")
    for spine in legend_ax.spines.values():
        spine.set_visible(False)
    for d in range(10):
        swatch = mpatches.Rectangle((-0.4, d - 0.4), 0.8, 0.8,
                                    facecolor=DIGIT_COLORS_RGB[d],
                                    edgecolor="#e4e4e7", linewidth=0.6)
        legend_ax.add_patch(swatch)
        legend_ax.text(0.0, d, str(d), ha="center", va="center",
                       fontsize=14, color="white", fontweight="bold")
    artists["legend_ax"] = legend_ax
    artists["top_margin"] = top_margin

    return artists


def _update_col_grid(rects, choices_n, active_n):
    """Update a column-grid's Rectangle facecolors for one frame.

    rects: list of matplotlib Rectangles (length N).
    choices_n: (N,) int digit choice per column.
    active_n: (N,) bool/float active mask.
    Active columns get solid digit color; inactive get dark slate.
    """
    n = len(rects)
    for i in range(n):
        act = bool(active_n[i])
        ch = int(choices_n[i])
        rects[i].set_facecolor(_cell_color(ch, act))


def _draw_round(artists, snap, images_np, labels_np, round_idx, n_rounds,
                row_titles):
    """Update all artists for one animation frame (one consensus round).

    Each sample: input digit + bottom columns grid + top columns grid.
    Active columns are colored by their predicted digit (solid color, no
    confidence dimming); inactive columns are dark slate (present but not
    voting). All columns are shown every round.
    """
    nsamp = len(images_np)
    has_abstract = artists["has_abstract"]
    draw_order = artists["draw_order"]

    for s in range(nsamp):
        # input digit
        artists["img_ims"][s].set_data(images_np[s])

        # bottom column grid at this round
        if snap.bottom_choices_round is not None and round_idx < len(snap.bottom_choices_round):
            b_choices = snap.bottom_choices_round[round_idx, s]      # (N,)
            b_active = snap.bottom_active_round[round_idx, s]        # (N,)
        else:
            b_choices = snap.bottom_choices[s]
            b_active = snap.active[s]
        # bot_rects are in draw_order; index back to column index
        rects_ordered = artists["bot_rects"][s]
        for k, idx in enumerate(draw_order):
            idx = int(idx)
            act = bool(b_active[idx])
            ch = int(b_choices[idx])
            rects_ordered[k].set_facecolor(_cell_color(ch, act))

        # top column grid at this round
        if has_abstract and snap.abstract_choices_round is not None \
                and round_idx < len(snap.abstract_choices_round):
            t_choices = snap.abstract_choices_round[round_idx, s]    # (N2,)
            t_active = snap.abstract_active_round[round_idx, s]      # (N2,)
        elif has_abstract:
            t_choices = snap.abstract_choices[s]
            t_active = snap.active2[s]
        else:
            t_choices = np.zeros(0, dtype=int)
            t_active = np.zeros(0)
        if has_abstract:
            _update_col_grid(artists["top_rects"][s], t_choices, t_active)

    # NOTE: suptitle is set by the caller (_render_batch_frames), not here,
    # to avoid double/overlapping titles.


# ════════════════════════════════════════════════════════════════════
# Batched GIF rendering (4 samples per batch so the four-panel layout has
# enough vertical room per row)
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

    # row titles from THIS batch's snapshot (tag consistent with the grids)
    row_titles = [row_title_fn(j, int(snap.pred_class[j]), int(labels_np[j]))
                  for j in range(nsamp)]

    n_rounds = len(snap.bottom_choices_round) if snap.bottom_choices_round is not None else 1
    artists = _build_figure(nsamp, row_titles, snap)

    frames = []
    for k in range(n_rounds):
        _draw_round(artists, snap, images_np, labels_np, k, n_rounds, row_titles)
        artists["fig"].suptitle(suptitle_fn(k, n_rounds), fontsize=13, color="black")
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
    """Create the Thousand-Brains Spotlight GIF (one row per digit class), BATCHED.

    Samples are split into batches of `batch_size` (default 4) so each batch
    figure has only 4 rows — giving the four-panel layout enough vertical
    room. The GIF plays batch 1 → linger → batch 2 → linger → ... then holds
    on the last batch's final frame (loop=1).

    Each row: input digit | bottom columns grid | top columns grid |
    decision mix (bottom + top → weighted final, with the winner highlighted).
    A horizontal digit-color legend sits at the bottom of every frame.
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
            return (f"ACN \u2014 Thousand-Brains Spotlight  |  "
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
    """Create the Robustness Spotlight GIF: one digit under 8 corruptions, BATCHED.

    8 conditions (clean + 7 corruptions) split into 2 batches of 4 so each
    batch figure has only 4 rows. The GIF plays batch 1 (rotations) → linger
    → batch 2 (shifts/occlude/noise) → linger, then holds (loop=1).
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
            return (f"ACN — Robustness Spotlight  |  GT: {label}  |  "
                    f"batch {_bi + 1}/{n_batches}  |  Round {min(k + 1, R)}/{R}")
        frames = _render_batch_frames(
            model, batch_imgs, batch_labels, _rtf,
            linger_frames, suptitle)
        all_frames.extend(frames)

    _save_frames_as_gif(all_frames, path, duration_ms)
    print(f"Robustness Spotlight GIF saved: {path}  ({n_batches} batches × ≤{batch_size}, "
          f"{len(all_frames)} frames)")
