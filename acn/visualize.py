"""Visualizations for ACN interpretable state.

All functions take a :class:`acn.inspect.StateSnapshot` (or list of them) and a path,
and write a PNG. Kept dependency-light (matplotlib only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from acn.inspect import StateSnapshot, conductance_matrix


# --------------------------------------------------------------------------- #
# Consensus-grid visualization (Sakana sheaf-ADMM style)
# --------------------------------------------------------------------------- #

# 10 distinct colors for digits 0-9. 0=red, 1=blue, 2=green per request.
DIGIT_COLORS = [
    (1.00, 0.20, 0.20),  # 0 red
    (0.20, 0.40, 1.00),  # 1 blue
    (0.20, 0.80, 0.30),  # 2 green
    (1.00, 0.60, 0.10),  # 3 orange
    (0.70, 0.20, 0.90),  # 4 purple
    (0.10, 0.80, 0.80),  # 5 cyan
    (0.95, 0.85, 0.20),  # 6 yellow
    (0.85, 0.40, 0.70),  # 7 pink
    (0.45, 0.25, 0.15),  # 8 brown
    (0.50, 0.50, 0.55),  # 9 grey
]


def digit_color_grid(preds: np.ndarray) -> np.ndarray:
    """Map (Hg, Wg) int predictions -> (Hg, Wg, 3) RGB color grid."""
    Hg, Wg = preds.shape
    rgb = np.zeros((Hg, Wg, 3))
    for v in range(10):
        rgb[preds == v] = DIGIT_COLORS[v]
    return rgb


def decode_history_predictions(model, snap: StateSnapshot, return_confidence: bool = False,
                               return_logits: bool = False):
    """Decode per-round per-node logits from snap.history.

    Returns (R, B, N) int array of argmax digit per neuron per round, where R is
    the number of recorded consensus rounds. Uses the model's decoder on each
    round's z (no grad). Requires snap.history (record=True during snapshot).

    If return_confidence=True, also returns (R, B, N) float array of confidence
    per neuron per round (in [0,1]). We use the normalized LOGIT MARGIN
    (top1 - top2) instead of max-softmax because softmax saturates (~0.9-0.998
    for every cell, no usable range). Margin has 1600x range: a blank patch has
    near-zero margin (no clear winner), a stroke patch has a large margin. We
    normalize per-batch by the max margin so the strongest cell -> 1.0.

    If return_logits=True, also returns (R, B, N, C) float array of the RAW
    per-column logits per round. Used by the fused-scorecard bar chart in
    make_consensus_gif so viewers can see the soft scorecard that actually
    drives the judgment (not just each column's argmax color).
    """
    import torch
    if snap.history is None:
        raise ValueError("snapshot has no history; re-run with record=True")
    R = len(snap.history)
    B, N, d = snap.z.shape
    C = model.decoder.mlp[-1].out_features
    device = next(model.parameters()).device
    preds = np.zeros((R, B, N), dtype=np.int64)
    confs = np.zeros((R, B, N), dtype=np.float32) if return_confidence else None
    logits_all = np.zeros((R, B, N, C), dtype=np.float32) if return_logits else None
    with torch.no_grad():
        for k, h in enumerate(snap.history):
            z = torch.from_numpy(h["z"]).to(device).float()
            logits = model.decoder(z)              # (B, N, C)
            preds[k] = logits.argmax(-1).cpu().numpy()
            if return_confidence:
                top2 = torch.topk(logits, 2, dim=-1).values  # (B, N, 2)
                margin = (top2[..., 0] - top2[..., 1]).clamp(min=0)  # (B, N)
                mx = margin.amax(dim=1, keepdim=True) + 1e-6
                confs[k] = (margin / mx).cpu().numpy()
            if return_logits:
                logits_all[k] = logits.cpu().numpy()
    if return_confidence and return_logits:
        return preds, confs, logits_all
    if return_logits:
        return preds, logits_all
    if return_confidence:
        return preds, confs
    return preds


def plot_conductance_heatmap(snap: StateSnapshot, path: str | Path, title: str = "Conductance") -> None:
    M = conductance_matrix(snap)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("mini network j"); ax.set_ylabel("mini network i")
    fig.colorbar(im, ax=ax, fraction=0.046)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_conductance_evolution(snap: StateSnapshot, path: str | Path) -> None:
    """D per edge over consensus rounds (requires snap.history)."""
    if snap.history is None:
        return
    rounds = len(snap.history)
    E = snap.D.shape[1]
    D_traj = np.zeros((rounds, E))
    for k, h in enumerate(snap.history):
        D_traj[k] = h["D"].mean(axis=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(rounds), D_traj, alpha=0.4, linewidth=0.7)
    ax.plot(np.arange(rounds), D_traj.mean(axis=1), color="black", linewidth=2, label="mean D")
    ax.set_xlabel("consensus round"); ax.set_ylabel("conductance D_ij")
    ax.set_title("Conductance evolution over rounds")
    ax.legend()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_u_norm(snap: StateSnapshot, path: str | Path) -> None:
    if snap.history is None:
        return
    rounds = len(snap.history)
    u_norm = np.zeros(rounds)
    for k, h in enumerate(snap.history):
        u_norm[k] = np.linalg.norm(h["u"], axis=-1).mean()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.arange(rounds), u_norm, marker="o")
    ax.set_xlabel("consensus round"); ax.set_ylabel("mean ||u_i||")
    ax.set_title("Stubbornness (dual norm) over rounds")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_x_z_trajectories(snap: StateSnapshot, path: str | Path, node: int = 0) -> None:
    if snap.history is None:
        return
    rounds = len(snap.history)
    d = snap.x.shape[-1]
    x_traj = np.stack([h["x"][:, node, :] for h in snap.history]).mean(axis=1)  # (rounds, d)
    z_traj = np.stack([h["z"][:, node, :] for h in snap.history]).mean(axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    for j in range(d):
        ax.plot(np.arange(rounds), x_traj[:, j], "--", alpha=0.5, linewidth=0.8)
        ax.plot(np.arange(rounds), z_traj[:, j], "-", alpha=0.8, linewidth=1.2)
    ax.set_xlabel("consensus round"); ax.set_ylabel("latent dim value")
    ax.set_title(f"x (dashed) vs z (solid) — node {node}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_active_links_on_image(
    snap: StateSnapshot,
    image: np.ndarray,
    patch_size: int,
    path: str | Path,
    prune_eps: float = 1e-3,
) -> None:
    """Overlay active communication links on the input image."""
    Dm = conductance_matrix(snap)
    coords = snap.coords
    centers = coords + patch_size / 2.0
    H, W = image.shape[-2:]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray")
    active = Dm >= prune_eps
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if active[i, j]:
                a, b = centers[i], centers[j]
                ax.plot([a[1], b[1]], [a[0], b[0]], color="red", alpha=float(Dm[i, j]), linewidth=1.5)
    ax.plot(centers[:, 1], centers[:, 0], "o", color="cyan", markersize=3)
    ax.set_title("Active links on image")
    ax.set_xticks([]); ax.set_yticks([])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
# Animated GIF: neuron consensus grid across rounds
# --------------------------------------------------------------------------- #


def _interpolated_preds(
    preds: np.ndarray, n_between: int, black_start: bool = True,
    active: np.ndarray | None = None,
    fused_preds: np.ndarray | None = None,
) -> np.ndarray:
    """Linearly interpolate per-neuron colors by blending adjacent rounds' RGB grids.

    `preds` is (R, B, N) int argmax of each column's personal top pick. We map
    each neuron to its digit color per round, then insert `n_between` cross-fade
    frames between consecutive rounds. If black_start=True, prepend an all-black
    round 0 so the GIF opens blank and color fills in as consensus proceeds.

    If `active` is given (R, B, N) in [0,1], INACTIVE columns (active<0.5) are
    forced to black for every round — they don't vote, so they render as silent
    black cells (the Thousand-Brains sparse-column figure: the digit's shape
    emerges in which cells light up against the black background).

    If `fused_preds` is given (R, B) int argmax of the FUSED scorecard per round
    (the model's global decision), then every ACTIVE cell is colored by that
    single fused color instead of its per-column personal pick. This is the
    "one color" view the blueprint describes: all active cells converge to the
    team's verdict. Inactive cells stay black. Pass None for the diagnostic
    per-column (distributed opinion) view.
    Returns (F, B, N, 3) float RGB per neuron.
    """
    R, B, N = preds.shape
    colors = np.zeros((R, B, N, 3))
    for k in range(R):
        for b in range(B):
            if fused_preds is not None:
                # global-decision view: every active cell = the fused argmax color
                v = int(fused_preds[k, b])
                colors[k, b] = DIGIT_COLORS[v]
            else:
                # distributed-opinion view: each cell = its own personal argmax
                for v in range(10):
                    colors[k, b][preds[k, b] == v] = DIGIT_COLORS[v]
            # black out inactive columns (sparse-column gating)
            if active is not None:
                inactive = active[k, b] < 0.5       # (N,) boolean
                colors[k, b][inactive] = 0.0        # black
    seq = []
    if black_start:
        black = np.zeros((B, N, 3))
        seq.append(black)
        for t in range(1, n_between + 1):
            a = t / (n_between + 1)
            seq.append((1 - a) * black + a * colors[0])
    frames = list(seq)
    for k in range(R - 1):
        frames.append(colors[k])
        for t in range(1, n_between + 1):
            a = t / (n_between + 1)
            frames.append((1 - a) * colors[k] + a * colors[k + 1])
    frames.append(colors[-1])
    return np.stack(frames)


def _per_round_fused(logits_per_round: np.ndarray, active_per_round: np.ndarray | None) -> np.ndarray:
    """Fused (consensus) scorecard per round, matching the actual judgment.

    This is the numpy equivalent of acn.networks.sparse_fuse, applied per round:
        fused = mean over ACTIVE columns of the raw per-column logits.
    Returns (R, B, C). This is exactly what the model's argmax runs on to decide
    correct/wrong — NOT the per-column argmax colors shown in the grid. Exposing
    it in the viz makes the judging mechanism transparent: a viewer can see WHY
    the fused winner wins even when most cells' personal top pick differs.
    """
    R, B, N, C = logits_per_round.shape
    if active_per_round is None:
        # dense fallback: all columns vote equally
        return logits_per_round.mean(axis=2)
    w = active_per_round[..., None]                  # (R, B, N, 1)
    summed = (w * logits_per_round).sum(axis=2)       # (R, B, C)
    count = w.sum(axis=2).clip(min=1e-6)              # (R, B, 1)
    return summed / count


def _interpolated_fused(fused_per_round: np.ndarray, n_between: int, black_start: bool = True) -> np.ndarray:
    """Interpolate the per-round fused scorecard to match the color-frame timeline.

    Mirrors _interpolated_preds so bar-chart frames stay in sync with grid frames.
    Returns (F, B, C).
    """
    R, B, C = fused_per_round.shape
    seq = []
    if black_start:
        zero = np.zeros((B, C))
        seq.append(zero)
        for t in range(1, n_between + 1):
            a = t / (n_between + 1)
            seq.append((1 - a) * zero + a * fused_per_round[0])
    frames = list(seq)
    for k in range(R - 1):
        frames.append(fused_per_round[k])
        for t in range(1, n_between + 1):
            a = t / (n_between + 1)
            frames.append((1 - a) * fused_per_round[k] + a * fused_per_round[k + 1])
    frames.append(fused_per_round[-1])
    return np.stack(frames)


def make_consensus_gif(
    model,
    episodes: list[dict],   # each: {snap, images (B,H,W), labels (B,), sample_kinds (B,), coords, stride}
    path: str | Path,
    duration_ms: int = 40,
    transition_steps: int = 10,
    linger_frames: int = 75,
    grid_mode: str = "fused",
) -> None:
    """Create a smooth multi-episode GIF showing neuron grids converging.

    Vertical layout: each row = one sample = [input digit (labeled with ground
    truth only, e.g. "digit 4") | 8x8 Mini networks grid | Mini networks decision
    bar chart with +/- axis], digit color legend on the right. White borders
    around every cell so the grid is visibly distinct. Black round-0 start;
    cross-fade transitions between rounds; linger on the converged result before
    the next batch.

    The per-row label shows only the ground-truth digit (no correct/wrong tag).
    The right-side color legend maps each digit to its color, so the viewer can
    tell whether the model is correct by comparing the grid/bars' winning color
    to the legend entry for the labeled digit.

    grid_mode controls what the grid colors show:
      - "fused" (default): every ACTIVE cell is colored by the FUSED argmax =
        the model's global decision. All active cells share one color (the
        blueprint's "converges to one color"). Inactive cells stay black. This
        is the team's verdict and is always consistent with the correct/wrong
        tag. THIS is what the architecture decides.
      - "per_column": each cell is colored by its OWN argmax (its personal top
        pick). Shows the distributed opinion of the 64 independent experts and
        how they (softly) move during consensus. Cells may differ in color even
        when the fused decision is one color — useful for diagnosing whether
        consensus is converging, but can look inconsistent with the tag.

    The fused-scorecard bar chart (3rd column) shows the ACTUAL judging signal:
    the mean of raw per-column logits over active columns, per digit slot, with
    a labeled +/- horizontal axis so it's clear where zero is. The model's
    prediction = argmax of this scorecard. Bars are colored by digit; the
    winner is drawn at full alpha with a black edge, others dimmed.
    """
    from matplotlib.animation import PillowWriter

    if not episodes:
        raise ValueError("need at least one episode")
    nsamp = len(episodes[0]["images"])
    coords = episodes[0]["coords"]
    stride = episodes[0]["stride"]
    Hg = int(coords[:, 0].max() // stride + 1)
    Wg = int(coords[:, 1].max() // stride + 1)

    # --- precompute each episode's interpolated color frames + fused scorecards ---
    ep_data = []
    for ep in episodes:
        preds, logits_per_round = decode_history_predictions(model, ep["snap"], return_logits=True)
        R = preds.shape[0]
        # per-round active mask from history (B, N) per round -> (R, B, N)
        active_per_round = None
        if ep["snap"].history is not None and "active" in ep["snap"].history[0]:
            active_per_round = np.stack([h["active"] for h in ep["snap"].history])
        # fused scorecard per frame (the ACTUAL judging signal; see _per_round_fused)
        fused_per_round = _per_round_fused(logits_per_round, active_per_round)   # (R, B, C)
        fused_argmax_per_round = fused_per_round.argmax(axis=-1)                 # (R, B) global decision per round
        fused_frames = _interpolated_fused(fused_per_round, n_between=transition_steps, black_start=True)
        # grid colors: fused global decision (one color) unless user asked for per_column diagnostic view
        grid_fused_preds = fused_argmax_per_round if grid_mode == "fused" else None
        color_frames = _interpolated_preds(
            preds, n_between=transition_steps, black_start=True, active=active_per_round,
            fused_preds=grid_fused_preds,
        )
        F_ep = color_frames.shape[0]
        frames_per_round = transition_steps + 1
        frame_round = [0] * (1 + transition_steps)
        for k in range(R - 1):
            frame_round += [k + 1] * frames_per_round
        frame_round += [R]
        frame_round = frame_round[:F_ep]
        grid_imgs_ep = []
        for f in range(F_ep):
            grids = []
            for s in range(nsamp):
                grid = np.full((Hg, Wg, 3), 1.0)
                for idx, (r, c) in enumerate(coords):
                    gr, gc = int(r // stride), int(c // stride)
                    grid[gr, gc] = color_frames[f, s, idx]
                grids.append(grid)
            grid_imgs_ep.append(grids)
        ep_data.append({
            "grid_imgs": grid_imgs_ep, "frame_round": frame_round, "R": R,
            "images": ep["images"], "labels": ep["labels"],
            "kinds": ep["sample_kinds"],
            "row_labels": ep.get("row_labels"),               # condition names per row (robustness viz)
            "fused_frames": fused_frames,           # (F_ep, B, C) fused scorecard per frame
        })

    # --- flat frame sequence with linger between episodes ---
    flat_frames = []
    for ei, ed in enumerate(ep_data):
        F_ep = len(ed["grid_imgs"])
        for f in range(F_ep):
            flat_frames.append((ei, f))
        for _ in range(linger_frames):
            flat_frames.append((ei, F_ep - 1))

    # --- figure: vertical layout ---
    # 4 columns: [input digit | neuron grid | fused-scorecard bars | digit legend]
    # The fused-scorecard bar chart shows the ACTUAL judging signal (mean of raw
    # logits over active columns, then argmax). This is what makes the viz honest:
    # viewers can see WHY the fused winner wins even when most cells' personal top
    # pick (the grid color) differs. Without it, a "majority blue but fused=pink"
    # result looks like a bug; with it, you see every cell contributing to slot 7.
    fig = plt.figure(figsize=(8.6, 2.4 * nsamp + 0.6))
    gs = fig.add_gridspec(nsamp, 4, width_ratios=[1, 2.0, 1.5, 0.32])
    img_axes = [fig.add_subplot(gs[r, 0]) for r in range(nsamp)]
    grid_axes = [fig.add_subplot(gs[r, 1]) for r in range(nsamp)]
    bar_axes = [fig.add_subplot(gs[r, 2]) for r in range(nsamp)]
    ax_legend = fig.add_subplot(gs[:, 3])
    fig.subplots_adjust(left=0.04, right=0.97, top=0.93, bottom=0.03,
                        wspace=0.18, hspace=0.30)

    # shared x-limit for all bar charts (symmetric, from global max|fused|)
    all_fused = np.concatenate(
        [ed["fused_frames"].reshape(-1, 10) for ed in ep_data], axis=0
    )
    x_lim = float(np.abs(all_fused).max()) * 1.15 + 0.5

    # legend (static)
    ax_legend.set_xlim(0, 1); ax_legend.set_ylim(-0.5, 10.5)
    ax_legend.set_xticks([]); ax_legend.set_yticks([])
    ax_legend.set_title("digit", fontsize=10, pad=6)
    for v in range(10):
        ax_legend.add_patch(plt.Rectangle((0.1, v - 0.35), 0.25, 0.7, color=DIGIT_COLORS[v]))
        ax_legend.text(0.45, v, str(v), va="center", fontsize=10)
    ax_legend.invert_yaxis()

    # grid imshow artists + white cell borders, one per row
    grid_artists = []
    for r in range(nsamp):
        im = grid_axes[r].imshow(ep_data[0]["grid_imgs"][0][r], interpolation="nearest", vmin=0, vmax=1)
        grid_axes[r].set_xticks([]); grid_axes[r].set_yticks([])
        for gr in range(Hg):
            for gc in range(Wg):
                grid_axes[r].add_patch(plt.Rectangle(
                    (gc - 0.5, gr - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.2, zorder=2,
                ))
        grid_artists.append(im)
    # header on the grid column (only once, on the first row)
    grid_axes[0].set_title("Mini networks", fontsize=10, pad=4)

    # input image artists
    img_artists = []
    for r in range(nsamp):
        im = img_axes[r].imshow(episodes[0]["images"][r], cmap="gray", vmin=0, vmax=1)
        img_axes[r].set_xticks([]); img_axes[r].set_yticks([])
        img_artists.append(im)

    # fused-scorecard bar artists: 10 horizontal bars per sample (one per digit).
    # Bar WIDTH is the fused score (updates every frame); bar COLOR is the digit
    # color; the WINNER (argmax of fused) is drawn at full alpha with a black
    # edge, the rest dimmed so the eye locks onto the consensus winner.
    bar_artists = []   # list of lists of 10 barh patches, per sample row
    for r in range(nsamp):
        ax = bar_axes[r]
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-0.6, 9.6)
        ax.set_yticks(range(10))
        ax.set_yticklabels([str(d) for d in range(10)], fontsize=8)
        # labeled +/- horizontal axis: show negative, zero, positive so the
        # viewer can see which bars are above/below zero (agreement vs dissent).
        tick_vals = [-x_lim, 0.0, x_lim]
        ax.set_xticks(tick_vals)
        ax.set_xticklabels([f"\u2212{x_lim:.1f}", "0", f"+{x_lim:.1f}"], fontsize=7)
        ax.tick_params(axis="x", length=2, pad=1)
        # emphasize the zero line (the decision boundary: + = evidence for, \u2212 = against)
        ax.axvline(0.0, color="0.35", linewidth=1.0, zorder=0)
        if r == nsamp - 1:
            ax.set_xlabel("fused logit  (\u2212  |  +)", fontsize=7, labelpad=1)
        bars = []
        for d in range(10):
            b = ax.barh(d, 0.0, color=DIGIT_COLORS[d], alpha=0.35,
                        edgecolor="none", height=0.7, zorder=1)
            bars.append(b)
        bar_artists.append(bars)
        if r == 0:
            ax.set_title("Mini networks decision", fontsize=10, pad=4)

    # per-row input label. If the episode provides `row_labels` (e.g. condition
    # names for the robustness viz), use them; otherwise default to the digit.
    # Always show the correct/wrong tag so the user sees the verdict.
    ed0 = ep_data[0]
    def _row_title(ed, r):
        if ed.get("row_labels") is not None:
            head = str(ed["row_labels"][r])
        else:
            head = f"digit {int(ed['labels'][r])}"
        kind = ed["kinds"][r]
        tag = "\u2713" if kind == "correct" else "\u2717"
        return f"{head}  {tag}"
    for r in range(nsamp):
        img_axes[r].set_title(_row_title(ed0, r), fontsize=10, pad=4)

    cur_ep = [-1]

    def draw(frame):
        ei, fi = flat_frames[frame]
        ed = ep_data[ei]
        fused_f = ed["fused_frames"][fi]        # (B, C) fused scorecard this frame
        for r in range(nsamp):
            grid_artists[r].set_data(ed["grid_imgs"][fi][r])
            # update the fused-scorecard bars for this sample
            scores = fused_f[r]                  # (C,)
            winner = int(scores.argmax())
            for d in range(10):
                bars_d = bar_artists[r][d]
                # barh width = score; redraw by setting the rectangle width
                for patch in bars_d:
                    patch.set_width(scores[d])
                is_winner = (d == winner)
                for patch in bars_d:
                    patch.set_alpha(1.0 if is_winner else 0.30)
                    patch.set_edgecolor("black" if is_winner else "none")
                    patch.set_linewidth(1.2 if is_winner else 0.0)
        if ei != cur_ep[0]:
            cur_ep[0] = ei
            for r in range(nsamp):
                img_artists[r].set_data(ed["images"][r])
                img_axes[r].set_title(_row_title(ed, r), fontsize=10, pad=4)
        k = ed["frame_round"][fi] if fi < len(ed["frame_round"]) else ed["R"]
        R = ed["R"]
        label = "round 0 (init)" if k == 0 else f"round {k}/{R}"
        ep_label = f"  |  batch {ei + 1}/{len(episodes)}" if len(episodes) > 1 else ""
        fig.suptitle(f"Adaptive Consensus Networks  \u2014  {label}{ep_label}", fontsize=13)
        return grid_artists

    anim = matplotlib.animation.FuncAnimation(
        fig, draw, frames=len(flat_frames), repeat=True, interval=duration_ms, blit=False,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=int(1000 / duration_ms))
    anim.save(path, writer=writer)
    plt.close(fig)


