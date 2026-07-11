"""Per-epoch consensus visualization for Sheaf-FB on MNIST (fast PIL renderer).

Generates a smooth GIF every few epochs showing **correctly predicted** test
samples. For each sample, the model's ``settle_history`` path decodes every
agent's state ``x^k`` with the shared decoder at **all K Forward-Backward
rounds**, yielding per-agent predictions ``[K, N, C]``. Each agent on the 7×7
grid is colored by its predicted digit (brightness = confidence), evolving
through the rounds as the population converges from disagreement to consensus.

Layout (per GIF): ``n_samples`` panels, each showing
[input image | 7×7 agent grid], plus a color→digit legend on the right.

Uses PIL for frame rendering (~5 ms/frame) instead of matplotlib (~300 ms/frame)
for smooth, lag-free playback.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Color mapping: digit → color
# ---------------------------------------------------------------------------

DIGIT_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

_DIGIT_RGB_FLOAT = []
for h in DIGIT_COLORS:
    h = h.lstrip("#")
    _DIGIT_RGB_FLOAT.append(tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)))

DIGIT_RGB_INT = [tuple(int(c * 255) for c in rgb) for rgb in _DIGIT_RGB_FLOAT]

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_DISPLAY = 140           # input image & agent grid display size (px)
_PANEL_W = _DISPLAY * 2 + 8   # input + gap + grid
_PANEL_H = _DISPLAY + 22      # grid + title bar
_SAMPLE_GAP = 14
_LEGEND_W = 95
_MARGIN = 12
_TITLE_H = 30


# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Sample selection: find correctly predicted test samples
# ---------------------------------------------------------------------------

@torch.no_grad()
def select_correct_samples(
    model, task, loader, *, n_samples: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find ``n_samples`` test images the model predicts correctly.

    Uses the same uniform vote as training/eval. Falls back to random samples
    if fewer than ``n_samples`` are correct.
    """
    model.eval()
    correct_imgs, correct_lbls = [], []
    all_imgs, all_lbls = [], []
    found = 0
    for images, labels in loader:
        all_imgs.append(images)
        all_lbls.append(labels)
        fwd, targets, _ = task.prepare(images, labels)
        logits = model(fwd["patches"])  # [N, B, C]
        probs = F.softmax(logits, dim=-1).mean(0)
        preds = probs.argmax(-1)
        for i in range(images.shape[0]):
            if preds[i].item() == labels[i].item():
                correct_imgs.append(images[i:i + 1])
                correct_lbls.append(labels[i:i + 1])
                found += 1
            if found >= n_samples:
                break
        if found >= n_samples:
            break

    if found >= n_samples:
        return (torch.cat(correct_imgs[:n_samples]),
                torch.cat(correct_lbls[:n_samples]))

    # Fallback: not enough correct predictions — use random samples
    all_imgs_cat = torch.cat(all_imgs)
    all_lbls_cat = torch.cat(all_lbls)
    n_use = min(n_samples, all_imgs_cat.shape[0])
    return all_imgs_cat[:n_use], all_lbls_cat[:n_use]


# ---------------------------------------------------------------------------
# Per-sample data collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_sample_data(
    model, task, images: torch.Tensor, labels: torch.Tensor,
    *, device: torch.device,
) -> dict:
    """Run ``settle_history`` for one sample and collect per-round predictions.

    Returns a dict with per-round per-agent predictions, fused predictions,
    grid shape, and the input image.
    """
    fwd, _, _ = task.prepare(images.to(device), labels.to(device))

    # Run the full Forward-Backward trajectory
    history = model.settle_history(fwd["patches"], nudged=False)
    x_traj = history["x"]  # [K, N, B, d]

    K, N, B, d = x_traj.shape
    grid_cols = int(math.isqrt(N))
    grid_rows = (N + grid_cols - 1) // grid_cols

    # Decode every agent's state at every round
    probs_per_round = []
    for k in range(K):
        logits = model._decode(x_traj[k])  # [N, B, C]
        probs = F.softmax(logits, dim=-1)  # [N, B, C]
        probs_per_round.append(probs[:, 0, :])  # [N, C] (batch=0)
    probs_per_round = torch.stack(probs_per_round)  # [K, N, C]

    # Per-round fused prediction (uniform vote, same as inference)
    fused_preds, fused_confs = [], []
    for k in range(K):
        fused = probs_per_round[k].mean(0)  # [C]
        pred = int(fused.argmax().item())
        fused_preds.append(pred)
        fused_confs.append(float(fused[pred].item()))

    label = int(labels[0].item())
    correct = (fused_preds[-1] == label)

    # Also collect energy trajectory for diagnostics
    total_e = history["total_energy"].cpu().numpy()  # [K]

    return {
        "probs": probs_per_round.cpu().numpy(),   # [K, N, C]
        "fused_preds": fused_preds,                 # [K]
        "fused_confs": fused_confs,                 # [K]
        "correct": correct,
        "label": label,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "N": N,
        "K": K,
        "input_img": images[0, 0].cpu().numpy(),    # [H, W] grayscale
        "total_energy": total_e,                    # [K]
    }


# ---------------------------------------------------------------------------
# PIL rendering
# ---------------------------------------------------------------------------

def _render_legend(font_label: ImageFont.FreeTypeFont) -> Image.Image:
    """Pre-render the digit→color legend (drawn once, pasted every frame)."""
    box = 22
    row_h = 26
    legend = Image.new("RGB", (_LEGEND_W, 10 * row_h + 4), (20, 20, 20))
    draw = ImageDraw.Draw(legend)
    for d in range(10):
        y = d * row_h
        draw.rectangle([6, y, 6 + box, y + box], fill=DIGIT_RGB_INT[d],
                       outline=(120, 120, 120), width=1)
        draw.text((6 + box + 8, y + 3), str(d), fill="white", font=font_label)
    return legend


def _render_input_image(gray: np.ndarray, size: int) -> Image.Image:
    """Convert a grayscale [H,W] array to a resized RGB PIL Image."""
    img = Image.fromarray((gray * 255).astype(np.uint8), mode="L")
    img = img.convert("RGB").resize((size, size), Image.NEAREST)
    return img


def _draw_agent_grid(
    draw: ImageDraw.ImageDraw, gx: int, gy: int, size: int,
    probs_k: np.ndarray, grid_rows: int, grid_cols: int,
) -> None:
    """Draw the agent grid at (gx, gy) with per-agent colors for round k."""
    cell_w = size / grid_cols
    cell_h = size / grid_rows
    for ci in range(len(probs_k)):
        r, c = ci // grid_cols, ci % grid_cols
        pred = int(probs_k[ci].argmax())
        conf = float(probs_k[ci][pred])
        rgb = DIGIT_RGB_INT[pred]
        bright = 0.25 + 0.75 * conf
        color = tuple(int(ch * bright) for ch in rgb)
        x0 = gx + int(c * cell_w)
        y0 = gy + int(r * cell_h)
        x1 = gx + int((c + 1) * cell_w) - 1
        y1 = gy + int((r + 1) * cell_h) - 1
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=(50, 50, 50))


# ---------------------------------------------------------------------------
# Main GIF generator
# ---------------------------------------------------------------------------

def make_consensus_gif(
    model, task, loader, path: str | Path,
    *,
    epoch: int,
    device: torch.device,
    n_samples: int = 4,
    frame_ms: int = 120,
    linger_frames: int = 5,
    linger_ms: int = 1000,
) -> Path:
    """Generate a smooth consensus-evolution GIF from correctly predicted test samples.

    Selects ``n_samples`` correctly predicted test images, runs the model through
    all ``K`` Forward-Backward rounds, and renders the per-agent prediction
    evolution as a smooth GIF with a digit→color legend on the right.

    Uses PIL for frame rendering (~5 ms/frame) instead of matplotlib (~300 ms/frame)
    for smooth, lag-free playback.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Select correctly predicted samples
    images, labels = select_correct_samples(
        model, task, loader, n_samples=n_samples, device=device)

    # 2. Collect per-agent data for all samples
    all_data: list[dict] = []
    for s in range(n_samples):
        data = _collect_sample_data(
            model, task, images[s:s + 1], labels[s:s + 1], device=device)
        all_data.append(data)

    K = all_data[0]["K"]

    # 3. Compute canvas size (single column of samples + legend)
    canvas_w = _MARGIN + _PANEL_W + 8 + _LEGEND_W + _MARGIN
    canvas_h = _TITLE_H + n_samples * _PANEL_H + (n_samples - 1) * _SAMPLE_GAP + _MARGIN

    # 4. Pre-render fixed elements (legend + input images)
    font_label = _load_font(14)
    font_title = _load_font(13)
    font_main = _load_font(16)
    legend_img = _render_legend(font_label)
    legend_x = _MARGIN + _PANEL_W + 8
    legend_y = _TITLE_H + 10

    input_imgs: list[Image.Image] = []
    for s in range(n_samples):
        input_imgs.append(_render_input_image(all_data[s]["input_img"], _DISPLAY))

    # 5. Compute panel positions
    def panel_pos(s: int) -> tuple[int, int]:
        px = _MARGIN
        py = _TITLE_H + s * (_PANEL_H + _SAMPLE_GAP)
        return px, py

    # 6. Render frames
    frame_imgs: list[Image.Image] = []
    for k in range(K):
        frame = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))
        draw = ImageDraw.Draw(frame)

        # Paste legend
        frame.paste(legend_img, (legend_x, legend_y))

        # Main title
        title_text = f"Epoch {epoch}  round {k+1}/{K}  —  Forward-Backward consensus"
        draw.text((_MARGIN, 6), title_text, fill="white", font=font_main)

        for s in range(n_samples):
            px, py = panel_pos(s)
            data = all_data[s]

            # Paste input image (pre-rendered)
            frame.paste(input_imgs[s], (px, py))

            # Draw agent grid
            gx = px + _DISPLAY + 8
            gy = py
            _draw_agent_grid(
                draw, gx, gy, _DISPLAY,
                data["probs"][k], data["grid_rows"], data["grid_cols"])

            # Panel title
            pred = data["fused_preds"][k]
            conf = data["fused_confs"][k]
            correct = data["correct"]
            mark = "✓" if correct else "✗"
            title = f"label {data['label']}  → {pred} ({conf:.0%}) {mark}"
            color = (100, 255, 100) if correct else (255, 100, 100)
            draw.text((px, py + _DISPLAY + 4), title, fill=color, font=font_title)

        frame_imgs.append(frame)

    # Linger on the final frame
    for _ in range(linger_frames):
        frame_imgs.append(frame_imgs[-1].copy())

    durations = [frame_ms] * K + [linger_ms] * linger_frames
    frame_imgs[0].save(
        str(path), save_all=True, append_images=frame_imgs[1:],
        duration=durations, loop=0, optimize=False, disposal=2)
    return path
