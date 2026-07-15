"""Consensus-evolution visualization for the Lean Sheaf-ADMM model (PIL).

Generates a GIF every epoch showing **correctly predicted** test samples. The
layout is vertical — one row per sample::

    Image   Neuron grid population     <- sample 1
    Image   Neuron grid population     <- sample 2
    Image   Neuron grid population     <- sample 3
    Image   Neuron grid population     <- sample 4

Each agent is colored by its predicted digit (brightness = confidence) across
the ``K`` ADMM rounds, showing the population converging from disagreement to
consensus. Self-contained (inlined palette + font loader, no sibling-package
dependency).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from .config import ModelConfig
from .graph import patchify
from .model import AdaptiveConsensusModel

# ---------------------------------------------------------------------------
# Color mapping: digit -> color (self-contained)
# ---------------------------------------------------------------------------

DIGIT_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
_DIGIT_RGB_FLOAT = []
for _h in DIGIT_COLORS:
    _h = _h.lstrip("#")
    _DIGIT_RGB_FLOAT.append(tuple(int(_h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)))
DIGIT_RGB_INT = [tuple(int(c * 255) for c in rgb) for rgb in _DIGIT_RGB_FLOAT]


def _load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Layout constants (vertical: one row per sample = image + neuron grid)
# ---------------------------------------------------------------------------

_DISPLAY = 120
_ROW_W = _DISPLAY + 8 + _DISPLAY          # image + gap + neuron grid
_ROW_H = _DISPLAY + 22                    # grid + title bar
_ROW_GAP = 10
_LEGEND_W = 95
_MARGIN = 12
_TITLE_H = 30


# ---------------------------------------------------------------------------
# Sample selection + per-sample data
# ---------------------------------------------------------------------------

def _patches(images, model_cfg: ModelConfig):
    return patchify(images, model_cfg.patch_size, model_cfg.stride).permute(1, 0, 2)


@torch.no_grad()
def select_correct_samples(model, loader, model_cfg, *, n_samples, device):
    """Fetch up to ``n_samples`` correctly-predicted test images (falls back to
    the first available test samples if not enough are correct)."""
    model.eval()
    correct_imgs, correct_lbls = [], []
    all_imgs, all_lbls = [], []
    found = 0
    for images, labels in loader:
        all_imgs.append(images)
        all_lbls.append(labels)
        logits, _ = model(_patches(images.to(device), model_cfg), K=model_cfg.K_eval)
        preds = logits.argmax(-1)
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
        return torch.cat(correct_imgs[:n_samples]), torch.cat(correct_lbls[:n_samples])
    all_imgs_cat = torch.cat(all_imgs)
    all_lbls_cat = torch.cat(all_lbls)
    n_use = min(n_samples, all_imgs_cat.shape[0])
    return all_imgs_cat[:n_use], all_lbls_cat[:n_use]


@torch.no_grad()
def _collect_sample(model, image, label, model_cfg, *, device):
    """Run the model on one test image; return per-round per-agent probs."""
    imgs = image.to(device)
    patches = _patches(imgs, model_cfg)
    logits, aux = model(patches, K=model_cfg.K_eval)
    K = aux["per_round_logits"].shape[0]
    probs = F.softmax(aux["per_round_logits"], dim=-1).cpu().numpy()  # (K, N, 1, C)
    probs = probs[:, :, 0, :]                                          # (K, N, C)

    fused_preds, fused_confs = [], []
    for kk in range(K):
        fused = probs[kk].mean(0)
        fused_preds.append(int(fused.argmax()))
        fused_confs.append(float(fused[fused_preds[-1]]))

    label = int(label[0].item())
    final_pred = int(logits[0].argmax().item())

    n_agents = probs.shape[1]
    grid_cols = int(math.isqrt(n_agents))
    grid_rows = (n_agents + grid_cols - 1) // grid_cols

    return {
        "probs": probs,
        "fused_preds": fused_preds,
        "fused_confs": fused_confs,
        "final_pred": final_pred,
        "correct": (final_pred == label),
        "label": label,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "input_img": imgs[0, 0].cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# PIL rendering helpers
# ---------------------------------------------------------------------------

def _render_legend(font_label):
    box, row_h = 22, 26
    legend = Image.new("RGB", (_LEGEND_W, 10 * row_h + 4), (20, 20, 20))
    draw = ImageDraw.Draw(legend)
    for d in range(10):
        y = d * row_h
        draw.rectangle([6, y, 6 + box, y + box], fill=DIGIT_RGB_INT[d],
                       outline=(120, 120, 120), width=1)
        draw.text((6 + box + 8, y + 3), str(d), fill="white", font=font_label)
    return legend


def _render_input_image(gray, size):
    img = Image.fromarray((gray * 255).astype(np.uint8), mode="L")
    return img.convert("RGB").resize((size, size), Image.NEAREST)


def _agent_color(probs_i):
    pred = int(probs_i.argmax())
    conf = float(probs_i[pred])
    rgb = DIGIT_RGB_INT[pred]
    bright = 0.25 + 0.75 * conf
    return tuple(int(ch * bright) for ch in rgb)


def _draw_neuron_grid(draw, gx, gy, size, probs_k, grid_rows, grid_cols):
    cell_w = size / grid_cols
    cell_h = size / grid_rows
    for ci in range(len(probs_k)):
        r, c = ci // grid_cols, ci % grid_cols
        x0 = gx + int(c * cell_w)
        y0 = gy + int(r * cell_h)
        x1 = gx + int((c + 1) * cell_w) - 1
        y1 = gy + int((r + 1) * cell_h) - 1
        draw.rectangle([x0, y0, x1, y1], fill=_agent_color(probs_k[ci]),
                       outline=(50, 50, 50))


# ---------------------------------------------------------------------------
# Consensus-evolution GIF (vertical: 4 test samples, one row each)
# ---------------------------------------------------------------------------

def make_consensus_gif(model: AdaptiveConsensusModel, loader, path: str | Path, *,
                       model_cfg: ModelConfig, epoch: int, device: torch.device,
                       n_samples: int = 4, frame_ms: int = 120,
                       linger_frames: int = 5, linger_ms: int = 800) -> Path:
    """Render a vertical-layout consensus GIF: one row per test sample, each row
    showing the input image alongside its neuron-grid population converging over
    the ``K`` ADMM rounds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images, labels = select_correct_samples(
        model, loader, model_cfg, n_samples=n_samples, device=device)

    all_data = [
        _collect_sample(model, images[s:s + 1], labels[s:s + 1], model_cfg, device=device)
        for s in range(images.shape[0])
    ]
    n = len(all_data)
    K = all_data[0]["probs"].shape[0]

    canvas_w = _MARGIN + _ROW_W + 8 + _LEGEND_W + _MARGIN
    canvas_h = _TITLE_H + n * _ROW_H + (n - 1) * _ROW_GAP + _MARGIN

    font_label = _load_font(14)
    font_title = _load_font(13)
    font_main = _load_font(16)
    legend_img = _render_legend(font_label)
    legend_x = _MARGIN + _ROW_W + 8
    legend_y = _TITLE_H + 10

    input_imgs = [_render_input_image(d["input_img"], _DISPLAY) for d in all_data]

    def row_y(i):
        return _TITLE_H + i * (_ROW_H + _ROW_GAP)

    frames = []
    for kk in range(K):
        frame = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))
        draw = ImageDraw.Draw(frame)
        frame.paste(legend_img, (legend_x, legend_y))
        draw.text((_MARGIN, 6),
                  f"Epoch {epoch}  round {kk + 1}/{K}  —  consensus dynamics",
                  fill="white", font=font_main)
        for i, data in enumerate(all_data):
            ry = row_y(i)
            px = _MARGIN
            frame.paste(input_imgs[i], (px, ry))
            gx = px + _DISPLAY + 8
            _draw_neuron_grid(draw, gx, ry, _DISPLAY,
                              data["probs"][kk], data["grid_rows"], data["grid_cols"])
            pred = data["fused_preds"][kk]
            conf = data["fused_confs"][kk]
            mark = "✓" if (pred == data["label"]) else "✗"
            title = f"sample {i + 1}  |  label {data['label']}  → {pred} ({conf:.0%}) {mark}"
            color = (100, 255, 100) if data["correct"] else (255, 100, 100)
            draw.text((px, ry + _DISPLAY + 4), title, fill=color, font=font_title)
        frames.append(frame)

    for _ in range(linger_frames):
        frames.append(frames[-1].copy())
    durations = [frame_ms] * K + [linger_ms] * linger_frames
    frames[0].save(str(path), save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=False, disposal=2)
    return path
