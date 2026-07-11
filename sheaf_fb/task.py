"""MNIST task for the Sheaf-FB network: decomposition, loss, evaluation.

Reuses the generic grid-decomposition / patchify / MNIST-loader utilities from
:mod:`acn.data` (they are geometry-only and independent of the ADMM solver).

The global prediction is the **uniform population vote**: the mean of the
per-agent softmax over all 49 agents, exactly as in the spec
(``p_global = (1/49) * sum_i decoder(x_i)``). Train and eval use the same
forward pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from acn.data import build_grid_edge_indices, grid_agent_centers, patchify_batch

from .config import DataConfig


class MNISTTaskFB:
    """MNIST decomposition + evaluation for the Sheaf-FB model."""

    name = "mnist"

    def __init__(self, data_cfg: DataConfig, device: torch.device):
        self.patch_size = data_cfg.patch_size
        self.stride = data_cfg.stride
        self.connectivity = data_cfg.connectivity
        self.num_classes = data_cfg.num_classes
        self.image_hw = (data_cfg.image_size, data_cfg.image_size)
        self.device = device

        centers = grid_agent_centers(self.image_hw, self.stride, self.patch_size)
        edges = build_grid_edge_indices(centers, self.stride, self.connectivity)
        self.N = int(centers.shape[0])
        self.centers = torch.as_tensor(centers, dtype=torch.long, device=device)
        self.edge_indices = torch.as_tensor(edges, dtype=torch.long, device=device)
        # Node positions are not needed for per-edge sheaf maps, but are kept for
        # parity with acn and for directional-map sharing.
        self.node_positions = torch.as_tensor(centers, dtype=torch.float32, device=device)

    def prepare(self, images: torch.Tensor, labels: torch.Tensor):
        """``images`` ``[B, 1, H, W]`` -> ``fwd`` dict, ``targets`` dict."""
        imgs = images.to(self.device).permute(0, 2, 3, 1).contiguous()  # [B, H, W, 1]
        patches = patchify_batch(imgs, self.centers, self.patch_size)  # [N, B, ps, ps, 1]
        return (
            {
                "patches": patches,
                "edge_indices": self.edge_indices,
                "node_positions": self.node_positions,
            },
            {"labels": labels.to(self.device)},
            {},
        )

    def global_prediction(self, logits: torch.Tensor) -> torch.Tensor:
        """Average per-agent softmax into a global probability vector.

        ``logits``: ``[N, B, C]`` -> ``p_global`` ``[B, C]``.
        """
        return F.softmax(logits, dim=-1).mean(0)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor, loss_type: str = "ce") -> torch.Tensor:
        """Global loss from per-agent logits (uniform vote)."""
        p_global = self.global_prediction(logits)
        if loss_type == "ce":
            return F.cross_entropy(p_global, labels)
        if loss_type == "mse":
            y = F.one_hot(labels, logits.shape[-1]).float()
            return 0.5 * torch.sum((p_global - y) ** 2)
        raise ValueError(f"unknown loss_type {loss_type!r}")

    def evaluate(self, logits: torch.Tensor, targets: dict) -> dict[str, float]:
        """Uniform population vote accuracy."""
        p_global = self.global_prediction(logits)
        pred = p_global.argmax(-1)
        return {"acc": (pred == targets["labels"]).float().mean().item()}
