"""AdaptiveConsensusNetwork: the full nn.Module.

Wires decomposition -> encoder -> ACNCore consensus -> decoder -> fusion.
Caches topology (edges, coords, restriction maps) per (image_size, patch, stride).
Exposes interpretable state via :meth:`forward`'s second return value.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from acn.config import ModelConfig
from acn.consensus import ACNCore, ConsensusState
from acn.decomposition import decompose
from acn.networks import Decoder, Encoder, RestrictionMaps, column_gate, sparse_fuse, confident_fuse
from acn.topology import (
    build_spatial_neighbors,
    build_all_pairs,
    init_conductance,
    overlap_weights,
)


@dataclass
class TopologyCache:
    patch_dim: int
    num_nodes: int
    num_edges: int
    edges: torch.Tensor       # (2, E)
    coords: torch.Tensor      # (N, 2)
    D_init: torch.Tensor      # (E,)


class AdaptiveConsensusNetwork(nn.Module):
    """The ACN model.

    Args:
        data_image_size, patch_size, stride: decomposition geometry (needed to size
            the encoder and build the topology cache).
        model_cfg: :class:`acn.config.ModelConfig`.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        stride: int,
        model_cfg: ModelConfig,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.cfg = model_cfg

        patch_dim = patch_size * patch_size  # grayscale; C=1 assumed
        self.patch_dim = patch_dim

        self.encoder = Encoder(
            patch_dim=patch_dim, d=model_cfg.latent_dim,
            hidden=model_cfg.enc_hidden, eps=model_cfg.A_eps,
        )
        self.decoder = Decoder(
            d=model_cfg.latent_dim, num_classes=model_cfg.num_classes,
            hidden=model_cfg.dec_hidden,
        )

        # learnable ADMM/consensus hyperparams (frozen, non-learnable)
        self.rho = nn.Parameter(torch.tensor(float(model_cfg.rho)), requires_grad=False)
        self.eta_z = nn.Parameter(torch.tensor(float(model_cfg.eta_z)), requires_grad=False)
        # conductance rates: FROZEN (non-learnable). Freezing lets the pure
        # Physarum flux dynamics differentiate links without the sparsity loss
        # silently crushing the rates.
        self.eta_D = nn.Parameter(
            torch.tensor(float(model_cfg.eta_D)), requires_grad=False
        )
        self.gamma_D = nn.Parameter(
            torch.tensor(float(model_cfg.gamma_D)), requires_grad=False
        )

        # topology + restriction maps: built eagerly so .to(device) moves them.
        self._topo: TopologyCache | None = None
        self._restriction: RestrictionMaps | None = None
        self._build_topology_from_geometry()

    # ------------------------------------------------------------------ #
    # topology construction (cached)
    # ------------------------------------------------------------------ #

    def _build_topology_from_geometry(self) -> None:
        dummy = torch.zeros(1, 1, self.image_size, self.image_size)
        deco = decompose(dummy, self.patch_size, self.stride)
        mode = self.cfg.topology_mode
        if mode == "allpairs":
            edges = build_all_pairs(deco.num_patches)
            # no spatial overlap for non-adjacent pairs; fall back to dense init
            # so every edge starts from the same conductance and lets the
            # Physarum dynamics decide which wires matter.
            D_init = init_conductance(
                edges, mode="dense",
                dense_value=self.cfg.D_init, overlap=None,
            )
        elif mode == "spatial":
            edges, _ = build_spatial_neighbors(deco, self.patch_size)
            ov = overlap_weights(deco, self.patch_size, edges)
            D_init = init_conductance(
                edges, mode=self.cfg.D_init_mode,
                dense_value=self.cfg.D_init, overlap=ov,
            )
        else:
            raise ValueError(f"unknown topology_mode {mode!r}; 'spatial' | 'allpairs'")
        E = edges.shape[1]
        self._topo = TopologyCache(
            patch_dim=self.patch_dim, num_nodes=deco.num_patches,
            num_edges=E, edges=edges, coords=deco.coords, D_init=D_init,
        )
        self._restriction = RestrictionMaps(
            num_edges=E, d=self.cfg.latent_dim,
        )

    def _ensure_topology(self, device: torch.device, dtype: torch.dtype) -> None:
        # topology is built eagerly in __init__; just ensure device placement.
        t = self._topo
        assert t is not None
        if t.edges.device != device:
            t.edges = t.edges.to(device)
            t.coords = t.coords.to(device)
            t.D_init = t.D_init.to(device)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def forward(
        self, images: torch.Tensor, record: bool = False
    ) -> tuple[torch.Tensor, ConsensusState]:
        """images: (B, 1, H, W) -> (logits (B, C), state)."""
        if images.dim() != 4:
            raise ValueError(f"expected (B,1,H,W); got {tuple(images.shape)}")
        device, dtype = images.device, images.dtype

        self._ensure_topology(device, dtype)
        topo = self._topo
        r_ij, r_ji = self._restriction.maps()

        # 1. decompose
        deco = decompose(images, self.patch_size, self.stride)
        patches = deco.patches if deco.patches.dtype == dtype else deco.patches.to(dtype)
        # 2. encode -> A, b, and s (per-column relevance logits for the gate)
        A, b, s = self.encoder(patches)                # (B, N, d, d), (B, N, d), (B, N)

        # 2b. learned sparse column gate (the adaptive mechanism): discover which
        # columns fire for THIS input. Inactive columns stay silent in consensus
        # + voting. Recomputed each forward pass.
        active = column_gate(s)                        # (B, N) in [0,1]

        # 3. consensus (masked by column activity)
        state = ACNCore.run(
            A=A, b=b,
            r_ij=r_ij, r_ji=r_ji,
            edges=topo.edges,
            D_init=topo.D_init,
            num_nodes=topo.num_nodes,
            rounds=self.cfg.rounds,
            diffusion_steps=self.cfg.diffusion_steps,
            rho=self.rho, eta_z=self.eta_z,
            eta_D=self.eta_D, gamma_D=self.gamma_D,
            D_clip=self.cfg.D_clip,
            record=record,
            detach_after=self.cfg.detach_after,
            active=active,
        )

        # 4. decode + fuse (only active columns vote)
        logits_per_node = self.decoder(state.z)        # (B, N, C)
        state.logits = logits_per_node
        state.relevance_logits = s                    # saved for the z-loss in training
        if self.cfg.fuse_mode == "confident":
            pred = confident_fuse(logits_per_node, state.active, tau=self.cfg.fuse_tau)
        else:
            pred = sparse_fuse(logits_per_node, state.active)
        return pred, state

    # ------------------------------------------------------------------ #
    # convenience for inspection / ablations
    # ------------------------------------------------------------------ #

    @property
    def restriction(self) -> RestrictionMaps:
        if self._restriction is None:
            raise RuntimeError("call forward() first to build topology")
        return self._restriction

    @property
    def topology(self) -> TopologyCache:
        if self._topo is None:
            raise RuntimeError("call forward() first to build topology")
        return self._topo
