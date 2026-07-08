"""AdaptiveConsensusNetwork: full nn.Module with reference-frame, motor, hierarchy, primer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from acn.config import ModelConfig, DataConfig
from acn.consensus import ACNCore, ConsensusState
from acn.decomposition import decompose, build_column_roster, decompose_multi_scale
from acn.networks import (
    AbstractEncoder,
    Decoder,
    Encoder,
    MotorSystem,
    PredictiveMaps,
    Primer,
    RestrictionMaps,
    hard_topk_gate,
    logit_margin_confidence,
    sparse_fuse,
    sparse_fuse_vectors,
)
from acn.topology import (
    build_all_pairs,
    build_multi_scale_neighbors,
    build_spatial_neighbors,
    init_conductance,
    overlap_weights,
)


@dataclass
class TopologyCache:
    patch_dim: int
    num_edges: int
    edges: torch.Tensor       # (2, E)
    coords: torch.Tensor      # (N, 2)
    D_init: torch.Tensor      # (E,)
    roster: list | None = None
    pool_to: int | None = None


class AdaptiveConsensusNetwork(nn.Module):
    """The ACN model."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        stride: int,
        model_cfg: ModelConfig,
        data_cfg: DataConfig | None = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.cfg = model_cfg
        self.data_cfg = data_cfg

        self.decomposition_mode = data_cfg.decomposition_mode if data_cfg else "single"
        if self.decomposition_mode == "multi":
            self.pool_to = data_cfg.pool_to
            patch_dim = self.pool_to * self.pool_to
        else:
            self.pool_to = None
            patch_dim = patch_size * patch_size
        self.patch_dim = patch_dim

        # ── Bottom layer ──
        self.bottom_encoder = Encoder(
            patch_dim=patch_dim,
            d=model_cfg.latent_dim,
            hidden=model_cfg.enc_hidden,
            eps=model_cfg.A_eps,
            use_positional=model_cfg.use_positional_encoding,
            pos_embed_dim=model_cfg.pos_embed_dim,
        )
        self.bottom_decoder = Decoder(
            d=model_cfg.latent_dim,
            num_classes=model_cfg.num_classes,
            hidden=model_cfg.dec_hidden,
        )
        self.bottom_motor = (
            MotorSystem(d=model_cfg.latent_dim, motor_dim=model_cfg.motor_dim)
            if model_cfg.use_motor
            else None
        )
        self.rho = nn.Parameter(torch.tensor(float(model_cfg.rho)), requires_grad=False)
        self.eta_z = nn.Parameter(torch.tensor(float(model_cfg.eta_z)), requires_grad=False)

        # Topology caches
        self._bottom_topo = None
        self._bottom_restriction = None
        self._build_bottom_topology()

        self.topk_k = model_cfg.topk_k
        self.topk_tau = model_cfg.topk_tau

        # ── Abstract layer ──
        self._abstract_topo = None
        self._abstract_restriction = None
        self.use_abstract = model_cfg.use_abstract_layer
        if self.use_abstract:
            self.abstract_pos = nn.Parameter(torch.randn(model_cfg.abstract_num_columns, 2))
            self.abstract_encoder = AbstractEncoder(
                d_in=model_cfg.latent_dim,
                d_out=model_cfg.abstract_dim,
                num_cols=model_cfg.abstract_num_columns,
                hidden=model_cfg.abstract_enc_hidden,
                eps=model_cfg.A_eps,
            )
            self.abstract_decoder = Decoder(
                d=model_cfg.abstract_dim,
                num_classes=model_cfg.num_classes,
                hidden=model_cfg.abstract_dec_hidden,
            )
            self._build_abstract_topology()
            self.primer_topdown = (
                Primer(model_cfg.latent_dim, model_cfg.abstract_dim)
                if model_cfg.use_abstract_topdown_primer
                else None
            )
            # predictive-coding inter-layer maps (predictions down, errors up)
            self.predictive_maps = PredictiveMaps(
                d_bottom=model_cfg.latent_dim, d_top=model_cfg.abstract_dim)
        else:
            self.abstract_pos = None
            self.abstract_encoder = None
            self.abstract_decoder = None
            self.predictive_maps = None

    # ------------------------------------------------------------------ #
    # topology construction
    # ------------------------------------------------------------------ #

    def _build_bottom_topology(self):
        dummy = torch.zeros(1, 1, self.image_size, self.image_size)
        if self.decomposition_mode == "multi":
            assert self.data_cfg is not None
            roster = build_column_roster(self.image_size, self.data_cfg.multi_scale_specs)
            N = len(roster)
            coords = torch.tensor([[s.row, s.col] for s in roster], dtype=torch.long)
            if self.cfg.topology_mode == "allpairs":
                edges = build_all_pairs(N)
                D_init = init_conductance(edges, mode="dense", dense_value=self.cfg.D_init)
            elif self.cfg.topology_mode == "spatial":
                edges, ov = build_multi_scale_neighbors(roster, overlap_min=1)
                D_init = init_conductance(edges, mode="overlap", overlap=ov)
            else:
                raise ValueError(f"unknown topology_mode {self.cfg.topology_mode!r}")
            E = edges.shape[1]
            self._bottom_topo = TopologyCache(
                patch_dim=self.patch_dim, num_edges=E,
                edges=edges, coords=coords, D_init=D_init,
                roster=roster, pool_to=self.pool_to,
            )
            self._bottom_restriction = RestrictionMaps(num_edges=E, d=self.cfg.latent_dim)
            self._bottom_num_nodes = N
            return
        # single-scale
        deco = decompose(dummy, self.patch_size, self.stride)
        mode = self.cfg.topology_mode
        if mode == "allpairs":
            edges = build_all_pairs(deco.num_patches)
            D_init = init_conductance(edges, mode="dense", dense_value=self.cfg.D_init)
        elif mode == "spatial":
            edges, _ = build_spatial_neighbors(deco, self.patch_size)
            ov = overlap_weights(deco, self.patch_size, edges)
            D_init = init_conductance(edges, mode="overlap", overlap=ov)
        else:
            raise ValueError(f"unknown topology_mode {mode!r}")
        E = edges.shape[1]
        self._bottom_topo = TopologyCache(
            patch_dim=self.patch_dim, num_nodes=deco.num_patches,
            num_edges=E, edges=edges, coords=deco.coords, D_init=D_init,
        )
        self._bottom_restriction = RestrictionMaps(num_edges=E, d=self.cfg.latent_dim)
        self._bottom_num_nodes = deco.num_patches

    def _build_abstract_topology(self):
        N = self.cfg.abstract_num_columns
        edges = build_all_pairs(N)
        D_init = torch.full((edges.shape[1],), self.cfg.D_init)
        self._abstract_topo = TopologyCache(
            patch_dim=0, num_edges=edges.shape[1],
            edges=edges,
            coords=torch.zeros(N, 2, dtype=torch.long),
            D_init=D_init,
        )
        self._abstract_restriction = RestrictionMaps(
            num_edges=edges.shape[1], d=self.cfg.abstract_dim,
        )
        self._abstract_num_nodes = N

    def _ensure_device(self, t: TopologyCache, device: torch.device, dtype: torch.dtype):
        if t.edges.device != device:
            t.edges = t.edges.to(device)
            t.coords = t.coords.to(device)
            t.D_init = t.D_init.to(device)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def forward(
        self, images: torch.Tensor, record: bool = False
    ) -> tuple[torch.Tensor, tuple[ConsensusState, ConsensusState | None]]:
        if images.dim() != 4:
            raise ValueError(f"expected (B,1,H,W); got {tuple(images.shape)}")
        device, dtype = images.device, images.dtype
        self._ensure_device(self._bottom_topo, device, dtype)
        if self._abstract_topo is not None:
            self._ensure_device(self._abstract_topo, device, dtype)

        b_topo = self._bottom_topo

        # 1. decompose
        if self.decomposition_mode == "multi":
            assert b_topo.roster is not None
            msd = decompose_multi_scale(images, b_topo.roster, b_topo.pool_to)
            patches = msd.patches if msd.patches.dtype == dtype else msd.patches.to(dtype)
            coords = b_topo.coords.to(device)
        else:
            deco = decompose(images, self.patch_size, self.stride)
            patches = deco.patches if deco.patches.dtype == dtype else deco.patches.to(dtype)
            coords = deco.coords.to(device)

        # 2. bottom encode -> what + where -> A, b, s
        A, b, s = self.bottom_encoder(patches, coords)
        active = hard_topk_gate(s, k=self.topk_k, tau=self.topk_tau)

        # 2b. feedforward sweep warm start (each column's own primal — no primer,
        #     no broadcast; biologically the fast forward pass)
        x_preview = ACNCore.primal(A, b, torch.zeros_like(b), torch.zeros_like(b), self.rho.item())
        x_preview = x_preview * active.unsqueeze(-1)
        z_init = x_preview                                    # each column's OWN answer
        z_global0 = sparse_fuse_vectors(x_preview, active)    # feedforward-sweep aggregate

        r_ij, r_ji = self._bottom_restriction.maps()

        # 3-4. unified hierarchy with BPTT (detach_after=8)
        if self.use_abstract and self.abstract_encoder is not None and self._abstract_topo is not None:
            a_topo = self._abstract_topo
            A2, b2, s2 = self.abstract_encoder(z_global0, self.abstract_pos)
            active2 = hard_topk_gate(s2, k=self.cfg.abstract_topk, tau=self.topk_tau)
            r2_ij, r2_ji = self._abstract_restriction.maps()

            z2_init = None
            if self.primer_topdown is not None:
                z2_preview = self.primer_topdown(z_global0)
                z2_init = z2_preview.unsqueeze(1).expand(-1, self.cfg.abstract_num_columns, -1)

            state_bottom, state2 = ACNCore.run_hierarchical(
                A_b=A, b_b=b,
                r_ij_b=r_ij, r_ji_b=r_ji,
                edges_b=b_topo.edges, D_init_b=b_topo.D_init,
                active_b=active,
                A_t=A2, b_t=b2,
                r_ij_t=r2_ij, r_ji_t=r2_ji,
                edges_t=a_topo.edges, D_init_t=a_topo.D_init,
                active_t=active2,
                decode_down=self.predictive_maps.decode_down,
                encode_up=self.predictive_maps.encode_up,
                rounds=self.cfg.hierarchy_rounds,
                bottom_diffusion_steps=self.cfg.bottom_diffusion_steps,
                top_diffusion_steps=self.cfg.top_diffusion_steps,
                rho=self.rho.item(), eta_z=self.eta_z.item(),
                pc_eta_top=self.cfg.pc_eta_top,
                pc_eta_bottom=self.cfg.pc_eta_bottom,
                record=record,
                detach_after=self.cfg.detach_after,
                bottom_motor_system=self.bottom_motor,
                bottom_z_init=z_init,
                top_z_init=z2_init,
            )
            state_bottom.relevance_logits = s
            state2.relevance_logits = s2
            logits_bottom = self.bottom_decoder(state_bottom.z)
            state_bottom.logits = logits_bottom
            # cooperative confidence-weighted readout
            logits_abstract = self.abstract_decoder(state2.z)
            pred_top = sparse_fuse(logits_abstract, state2.active)
            pred_bottom = sparse_fuse(logits_bottom, state_bottom.active)
            w_top = logit_margin_confidence(pred_top)
            w_bot = logit_margin_confidence(pred_bottom)
            w = torch.stack([w_top, w_bot], dim=-1)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
            pred = w[:, 0:1] * pred_top + w[:, 1:2] * pred_bottom
            return pred, (state_bottom, state2)

        # no abstract layer: single-layer fallback
        state_bottom = ACNCore.run(
            A=A, b=b,
            r_ij=r_ij, r_ji=r_ji,
            edges=b_topo.edges,
            D_init=b_topo.D_init,
            num_nodes=b_topo.coords.shape[0],
            rounds=self.cfg.rounds,
            diffusion_steps=self.cfg.diffusion_steps,
            rho=self.rho.item(), eta_z=self.eta_z.item(),
            record=record,
            detach_after=self.cfg.detach_after,
            active=active,
            motor_system=self.bottom_motor,
            z_init=z_init,
        )
        state_bottom.relevance_logits = s
        logits_bottom = self.bottom_decoder(state_bottom.z)
        state_bottom.logits = logits_bottom
        pred_bottom = sparse_fuse(logits_bottom, state_bottom.active)
        return pred_bottom, (state_bottom, None)
