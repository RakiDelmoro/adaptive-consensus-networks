"""Smoke tests for the Lean Sheaf-ADMM consensus network."""

import torch

from adaptive_consensus import (
    AdaptiveConsensusModel,
    ModelConfig,
    build_grid,
    patchify,
    sinusoidal_pos_code,
)
from adaptive_consensus.graph import add_noise, add_padding


def _make_model(cfg=None):
    cfg = cfg or ModelConfig()
    ei, _npos, _gh, _gw, n = build_grid(
        cfg.image_size, cfg.patch_size, cfg.stride, cfg.connectivity)
    return AdaptiveConsensusModel(
        cfg, patch_dim=cfg.patch_size ** 2, edge_indices=ei, n_agents=n), ei, n


def test_grid_structure():
    cfg = ModelConfig()
    ei, _npos, gh, gw, n = build_grid(
        cfg.image_size, cfg.patch_size, cfg.stride, cfg.connectivity)
    assert (gh, gw) == (7, 7)
    assert n == 49
    assert ei.shape[1] == 2
    assert ei.shape[0] == 84          # 4-connected 7x7 grid edges


def test_patchify_shape():
    img = torch.randn(2, 1, 28, 28)
    p = patchify(img, 4, 4)
    assert p.shape == (2, 49, 16)


def test_forward_shape_and_grad():
    cfg = ModelConfig(K=4, T=2)
    model, _ei, _n = _make_model(cfg)
    img = torch.randn(3, 1, 28, 28)
    patches = patchify(img, cfg.patch_size, cfg.stride).permute(1, 0, 2)
    logits, aux = model(patches)                       # default K (training)
    assert logits.shape == (3, 10)
    assert aux["n_rounds"] == cfg.K
    assert aux["per_round_logits"].shape == (cfg.K, 49, 3, 10)
    # eval uses more rounds (K_eval > K)
    logits_e, aux_e = model(patches, K=cfg.K_eval)
    assert aux_e["n_rounds"] == cfg.K_eval
    assert aux_e["per_round_logits"].shape[0] == cfg.K_eval
    labels = torch.randint(0, 10, (3,))
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    # default objective_mode is lasso -> encoder has q_diag_head, not L_head
    assert model.encoder.q_diag_head.weight.grad is not None
    assert model.restriction.F.grad is not None
    assert model.decoder.net[-1].weight.grad is not None
    assert model.rho_log.grad is not None


def test_no_gate_or_halting_modules():
    """The lean model has no learned gate/halting networks."""
    model, _ei, _n = _make_model()
    assert not hasattr(model, "edge_gate")
    assert not hasattr(model, "channel_gate")
    assert not hasattr(model, "halting")
    assert not hasattr(model, "theta_edge")
    assert not hasattr(model, "theta_channel")
    assert not hasattr(model, "theta_halt")


def test_train_eval_same_mechanism():
    cfg = ModelConfig(K=3, T=1)
    model, _ei, _n = _make_model(cfg)
    model.eval()
    img = torch.randn(2, 1, 28, 28)
    patches = patchify(img, cfg.patch_size, cfg.stride).permute(1, 0, 2)
    with torch.no_grad():
        l1, _ = model(patches)
        l2, _ = model(patches)
    assert torch.allclose(l1, l2)


def test_convergence_log():
    cfg = ModelConfig(K=3, K_eval=5, T=1)
    model, _ei, _n = _make_model(cfg)
    img = torch.randn(2, 1, 28, 28)
    patches = patchify(img, cfg.patch_size, cfg.stride).permute(1, 0, 2)
    _, aux = model(patches, K=cfg.K_eval)
    log = aux["conv_log"]
    assert log.avg_disagreement.shape == (cfg.K_eval,)
    assert log.edge_energy.shape[0] == cfg.K_eval
    assert log.edge_energy.shape[1] == model.num_edges
    assert log.t_used.shape == (cfg.K_eval,)


def test_corruptions_shapes():
    img = torch.rand(4, 1, 28, 28)
    assert add_noise(img, 0.3).shape == img.shape
    assert (add_noise(img, 0.3) >= 0).all() and (add_noise(img, 0.3) <= 1).all()
    padded = add_padding(img, 4, size=28)
    assert padded.shape == img.shape


# ---------------------------------------------------------------------------
# Reference-frame additions (Tier 1)
# ---------------------------------------------------------------------------

def test_sinusoidal_pos_code():
    pe = sinusoidal_pos_code(49, 8)
    assert pe.shape == (49, 8)
    # distinct positions have distinct codes
    assert not torch.allclose(pe[0], pe[1])
    assert not torch.allclose(pe[0], pe[7])   # different row
    # d_pos=0 yields an empty code
    assert sinusoidal_pos_code(49, 0).shape == (49, 0)


def test_reference_frame_code_wired_into_encoder():
    cfg = ModelConfig(d_pos=8, K=2, T=1)
    model, _ei, _n = _make_model(cfg)
    assert model.pos_code.shape == (49, 8)
    # encoder input dim must be patch_dim + d_pos
    assert model.encoder.trunk[0].in_features == cfg.patch_size ** 2 + 8


def test_confidence_weighted_voting_shapes_and_grad():
    cfg = ModelConfig(confidence_weighted=True, K=3, T=1)
    model, _ei, _n = _make_model(cfg)
    img = torch.randn(3, 1, 28, 28)
    patches = patchify(img, cfg.patch_size, cfg.stride).permute(1, 0, 2)
    logits, aux = model(patches)
    assert logits.shape == (3, 10)
    # decoder has a confidence head with grad
    labels = torch.randint(0, 10, (3,))
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert model.decoder.conf_head[-1].weight.grad is not None


def test_uniform_voting_legacy_path():
    """confidence_weighted=False reproduces the uniform mean-softmax vote."""
    cfg = ModelConfig(confidence_weighted=False, d_pos=0, K=2, T=1)
    model, _ei, _n = _make_model(cfg)
    img = torch.randn(2, 1, 28, 28)
    patches = patchify(img, cfg.patch_size, cfg.stride).permute(1, 0, 2)
    logits, aux = model(patches)
    # recompute the uniform vote by hand from per_round_logits
    last = aux["per_round_logits"][-1]               # (N, B, C)
    probs = torch.softmax(last, dim=-1)
    expected = torch.log(probs.mean(0).clamp_min(1e-8))
    assert torch.allclose(logits, expected, atol=1e-5)


def test_disagreement_signals_are_differentiable():
    cfg = ModelConfig(edge_energy_weight=0.01, K=3, T=1)
    model, _ei, _n = _make_model(cfg)
    img = torch.randn(2, 1, 28, 28)
    patches = patchify(img, cfg.patch_size, cfg.stride).permute(1, 0, 2)
    logits, aux = model(patches)
    assert aux["edge_energy_final"].shape[0] == model.num_edges
    assert aux["edge_energy_final"].shape[1] == 2
    assert aux["disagreement_final"].dim() == 0
    loss = aux["edge_energy_final"].mean() + aux["disagreement_final"]
    loss.backward()
    # the restriction maps are what the edge-energy regularizer pushes on
    assert model.restriction.F.grad is not None
    assert torch.isfinite(model.restriction.F.grad).all()


def test_forward_preserves_output_shape():
    """Default (lasso + cg_project) forward produces the expected shapes."""
    cfg = ModelConfig(K=3, T=1)
    model, _ei, _n = _make_model(cfg)
    img = torch.randn(2, 1, 28, 28)
    patches = patchify(img, cfg.patch_size, cfg.stride).permute(1, 0, 2)
    logits, aux = model(patches)
    assert logits.shape == (2, 10)
    assert aux["per_round_logits"].shape == (3, 49, 2, 10)


# ---------------------------------------------------------------------------
# Solver tests (lasso diagonal-prox x-update, cg_project z-update)
# ---------------------------------------------------------------------------

def test_default_config_is_lasso_cg_project():
    cfg = ModelConfig()
    assert cfg.objective_mode == "lasso"
    assert cfg.z_solver == "cg_project"
    assert cfg.l1_weight == 0.00634
    assert cfg.tikhonov_eps == 1e-5


def test_lasso_encoder_outputs():
    cfg = ModelConfig(objective_mode="lasso", K=2, T=1)
    model, _ei, _n = _make_model(cfg)
    assert hasattr(model.encoder, "q_diag_head")
    assert hasattr(model.encoder, "l1_head")
    assert not hasattr(model.encoder, "L_head")
    # encoder input includes the d_pos reference-frame code (patch_dim + d_pos)
    img = torch.randn(2, 1, 28, 28)
    patches = patchify(img, 4, 4).permute(1, 0, 2)              # (49, 2, 16)
    pos = model.pos_code.unsqueeze(1).expand(-1, 2, -1)         # (49, 2, 8)
    enc_in = torch.cat([patches, pos], dim=-1)                  # (49, 2, 24)
    enc_out = model.encoder(enc_in)
    assert set(enc_out) == {"q_diag", "q", "l1_weight"}
    assert enc_out["q_diag"].shape == (49, 2, 16)
    assert (enc_out["q_diag"] > 0).all()       # strictly positive curvature
    assert (enc_out["l1_weight"] >= 0).all()    # non-negative L1


def test_quadratic_encoder_outputs():
    cfg = ModelConfig(objective_mode="quadratic", K=2, T=1)
    model, _ei, _n = _make_model(cfg)
    assert hasattr(model.encoder, "L_head")
    assert not hasattr(model.encoder, "q_diag_head")
    img = torch.randn(2, 1, 28, 28)
    patches = patchify(img, 4, 4).permute(1, 0, 2)              # (49, 2, 16)
    pos = model.pos_code.unsqueeze(1).expand(-1, 2, -1)         # (49, 2, 8)
    enc_in = torch.cat([patches, pos], dim=-1)                  # (49, 2, 24)
    enc_out = model.encoder(enc_in)
    assert set(enc_out) == {"L", "p"}
    assert enc_out["L"].shape == (49, 2, 16, 4)


def test_lasso_diagonal_prox_soft_thresholds():
    """The lasso x-update should zero out components below the L1 threshold."""
    z = torch.zeros(1, 1, 4)
    u = torch.zeros(1, 1, 4)
    q_diag = torch.ones(1, 1, 4)
    q = torch.zeros(1, 1, 4)
    l1 = torch.full((1, 1, 4), 0.5)
    rho = 1.0
    x = AdaptiveConsensusModel._local_solve_lasso(z, u, q_diag, q, l1, rho)
    # v = z - u = 0, t = 0, soft_threshold(0, thr>0) = 0 -> x all zero
    assert torch.all(x == 0)
    # with a strong signal, large components survive, small ones zeroed
    z2 = torch.tensor([[[0.3, 2.0, -1.5, 0.2]]])
    x2 = AdaptiveConsensusModel._local_solve_lasso(z2, u, q_diag, q, l1, rho)
    # a = 1 + 0 + 1 = 2, thr = 0.5/2 = 0.25; t = z/2
    # |t|: 0.15, 1.0, 0.75, 0.1 -> zeros where |t| < 0.25 (dims 0, 3)
    assert x2[0, 0, 0] == 0 and x2[0, 0, 3] == 0
    assert x2[0, 0, 1] != 0 and x2[0, 0, 2] != 0


def test_cg_project_reduces_edge_energy():
    """The cg_project z-update should reduce the sheaf edge energy (toward Fz=0)."""
    cfg = ModelConfig(z_solver="cg_project", T=5, K=2, tikhonov_eps=1e-5)
    model, _ei, _n = _make_model(cfg)
    Fm = model.restriction()
    z_target = torch.randn(49, 2, 16)
    z_prev = torch.zeros(49, 2, 16)
    e_before = model._edge_energy(z_target, Fm).mean().item()
    z_new = model._cg_project(z_target, z_prev, Fm)
    e_after = model._edge_energy(z_new, Fm).mean().item()
    assert e_after < e_before


def test_defaults_train_and_grad():
    """End-to-end: default config trains and backprops."""
    cfg = ModelConfig(K=3, T=3)
    model, _ei, _n = _make_model(cfg)
    img = torch.randn(3, 1, 28, 28)
    patches = patchify(img, 4, 4).permute(1, 0, 2)
    logits, aux = model(patches)
    labels = torch.randint(0, 10, (3,))
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert model.encoder.q_diag_head.weight.grad is not None
    assert model.encoder.l1_head.weight.grad is not None
    assert model.restriction.F.grad is not None
    assert model.rho_log.grad is not None
    assert torch.isfinite(logits).all()


def test_legacy_quadratic_gd_mode_still_works():
    """The legacy quadratic + gd mode still runs end-to-end."""
    cfg = ModelConfig(objective_mode="quadratic", z_solver="gd", K=3, T=2)
    model, _ei, _n = _make_model(cfg)
    img = torch.randn(2, 1, 28, 28)
    patches = patchify(img, 4, 4).permute(1, 0, 2)
    logits, aux = model(patches)
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


def test_default_readout_is_x():
    """Default dec_readout='x' (local proposal, not consensus)."""
    cfg = ModelConfig()
    assert cfg.dec_readout == "x"


def test_readout_x_vs_z_produces_different_outputs():
    """Decoding x vs z should give different predictions (they're different vars)."""
    torch.manual_seed(0)
    m_x, _ei, _n = _make_model(ModelConfig(dec_readout="x", K=3, T=1))
    torch.manual_seed(0)
    m_z, _ei, _n = _make_model(ModelConfig(dec_readout="z", K=3, T=1))
    img = torch.randn(2, 1, 28, 28)
    patches = patchify(img, 4, 4).permute(1, 0, 2)
    torch.manual_seed(0)
    ox, _ = m_x(patches)
    torch.manual_seed(0)
    oz, _ = m_z(patches)
    assert not torch.allclose(ox, oz)
