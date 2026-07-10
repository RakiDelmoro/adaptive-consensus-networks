"""Smoke tests for PCN-ACN (predictive-coding consensus, equilibrium prop).

Per-column readout (Sheaf-ADMM-style): the decoder runs on each column
individually, the label is broadcast to all columns, the global prediction is
the average of the per-column logits. Free state is {h1, ℓ}.
"""

import torch
from acn.config import get_preset
from acn.model import ACNv2


def test_mnist_forward():
    cfg = get_preset("mnist")
    model = ACNv2(cfg.data.image_size, cfg.data.patch_size, cfg.data.stride,
                  cfg.model, data_cfg=cfg.data)
    x = torch.randn(2, 1, 28, 28)
    pred, state = model(x)
    assert pred.shape == (2, 10)
    assert state.h1.shape == (2, model._N, cfg.model.latent_dim)
    assert state.llogits.shape == (2, model._N, 10)
    assert torch.isfinite(pred).all()


def test_poc_forward():
    cfg = get_preset("poc")
    model = ACNv2(cfg.data.image_size, cfg.data.patch_size, cfg.data.stride,
                  cfg.model, data_cfg=cfg.data)
    x = torch.randn(4, 1, 8, 8)
    pred, state = model(x)
    assert pred.shape == (4, 10)
    assert torch.isfinite(pred).all()


def test_eqprop_grad():
    """EP loss gives gradients to all subsystems, no BPTT."""
    cfg = get_preset("poc")
    model = ACNv2(cfg.data.image_size, cfg.data.patch_size, cfg.data.stride,
                  cfg.model, data_cfg=cfg.data)
    model.train()
    x = torch.randn(4, 1, 8, 8)
    y = torch.randint(0, 10, (4,))
    loss = model.eqprop_loss(x, y)
    loss.backward()
    assert model.column_encoder.mlp[0].weight.grad is not None
    assert model.lateral_predictor.mlp[0].weight.grad is not None
    assert model.column_decoder.mlp.weight.grad is not None
    assert torch.isfinite(loss)


def test_nudge_reaches_all():
    """G2: the nudge reaches every state variable (the prediction-error coupling)."""
    cfg = get_preset("poc")
    model = ACNv2(cfg.data.image_size, cfg.data.patch_size, cfg.data.stride,
                  cfg.model, data_cfg=cfg.data)
    model.eval()
    x = torch.randn(4, 1, 8, 8)
    y = torch.randint(0, 10, (4,))
    from acn.settle import settle
    with torch.no_grad():
        ctx, sc, state0 = model._prepare(x)
        s_free, _ = settle(state0, ctx, sc, beta=0.0, target=None, **model._settle_kwargs())
        s_nudged, _ = settle(s_free, ctx, sc, beta=cfg.model.beta, target=y,
                             **model._settle_kwargs())
    for kk in s_free.keys():
        sf = getattr(s_free, kk); sn = getattr(s_nudged, kk)
        frac = (sn - sf).norm().item() / (sf.norm().item() + 1e-6)
        assert frac > 0.005, f"{kk} did not move under nudge (frac={frac:.4f})"
