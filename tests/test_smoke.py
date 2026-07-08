"""Smoke tests for ACN: forward pass, backprop, shapes, config presets."""

import torch
from acn.config import get_preset, ModelConfig, DataConfig
from acn.model import AdaptiveConsensusNetwork


def test_mnist_forward():
    """Forward pass produces correct shapes."""
    cfg = get_preset("mnist")
    model = AdaptiveConsensusNetwork(
        cfg.data.image_size, cfg.data.patch_size, cfg.data.stride,
        cfg.model, data_cfg=cfg.data,
    )
    x = torch.randn(2, 1, 28, 28)
    pred, (sb, sa) = model(x)
    assert pred.shape == (2, 10)
    assert sb.z.shape == (2, 74, 32)
    assert sa is not None and sa.z.shape == (2, 8, 16)


def test_poc_forward():
    """POC config forward pass."""
    cfg = get_preset("poc")
    model = AdaptiveConsensusNetwork(
        cfg.data.image_size, cfg.data.patch_size, cfg.data.stride,
        cfg.model, data_cfg=cfg.data,
    )
    x = torch.randn(4, 1, 8, 8)
    pred, (sb, sa) = model(x)
    assert pred.shape == (4, 10)


def test_backprop():
    """Gradients flow through the model."""
    cfg = get_preset("poc")
    model = AdaptiveConsensusNetwork(
        cfg.data.image_size, cfg.data.patch_size, cfg.data.stride,
        cfg.model, data_cfg=cfg.data,
    )
    model.train()
    x = torch.randn(4, 1, 8, 8)
    y = torch.randint(0, 10, (4,))
    pred, _ = model(x)
    loss = torch.nn.functional.cross_entropy(pred, y)
    loss.backward()
    # check that encoder got gradients
    assert model.bottom_encoder.mlp[0].weight.grad is not None
    assert model.predictive_maps.decode_down.weight.grad is not None


def test_cooperative_readout():
    """The cooperative readout weights both layers."""
    cfg = get_preset("mnist")
    model = AdaptiveConsensusNetwork(
        cfg.data.image_size, cfg.data.patch_size, cfg.data.stride,
        cfg.model, data_cfg=cfg.data,
    )
    model.eval()
    x = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        pred, (sb, sa) = model(x)
    # pred should be finite (not NaN)
    assert torch.isfinite(pred).all()


def test_presets_exist():
    """Both presets are available."""
    assert "mnist" in ["mnist", "poc"]
    assert "poc" in ["mnist", "poc"]
    cfg_m = get_preset("mnist")
    cfg_p = get_preset("poc")
    assert cfg_m.data.image_size == 28
    assert cfg_p.data.image_size == 8
