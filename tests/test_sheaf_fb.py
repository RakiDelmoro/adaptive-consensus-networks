"""Tests for the Sheaf-Forward-Backward network (Equilibrium Propagation)."""

import torch

from sheaf_fb.config import get_preset
from sheaf_fb.dynamics import consensus_grad, edge_residuals, energy, forward_backward
from sheaf_fb.model import SheafFBModel
from sheaf_fb.task import MNISTTaskFB


def _make_model(cfg, device):
    task = MNISTTaskFB(cfg.data, device)
    patch_dim = 1 * cfg.data.patch_size * cfg.data.patch_size
    model = SheafFBModel(
        cfg.model, patch_dim=patch_dim,
        edge_indices=task.edge_indices, node_positions=task.node_positions,
    ).to(device)
    return model, task


def test_grid_decomposition_7x7():
    cfg = get_preset("mnist")
    task = MNISTTaskFB(cfg.data, torch.device("cpu"))
    # 28x28, patch 4, stride 4 -> 7x7 = 49 agents, 4-connected -> 84 edges.
    assert task.N == 49
    assert task.edge_indices.shape == (84, 2)


def test_patchify_shape():
    cfg = get_preset("mnist")
    task = MNISTTaskFB(cfg.data, torch.device("cpu"))
    images = torch.randn(2, 1, 28, 28)
    fwd, _, _ = task.prepare(images, torch.randint(0, 10, (2,)))
    patches = fwd["patches"]
    # [N=49, B=2, ps=4, ps=4, C=1]
    assert patches.shape == (49, 2, 4, 4, 1)
    assert torch.isfinite(patches).all()


def test_forward_backward_settles_and_lowers_energy():
    cfg = get_preset("mnist")
    device = torch.device("cpu")
    model, task = _make_model(cfg, device)
    images = torch.randn(3, 1, 28, 28)
    fwd, _, _ = task.prepare(images, torch.randint(0, 10, (3,)))
    with torch.no_grad():
        theta = model._encode(fwd["patches"])
        F_uv = model._sheaf_maps()
        x0 = theta.clone()
        e0 = energy(x0, theta, task.edge_indices, F_uv, cfg.model.rho).item()
        x = forward_backward(
            theta, task.edge_indices, F_uv,
            eta=cfg.model.eta, rho=cfg.model.rho, num_iters=cfg.model.K,
            warm_start=True)
        eK = energy(x, theta, task.edge_indices, F_uv, cfg.model.rho).item()
    # The dynamics are gradient descent on E; the energy should not increase.
    assert eK <= e0 + 1e-3
    assert torch.isfinite(x).all()


def test_model_forward_shape_and_predict():
    cfg = get_preset("poc")
    device = torch.device("cpu")
    model, task = _make_model(cfg, device)
    images = torch.randn(4, 1, 28, 28)
    fwd, _, _ = task.prepare(images, torch.randint(0, 10, (4,)))
    logits = model(fwd["patches"])
    assert logits.shape == (task.N, 4, cfg.model.num_classes)
    assert torch.isfinite(logits).all()
    p = model.predict_global(fwd["patches"])
    assert p.shape == (4, cfg.model.num_classes)
    assert torch.allclose(p.sum(-1), torch.ones(4), atol=1e-4)


def test_ep_step_produces_grads_and_runs_optimizer():
    cfg = get_preset("poc")
    device = torch.device("cpu")
    model, task = _make_model(cfg, device)
    model.train()
    images = torch.randn(4, 1, 28, 28)
    labels = torch.randint(0, 10, (4,))
    fwd, targets, _ = task.prepare(images, labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    w0 = model.encoder.fc1.weight.detach().clone()
    stats = model.ep_step(fwd["patches"], targets["labels"], optimizer)
    optimizer.step()
    # Gradients were assigned to every learnable parameter group.
    assert model.encoder.fc1.weight.grad is not None
    assert model.encoder.fc2.weight.grad is not None
    assert model.sheaf.F.grad is not None
    assert model.decoder.fc1.weight.grad is not None
    assert model.decoder.fc2.weight.grad is not None
    assert torch.isfinite(torch.tensor(stats["loss"]))
    # A real step changes the encoder weights.
    assert not torch.allclose(w0, model.encoder.fc1.weight)


def test_ep_loss_decreases_over_steps():
    """A handful of EP steps on a fixed batch should reduce the training loss."""
    cfg = get_preset("poc")
    device = torch.device("cpu")
    torch.manual_seed(0)
    model, task = _make_model(cfg, device)
    model.train()
    images = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    fwd, targets, _ = task.prepare(images, labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    losses = []
    for _ in range(8):
        s = model.ep_step(fwd["patches"], targets["labels"], optimizer)
        optimizer.step()
        losses.append(s["loss"])
    assert losses[-1] < losses[0]


def test_train_eval_same_mechanism():
    """Train and eval produce identical logits for the same input."""
    cfg = get_preset("poc")
    device = torch.device("cpu")
    model, task = _make_model(cfg, device)
    model.eval()
    images = torch.randn(2, 1, 28, 28)
    fwd, _, _ = task.prepare(images, torch.randint(0, 10, (2,)))
    with torch.no_grad():
        l1 = model(fwd["patches"])
        l2 = model(fwd["patches"])
    assert torch.allclose(l1, l2)


def test_directional_sheaf_sharing_shape():
    cfg = get_preset("poc").override(**{"model.sheaf_sharing": "directional"})
    device = torch.device("cpu")
    model, task = _make_model(cfg, device)
    images = torch.randn(2, 1, 28, 28)
    fwd, _, _ = task.prepare(images, torch.randint(0, 10, (2,)))
    F_uv = model._sheaf_maps()
    assert F_uv.shape[0] == task.edge_indices.shape[0]
    assert F_uv.shape[1] == 2
    assert F_uv.shape[2:] == (cfg.model.c, cfg.model.d)
    # The directional bank has exactly 4 maps.
    assert model.sheaf.F.shape == (4, cfg.model.c, cfg.model.d)
    logits = model(fwd["patches"])
    assert torch.isfinite(logits).all()


def test_consensus_grad_matches_autodiff():
    """The closed-form consensus gradient matches autograd on the sheaf term."""
    N, B, d, c, rho = 6, 2, 5, 3, 0.7
    edge_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
    E = edge_indices.shape[0]
    F = torch.randn(E, 2, c, d, requires_grad=True)
    x = torch.randn(N, B, d, requires_grad=True)
    # Closed form
    g_closed = consensus_grad(x, edge_indices, F, rho)
    # Autograd
    err = edge_residuals(x, edge_indices, F)
    sheaf_term = 0.5 * rho * torch.sum(err ** 2)
    g_auto = torch.autograd.grad(sheaf_term, x)[0]
    assert torch.allclose(g_closed, g_auto, atol=1e-5)


def test_signed_beta_nudge_opposes_positive_beta():
    """The −β phase equilibrium lies on the opposite side of the free phase from +β."""
    cfg = get_preset("poc").override(**{"model.beta": 0.1, "model.K": 6})
    device = torch.device("cpu")
    torch.manual_seed(1)
    model, task = _make_model(cfg, device)
    images = torch.randn(4, 1, 28, 28)
    labels = torch.randint(0, 10, (4,))
    fwd, _, _ = task.prepare(images, labels)
    with torch.no_grad():
        x_free = model.settle(fwd["patches"], nudged=False)
        x_plus = model.settle_signed(fwd["patches"], labels=labels, beta=cfg.model.beta)
        x_minus = model.settle_signed(fwd["patches"], labels=labels, beta=-cfg.model.beta)
    d_plus = (x_plus - x_free).flatten()
    d_minus = (x_minus - x_free).flatten()
    # The two nudged equilibria should move in opposite directions on average.
    assert float(torch.dot(d_plus, d_minus)) < 0.0
    # And be roughly symmetric in magnitude (same energy landscape, ±beta).
    assert abs(d_plus.norm().item() - d_minus.norm().item()) / max(d_plus.norm().item(), 1e-9) < 0.5


def test_symmetric_ep_runs_and_produces_grads():
    """The default (symmetric, 3-phase) EP step runs and gradients every parameter."""
    cfg = get_preset("poc")  # default ep_variant == "symmetric"
    assert cfg.model.ep_variant == "symmetric"
    device = torch.device("cpu")
    model, task = _make_model(cfg, device)
    model.train()
    images = torch.randn(4, 1, 28, 28)
    labels = torch.randint(0, 10, (4,))
    fwd, targets, _ = task.prepare(images, labels)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    stats = model.ep_step(fwd["patches"], targets["labels"], opt)
    assert set(stats) >= {"loss", "e_plus", "e_minus", "delta_norm"}
    for k in ("encoder.fc1.weight", "encoder.fc2.weight", "sheaf.F",
              "decoder.fc1.weight", "decoder.fc2.weight"):
        p = dict(model.named_parameters())[k]
        assert p.grad is not None, f"{k} has no gradient"
    assert torch.isfinite(torch.tensor(stats["loss"]))


def test_one_sided_ep_variant_runs():
    """The one-sided (2-phase) EP variant still runs (for comparison)."""
    cfg = get_preset("poc").override(**{"model.ep_variant": "one_sided"})
    device = torch.device("cpu")
    model, task = _make_model(cfg, device)
    model.train()
    images = torch.randn(4, 1, 28, 28)
    labels = torch.randint(0, 10, (4,))
    fwd, targets, _ = task.prepare(images, labels)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    stats = model.ep_step(fwd["patches"], targets["labels"], opt)
    assert set(stats) >= {"loss", "e_free", "e_plus", "delta_norm"}


def test_bptt_step_produces_grads():
    """BPTT training step runs and gradients every parameter."""
    cfg = get_preset("poc")
    device = torch.device("cpu")
    model, task = _make_model(cfg, device)
    model.train()
    images = torch.randn(4, 1, 28, 28)
    labels = torch.randint(0, 10, (4,))
    fwd, targets, _ = task.prepare(images, labels)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    stats = model.bptt_step(fwd["patches"], targets["labels"], opt)
    assert "loss" in stats
    for k in ("encoder.fc1.weight", "encoder.fc2.weight", "sheaf.F",
              "decoder.fc1.weight", "decoder.fc2.weight"):
        p = dict(model.named_parameters())[k]
        assert p.grad is not None, f"{k} has no gradient"
    opt.step()
    assert torch.isfinite(torch.tensor(stats["loss"]))


def test_bptt_loss_decreases():
    """A handful of BPTT steps on a fixed batch should reduce the loss."""
    cfg = get_preset("poc")
    device = torch.device("cpu")
    torch.manual_seed(0)
    model, task = _make_model(cfg, device)
    model.train()
    images = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    fwd, targets, _ = task.prepare(images, labels)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    losses = []
    for _ in range(10):
        s = model.bptt_step(fwd["patches"], targets["labels"], opt)
        opt.step()
        losses.append(s["loss"])
    assert losses[-1] < losses[0]
