"""Grid graph + patchify for MNIST (no CNN — patches are flat vectors).

28x28, patch 4, stride 4 -> 7x7 = 49 agents, each owning a non-overlapping 4x4
patch (16 pixels), with 4-connected grid adjacency (84 undirected edges).
"""

from __future__ import annotations

import math

import torch


def grid_shape(image_size: int, patch_size: int, stride: int) -> tuple[int, int]:
    gh = (image_size - patch_size) // stride + 1
    gw = (image_size - patch_size) // stride + 1
    return gh, gw


def build_grid(image_size: int, patch_size: int, stride: int, connectivity: int = 4):
    """Return ``(edge_indices, node_positions, gh, gw)``.

    * ``edge_indices``: ``(E, 2)`` undirected grid edges (each stored once).
    * ``node_positions``: ``(N, 2)`` ``(y, x)`` center coords, row-major.
    * ``gh, gw``: grid side lengths (7, 7 for the default).
    """
    gh, gw = grid_shape(image_size, patch_size, stride)
    n = gh * gw

    # center coords
    c = patch_size // 2
    coords = [(c + r * stride, c + cc * stride) for r in range(gh) for cc in range(gw)]
    node_positions = torch.tensor(coords, dtype=torch.float32)  # (N, 2) yx

    def idx(r, cc):
        return r * gw + cc

    edges = []
    for r in range(gh):
        for cc in range(gw):
            i = idx(r, cc)
            if cc + 1 < gw:
                edges.append((i, idx(r, cc + 1)))     # east
            if r + 1 < gh:
                edges.append((i, idx(r + 1, cc)))     # south
            if connectivity >= 8:
                if r + 1 < gh and cc + 1 < gw:
                    edges.append((i, idx(r + 1, cc + 1)))   # south-east
                if r + 1 < gh and cc - 1 >= 0:
                    edges.append((i, idx(r + 1, cc - 1)))   # south-west
    edge_indices = torch.tensor(edges, dtype=torch.long)
    assert edge_indices.shape[0] > 0
    return edge_indices, node_positions, gh, gw, n


def patchify(image: torch.Tensor, patch_size: int, stride: int) -> torch.Tensor:
    """Split ``(B, 1, H, W)`` into non-overlapping ``patch_size`` patches.

    Returns ``(B, N, patch_size**2)`` in row-major order. Requires
    ``(H - patch_size) % stride == 0``.
    """
    b, c, h, w = image.shape
    gh = (h - patch_size) // stride + 1
    gw = (w - patch_size) // stride + 1
    p = image.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    # (B, C, gh, gw, ps, ps) -> (B, gh*gw, C*ps*ps)
    p = p.contiguous().reshape(b, gh * gw, c * patch_size * patch_size)
    return p


def add_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    return (images + torch.randn_like(images) * sigma).clamp(0.0, 1.0)


def add_padding(images: torch.Tensor, pad: int, size: int = 28) -> torch.Tensor:
    if pad <= 0:
        return images
    padded = torch.nn.functional.pad(images, (pad, pad, pad, pad), mode="constant", value=0.0)
    return torch.nn.functional.interpolate(
        padded, size=(size, size), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Reference-frame code ("where" / grid-cell signal)
# ---------------------------------------------------------------------------

def sinusoidal_pos_code(n_agents: int, d_pos: int, *,
                        device: torch.device | None = None,
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return a 2D sinusoidal reference-frame code ``(n_agents, d_pos)``.

    The grid is assumed to be row-major (matching :func:`build_grid`): a square
    grid when ``n_agents`` is a perfect square, otherwise the smallest
    ``gh x gw`` rectangle with ``gh*gw >= n_agents``. The code is split into a
    y-half and an x-half, each a standard sinusoidal positional encoding of the
    row / column index. This is the "where" signal that binds each column's
    features to a location in the shared reference frame.
    """
    if d_pos <= 0:
        return torch.zeros(n_agents, 0, device=device, dtype=dtype)

    gh = int(math.isqrt(n_agents))
    if gh * gh == n_agents:
        gw = gh
    else:
        gh = int(math.ceil(math.sqrt(n_agents)))
        gw = int(math.ceil(n_agents / gh))

    half = max(d_pos // 2, 1)

    def axis(positions: torch.Tensor, dim: int) -> torch.Tensor:
        dim = max(dim, 2)
        freq = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / dim))
        ang = positions.unsqueeze(-1) * freq.unsqueeze(0)   # (L, dim//2)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (L, dim)

    ys = torch.arange(gh, device=device, dtype=dtype)
    xs = torch.arange(gw, device=device, dtype=dtype)
    py = axis(ys, half)                                   # (gh, half)
    px = axis(xs, half)                                   # (gw, half)

    pe = torch.zeros(n_agents, 2 * half, device=device, dtype=dtype)
    for r in range(gh):
        for c in range(gw):
            i = r * gw + c
            if i >= n_agents:
                continue
            pe[i, :half] = py[r]
            pe[i, half:] = px[c]

    if d_pos < 2 * half:
        pe = pe[:, :d_pos]
    elif d_pos > 2 * half:
        pe = torch.nn.functional.pad(pe, (0, d_pos - 2 * half))
    return pe
