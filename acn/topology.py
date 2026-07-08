"""Edge topology for ACN mini networks.

Two builders:
  * :func:`build_spatial_neighbors` — edges only between overlapping patches
    (the original; consensus is local — distant active columns can't talk).
  * :func:`build_all_pairs` — a complete graph over all columns, so any pair of
    active columns is wired regardless of image position (global consensus;
    "active columns talk to each other wherever they are"). The gate's
    `edge_mask` still switches off wires touching inactive columns, so for a
    given input only the active pairs actually carry flow.

We also expose conductance initialization strategies.
"""

from __future__ import annotations

import torch

from acn.decomposition import Decomposition, ColumnSpec


def build_spatial_neighbors(
    deco: Decomposition, patch_size: int, overlap_min: int = 1
) -> tuple[torch.Tensor, list[list[int]]]:
    """Build edges between overlapping patches.

    Two patches are neighbors iff their patches overlap by at least `overlap_min`
    pixels along BOTH axes.

    Returns:
      edges: (2, E) long tensor of (i, j) with i < j, undirected, no self-loops.
      neighbors: list of length N; neighbors[i] = sorted list of adjacent node ids.
    """
    coords = deco.coords  # (N, 2)
    N = coords.shape[0]
    r = coords[:, 0]
    c = coords[:, 1]

    # pairwise overlap along each axis: max(0, p - |d|)
    dr = (r[:, None] - r[None, :]).abs()
    dc = (c[:, None] - c[None, :]).abs()
    overlap_r = (patch_size - dr).clamp(min=0)
    overlap_c = (patch_size - dc).clamp(min=0)
    overlap = torch.minimum(overlap_r, overlap_c)  # min overlap along both axes

    # adjacency mask: overlap >= overlap_min and i != j
    mask = (overlap >= overlap_min)
    mask.fill_diagonal_(False)
    # upper triangle (i < j) for unique undirected edges
    tri = torch.triu(torch.ones_like(mask), diagonal=1).bool()
    mask = mask & tri
    edges = mask.nonzero(as_tuple=False).T  # (2, E)

    neighbors: list[list[int]] = [[] for _ in range(N)]
    if edges.numel() > 0:
        ei, ej = edges[0].tolist(), edges[1].tolist()
        for i, j in zip(ei, ej):
            neighbors[i].append(j)
            neighbors[j].append(i)
    return edges.contiguous(), neighbors


def build_all_pairs(num_nodes: int) -> torch.Tensor:
    """Complete graph over all columns: every (i, j) with i < j, shape (2, E).

    For N=64 → E = N*(N-1)/2 = 2016. The `edge_mask` (active[ei]*active[ej])
    in the consensus loop then keeps only the active-active pairs live per
    input, so this realizes "every active column is wired to every other active
    column, regardless of image position" without per-sample edge handling.
    """
    idx = torch.triu_indices(num_nodes, num_nodes, offset=1)  # (2, E)
    return idx.contiguous()


def build_multi_scale_neighbors(
    roster: list[ColumnSpec], overlap_min: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build edges between overlapping patches of DIFFERENT sizes.

    Two columns are neighbors iff their patches (which may be different sizes)
    overlap by >= overlap_min pixels on BOTH axes. This generalizes
    :func:`build_spatial_neighbors` to mixed receptive fields.

    A small 4x4 patch can be entirely inside a big 8x8 patch — that counts as
    neighbors (they share pixels). The Physarum dynamics decide which wires
    actually matter.

    Returns:
      edges: (2, E) long tensor of (i, j) with i < j, undirected, no self-loops.
      overlap: (E,) float tensor of the min-overlap-along-both-axes per edge
               (used for conductance init).
    """
    N = len(roster)
    ei_list, ej_list, ov_list = [], [], []
    for i in range(N):
        si, ri, ci = roster[i].size, roster[i].row, roster[i].col
        for j in range(i + 1, N):
            sj, rj, cj = roster[j].size, roster[j].row, roster[j].col
            # rectangle intersection along both axes (patches may differ in size)
            overlap_r = max(0, min(ri + si, rj + sj) - max(ri, rj))
            overlap_c = max(0, min(ci + si, cj + sj) - max(ci, cj))
            ov = min(overlap_r, overlap_c)
            if ov >= overlap_min:
                ei_list.append(i)
                ej_list.append(j)
                ov_list.append(float(ov))
    if not ei_list:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)
    edges = torch.tensor([ei_list, ej_list], dtype=torch.long)  # (2, E)
    overlap = torch.tensor(ov_list, dtype=torch.float32)        # (E,)
    return edges, overlap


def edge_to_pair_index(edges: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """Return edges as a (E, 2) long tensor on the given device."""
    return edges.T.contiguous().to(device=device, dtype=torch.long)


def overlap_weights(
    deco: Decomposition, patch_size: int, edges: torch.Tensor
) -> torch.Tensor:
    """Per-edge overlap area (min overlap along both axes), shape (E,)."""
    coords = deco.coords
    ei, ej = edges[0], edges[1]
    dr = (coords[ei, 0] - coords[ej, 0]).abs()
    dc = (coords[ei, 1] - coords[ej, 1]).abs()
    ov = torch.minimum(
        (patch_size - dr).clamp(min=0),
        (patch_size - dc).clamp(min=0),
    ).float()
    return ov


def init_conductance(
    edges: torch.Tensor,
    mode: str,
    dense_value: float = 1.0,
    overlap: torch.Tensor | None = None,
) -> torch.Tensor:
    """Initial per-edge conductance, shape (E,)."""
    if edges.numel() == 0:
        return torch.zeros(0)
    if mode == "dense":
        return torch.full((edges.shape[1],), float(dense_value))
    if mode == "overlap":
        if overlap is None:
            raise ValueError("overlap init requires overlap weights")
        return overlap.clamp(0.0, 1.0)
    raise ValueError(f"unknown D_init_mode {mode!r}")
