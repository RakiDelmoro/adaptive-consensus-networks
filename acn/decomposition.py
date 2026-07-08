"""Image decomposition into overlapping patches (batched, torch).

Two modes:
  * :func:`decompose` — the original single-scale mode: one patch_size on a
    regular grid. Returns patches and the grid coordinates of each patch's
    top-left corner, so :mod:`acn.topology` can build spatial-neighbor edges
    from overlap.
  * :func:`build_column_roster` + :func:`decompose_multi_scale` — Thousand-
    Brains multi-scale mode: columns with varying receptive fields (4x4, 6x6,
    8x8) tiled across the image. Each column has a permanent (size, row, col)
    identity (the roster); every forward pass extracts each column's patch and
    adaptive-pools it to a fixed internal size so one shared encoder works.
    See PLAN_2026-07-06.md (Decisions 1, 2, 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ColumnSpec:
    """Permanent identity of one column in multi-scale mode.

    size: patch side length (e.g. 4, 6, 8)
    row, col: top-left pixel position of this column's patch on the image
    """
    size: int
    row: int
    col: int


@dataclass
class MultiScaleDecomposition:
    """Result of decomposing a batch of images into multi-scale patches.

    patches: (B, N, pool_to**2 * C) flattened pooled patch contents (all the
             same size after adaptive pooling, so one encoder handles them)
    roster:  list of N ColumnSpec — each column's permanent (size, row, col)
    coords:  (N, 2) long tensor of (row, col) — convenience for topology
    sizes:   (N,) long tensor of patch sizes — convenience for topology
    """
    patches: torch.Tensor       # (B, N, pool_to*pool_to*C)
    roster: list[ColumnSpec]
    coords: torch.Tensor        # (N, 2) long
    sizes: torch.Tensor         # (N,) long

    @property
    def num_patches(self) -> int:
        return self.patches.shape[1]

    @property
    def patch_dim(self) -> int:
        return self.patches.shape[2]


@dataclass
class Decomposition:
    """Result of decomposing a batch of images into patches.

    patches: (B, N, C*P*P) flattened patch contents
    coords:  (N, 2) integer (row, col) top-left of each patch (shared across batch)
    grid:    (Hg, Wg) grid shape of patch centers
    """

    patches: torch.Tensor   # (B, N, patch_dim)
    coords: torch.Tensor    # (N, 2) long
    grid: tuple[int, int]

    @property
    def num_patches(self) -> int:
        return self.patches.shape[1]

    @property
    def patch_dim(self) -> int:
        return self.patches.shape[2]


def num_patches_along(size: int, patch: int, stride: int) -> int:
    """Number of patches fitting along one axis with a leading patch at offset 0.

    Patches start at 0, stride, 2*stride, ... and must fit entirely within `size`.
    """
    if patch > size:
        raise ValueError(f"patch {patch} > size {size}")
    n = (size - patch) // stride + 1
    if n < 1:
        n = 1
    return n


def decompose(
    images: torch.Tensor,
    patch_size: int,
    stride: int,
) -> Decomposition:
    """Extract overlapping patches from a batch of images.

    images: (B, C, H, W)
    -> patches (B, N, C*P*P), ordered row-major over the patch grid.
    """
    if images.dim() != 4:
        raise ValueError(f"expected (B,C,H,W); got {tuple(images.shape)}")
    B, C, H, W = images.shape
    Hg = num_patches_along(H, patch_size, stride)
    Wg = num_patches_along(W, patch_size, stride)

    # unfold -> (B, C, Hg, Wg, P, P)
    unfolded = F.unfold(
        images, kernel_size=patch_size, stride=stride
    )  # (B, C*P*P, L) where L = Hg*Wg
    patches = unfolded.transpose(1, 2).contiguous()  # (B, L, C*P*P)

    coords = torch.stack(
        torch.meshgrid(
            torch.arange(Hg) * stride,
            torch.arange(Wg) * stride,
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2).long()  # (N, 2) (row, col)

    return Decomposition(patches=patches, coords=coords, grid=(Hg, Wg))


def reconstruct_coverage(coords: torch.Tensor, patch_size: int, image_size: int) -> torch.Tensor:
    """Return a (image_size, image_size) mask of cells covered by >=1 patch."""
    mask = torch.zeros(image_size, image_size, dtype=torch.int32)
    for (r, c) in coords.tolist():
        mask[r:r + patch_size, c:c + patch_size] += 1
    return mask


# ===================================================================== #
# Multi-scale decomposition (Thousand-Brains varying receptive fields)
# ===================================================================== #


def build_column_roster(
    image_size: int,
    multi_scale_specs: tuple[tuple[int, int], ...],
) -> list[ColumnSpec]:
    """Build the fixed roster of columns for multi-scale mode.

    Each spec is (patch_size, stride). Patches of that size are tiled across
    the image at that stride (leading patch at offset 0). The roster is the
    concatenation of all sizes' grids — each column has a permanent (size,
    row, col) identity, the same for every input (Decision 1: fixed
    assignment).

    Example: image_size=28, specs=((4,4),(6,6),(8,8))
      -> 4x4 grid: 7x7=49 columns at stride 4
      -> 6x6 grid: 4x4=16 columns at stride 6  (positions 0,6,12,18)
      -> 8x8 grid: 3x3=9 columns at stride 8   (positions 0,8,16)
      -> 74 columns total
    """
    roster: list[ColumnSpec] = []
    for patch_size, stride in multi_scale_specs:
        n_along = num_patches_along(image_size, patch_size, stride)
        for r in range(n_along):
            for c in range(n_along):
                row = r * stride
                col = c * stride
                if row + patch_size > image_size or col + patch_size > image_size:
                    continue  # safety: skip patches that would overflow
                roster.append(ColumnSpec(size=patch_size, row=row, col=col))
    return roster


def decompose_multi_scale(
    images: torch.Tensor,
    roster: list[ColumnSpec],
    pool_to: int,
) -> MultiScaleDecomposition:
    """Extract each column's patch and adaptive-pool to a fixed size.

    images: (B, C, H, W)
    -> patches (B, N, pool_to*pool_to*C), all the same size after pooling so
       one shared encoder handles them regardless of original patch size.

    For each column in the roster we slice its (size x size) patch out of the
    image, then F.adaptive_avg_pool2d it to pool_to x pool_to. This realizes
    Decision 3: one shared encoder, patches of different receptive fields
    unified by pooling. The "what size did this column see" info is preserved
    in the column's roster identity, not lost.
    """
    if images.dim() != 4:
        raise ValueError(f"expected (B,C,H,W); got {tuple(images.shape)}")
    B, C, H, W = images.shape
    N = len(roster)
    device, dtype = images.device, images.dtype
    out_dim = pool_to * pool_to * C
    patches = torch.empty(B, N, out_dim, device=device, dtype=dtype)
    coords = torch.zeros(N, 2, dtype=torch.long)
    sizes = torch.zeros(N, dtype=torch.long)
    for i, spec in enumerate(roster):
        p = spec.size
        r0, c0 = spec.row, spec.col
        # slice (B, C, p, p) — all columns see the same image, so this is
        # the same slice across the batch
        patch = images[:, :, r0:r0 + p, c0:c0 + p]            # (B, C, p, p)
        # adaptive-pool to (pool_to, pool_to) regardless of p
        pooled = F.adaptive_avg_pool2d(patch, (pool_to, pool_to))  # (B, C, pool_to, pool_to)
        patches[:, i, :] = pooled.reshape(B, out_dim)
        coords[i, 0] = r0
        coords[i, 1] = c0
        sizes[i] = p
    return MultiScaleDecomposition(patches=patches, roster=roster, coords=coords, sizes=sizes)


def reconstruct_coverage_multi_scale(roster: list[ColumnSpec], image_size: int) -> torch.Tensor:
    """Return a (image_size, image_size) mask of cells covered by >=1 patch
    in multi-scale mode (patches of different sizes)."""
    mask = torch.zeros(image_size, image_size, dtype=torch.int32)
    for spec in roster:
        mask[spec.row:spec.row + spec.size, spec.col:spec.col + spec.size] += 1
    return mask
