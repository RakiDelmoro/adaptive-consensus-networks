"""Image decomposition into overlapping patches (batched, torch).

``decompose(images, patch_size, stride)`` returns patches and the grid coordinates
of each patch's top-left corner, so :mod:`acn.topology` can build spatial-neighbor
edges from overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


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
