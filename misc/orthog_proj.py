# Torch translation of
# https://github.com/vsitzmann/neural-isometries/blob/main/nn/fmaps.py

import torch


def ortho_det(U):
    return torch.linalg.det(U)


# Stable and differentiable procrustes projection
def orthogonal_projection_kernel(X, special=True):
    U, _, VH = torch.linalg.svd(X)

    # if (X.shape[-2] == X.shape[-1] and special):
    #       VH[..., -1, :] = ortho_det(torch.einsum("...ij, ...jk -> ...ik", U, VH))[..., None] * VH[..., -1, :]

    # R = torch.einsum( "...ij, ...jk -> ...ik", U, VH )
    if X.shape[-2] == X.shape[-1] and special:
        # Calculate the modified last row
        modified_last_row = (
            ortho_det(torch.einsum("...ij, ...jk -> ...ik", U, VH))[..., None]
            * VH[..., -1, :]
        )

        # Update VH out-of-place by creating a new tensor
        VH_updated = VH.clone()
        VH_updated[..., -1, :] = modified_last_row

    else:
        VH_updated = VH

    R = torch.einsum("...ij, ...jk -> ...ik", U, VH_updated)

    return R


def delta_mask(evals):
    mask = torch.exp(-1.0 * torch.abs(evals[..., None] - evals[..., None, :]))

    return mask

    # Solves for isometry best taking projection of A to projection of B in the eigenbasis
    # Equation (8) in the paper


def _iso_solve(A, B, Phi, Lambda):
    LMask = delta_mask(Lambda)
    # M removed, for now.
    PhiTMB = torch.einsum("...ji, ...jk->...ik", Phi[None, ...], B)
    PhiTMA = torch.einsum("...ji, ...jk->...ik", Phi[None, ...], A)

    tauOmega = orthogonal_projection_kernel(
        LMask[None, ...] * jnp.einsum("...ij,...kj->...ik", PhiTMB, PhiTMA)
    )

    return tauOmega
