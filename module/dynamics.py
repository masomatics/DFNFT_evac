import torch
from einops import rearrange, repeat, einsum
from misc import orthog_proj as op
import pdb


def _rep_M(M, T):
    return repeat(M, "n a1 a2 -> n t a1 a2", t=T)


def _mse(tens1, tens2):
    return torch.mean((tens1 - tens2) ** 2)


def _orthog_error(mat):
    return _mse(mat @ mat.T, torch.eye(mat.shape[0]).to(mat.device))


def _deltas(tens1, tens2, allbutaxis=0):
    delta = tens1, tens2
    axes_to_sum = [i for i in range(delta.dim()) if i != allbutaxis]
    deltas = torch.mean((delta) ** 2, axis=axes_to_sum)
    return deltas


def _solve(A, B, mask=None):
    # return B A^-1　 : solve M for  M A= B
    # A = A.to(dtype=torch.float64)  # b n d
    # B = B.to(dtype=torch.float64)  # b n d

    b, n, d = A.shape
    assert A.shape == B.shape

    AAT = A @ A.transpose(-2, -1)
    BAT = B @ A.transpose(-2, -1)
    if mask is not None:
        mask = mask.to(AAT.device)
        AAT = AAT * mask
        BAT = BAT * mask
    M = torch.linalg.solve(AAT, BAT, left=False)  # b  n  n

    return M


# Regression and Internal Prediction Module
class Dynamics(object):
    def __init__(self):
        self.M = None

    def __call__(self, H, intervene_fxn=None):
        # Applying to "token direction"
        # gH =  H @ _rep_M(self.M, T=H.shape[1])
        self.M = self.intervene(self.M, intervene_fxn)
        gH = einsum(H, _rep_M(self.M, T=H.shape[1]), "b t n d, b t m n -> b t m d")
        return gH

    def _compute_M(self, H, mask=None, orth_proj=False):
        self.mask = mask
        # H = H.to(dtype=torch.float64)
        H0, H1 = H[:, :-1], H[:, 1:]  # b 1 n d

        _H0 = H0.reshape(H0.shape[0], -1, H0.shape[-1])  # b n d
        _H1 = H1.reshape(H1.shape[0], -1, H1.shape[-1])  # b n d

        M_star = _solve(_H0, _H1, mask=mask)
        if orth_proj:
            M_star = op.orthogonal_projection_kernel(M_star)
        self.M = M_star * mask[None, :]

        # check  "_mse( M_star[0] @_H0[0] , _H1[0])" and  _mse(self(H0), H1) is small
        # print(f"""H0 svd : {torch.linalg.svd(H0)[1][0]} """ )
        # print(f"""Regression error: {_mse(self(H0), H1)}""" )

        return M_star

    def intervene(self, Mmat, intervene_fxn=None):
        if intervene_fxn is None:
            return Mmat
        else:
            return intervene_fxn(Mmat)


### Dim sideoperatrions
class DynamicsDimSide(Dynamics):
    def __init__(self):
        super().__init__()

    def __call__(self, H, intervene_fxn=None):
        # Applying to "token direction"
        # gH =  H @ _rep_M(self.M, T=H.shape[1])
        self.M = self.intervene(self.M, intervene_fxn)
        gH = einsum(H, _rep_M(self.M, T=H.shape[1]), "b t n d, b t d d2 -> b t n d2")
        return gH

    def _compute_M(self, H, mask=None, orth_proj=False):
        # H = H.to(dtype=torch.float64)
        self.mask = mask
        H0, H1 = H[:, :-1], H[:, 1:]  # b 1 n d
        # print(H0[0, 0, 0], "ForDebug LAT in Dynam")
        # print(H1[0, 0, 0], "ForDebug LAT in Dynam")
        _H0 = H0.reshape(H0.shape[0], -1, H0.shape[-1])  # b n d
        _H1 = H1.reshape(H1.shape[0], -1, H1.shape[-1])  # b n d
        M_star = _solveDim(_H0, _H1, mask=mask)
        # print(M_star[0, 0], "FOR Debu M ")

        if orth_proj:
            M_star = op.orthogonal_projection_kernel(M_star)
        self.M = M_star  # check  "_mse( M_star[0] @_H0[0] , _H1[0])" and  _mse(self(H0), H1) is small
        # print(f"""H0 svd : {torch.linalg.svd(H0)[1][0]} """ )
        # print(f"""Regression error: {_mse(self(H0), H1)}""" )

        return M_star


def _solveDim(A, B, mask=None):
    # return M= B^-1 A : solve A M = B
    # A = A.to(dtype=torch.float64)  # b n d
    # B = B.to(dtype=torch.float64)  # b n d

    b, n, d = A.shape
    assert A.shape == B.shape

    ATA = A.transpose(-2, -1) @ A  # b d n  , b n d -> b d d
    ATB = A.transpose(-2, -1) @ B  # b d n  , b n d -> b d d
    if mask is not None:
        ATA = ATA * mask
        ATB = ATB * mask

    #  AM =B ->  AT A M = AT B
    M = torch.linalg.solve(ATA, ATB)  # b  n  n

    return M
