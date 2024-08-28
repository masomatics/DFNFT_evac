from torch import nn
import torch
from einops import rearrange, einsum
import math
import pdb
from torch import Tensor


class SimpleMaskModule(nn.Module):
    def __init__(self, dimRep: int, dimVec: int):
        super().__init__()
        self.dimRep = dimRep
        self.dimVec = dimVec
        self.do_normalize = True
        lambdainitial = torch.ones(self.dimRep, self.dimVec)
        lambdainitial = lambdainitial + torch.normal(
            torch.zeros_like(lambdainitial), 0.01
        )
        self.lambdas = nn.Parameter(lambdainitial)

    def compute_delta(self, lambdas):
        lambdas_expand_one = lambdas.unsqueeze(1)
        lambdas_expand_two = lambdas.unsqueeze(0)
        delta = torch.sum(-torch.abs(lambdas_expand_one - lambdas_expand_two), dim=-1)
        return delta

    def create_mask(self, lambda_prev=None):
        lambdas = self.forward_lambda(lambda_prev)
        mymask = torch.exp(self.compute_delta(lambdas))
        if lambda_prev is not None:
            mymask = lambda_prev * mymask
        return mymask

    def __call__(self, lambda_prev=None):
        return self.create_mask(lambda_prev)

    def get_laplacian(self, matrix: Tensor) -> Tensor:
        assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
        return torch.diag(torch.sum(matrix, dim=1)) - matrix

    def forward_lambda(self, in_lambda=None):
        return self.lambdas
