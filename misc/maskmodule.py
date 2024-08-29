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
        self.own_mask = None  # OWN MASK IS A PREV MASK, to be used by Decoder
        self.dynamics_mask = None

    def compute_delta(self, lambdas):
        lambdas_expand_one = lambdas.unsqueeze(1)
        lambdas_expand_two = lambdas.unsqueeze(0)
        delta = torch.sum(-torch.abs(lambdas_expand_one - lambdas_expand_two), dim=-1)
        return delta

    # Just creating the mask and forward ing the lambda.
    def create_mask(self, lambdas=None, **kwargs):
        if lambdas is None:
            lambdas = self.lambdas
        lambdamask = torch.exp(self.compute_delta(lambdas))
        return lambdamask  # This will be used for the next / dynamic mask.

    # REMEMBER the input Lambda used as an input and create the mask with it.
    def __call__(self, lambda_prev=None, prev_mask=None, **kwargs):
        self.own_mask = prev_mask  # If this is a Module of the first layer, this will be set to zero, to be used by "encoder/decoder. "
        lambdas_next = self.forward_lambda(lambda_prev)
        dynamics_mask = self.create_mask(lambdas_next)
        self.dynamics_mask = dynamics_mask
        return dynamics_mask, lambdas_next  # This will be used for dynamics.

    def get_laplacian(self, matrix: Tensor) -> Tensor:
        assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
        return torch.diag(torch.sum(matrix, dim=1)) - matrix

    # Default ignores the in_lambda, and uses its "own" lambda.
    def forward_lambda(self, in_lambda=None):
        return self.lambdas


class SimpleStackModule(SimpleMaskModule):
    # REMEMBER the input Lambda used as an input and create the mask with it.
    def forward_lambda(self, in_lambda=None):
        return self.lambdas

    def __call__(self, lambda_prev=None, prev_mask=None, **kwargs):
        self.own_mask = prev_mask  # If this is a Module of the first layer, this will be set to zero, to be used by "encoder/decoder. "
        lambdas_next = self.forward_lambda(lambda_prev)
        dynamics_mask = self.create_mask(lambdas_next) * prev_mask
        return dynamics_mask, lambdas_next  # This will be used for dynamics.
