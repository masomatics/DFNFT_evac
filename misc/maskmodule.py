from torch import nn
import torch
from einops import rearrange, einsum
import math
import pdb
from torch import Tensor
import torch.nn.functional as F
import sys
from omegaconf import OmegaConf

sys.path.append("../")
sys.path.append("./")

sys.path.append("./module")

from module import RotatingLayers as rl
import omegaconf
from module import mlp_lambda_mask as mlm


"""
Mask determined at each layer independently.
"""


class SimpleMaskModule(nn.Module):
    def __init__(self, dimRep: int, dimVec: int, **kwargs):
        super().__init__()
        self.dimRep = dimRep
        self.dimVec = dimVec
        self.do_normalize = True
        lambdainitial = torch.ones(self.dimRep, self.dimVec)
        lambdainitial = lambdainitial + torch.normal(
            torch.zeros_like(lambdainitial), 0.0001
        )
        # if dimVec > 1:
        #     lambdainitial = F.normalize(lambdainitial, p=2, dim=1)
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

    def __call__(self, lambda_prev=None, prev_mask=None, **kwargs):
        self.own_mask = prev_mask  # If this is a Module of the first layer, this will be set to zero, to be used by "encoder/decoder. "
        dynamics_mask, lambdas_next = self.forward_mask(
            lambda_prev=None, prev_mask=None, **kwargs
        )
        self.dynamics_mask = dynamics_mask  # THIS LINE IS REQUIRED AT ALL TIME.
        return dynamics_mask, lambdas_next

    def forward_mask(self, lambda_prev=None, prev_mask=None, **kwargs):
        lambdas_next = self.forward_lambda(lambda_prev)
        dynamics_mask = self.create_mask(lambdas_next)
        return dynamics_mask, lambdas_next

    def get_laplacian(self, matrix: Tensor) -> Tensor:
        assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
        return torch.diag(torch.sum(matrix, dim=1)) - matrix

    # Default ignores the in_lambda, and uses its "own" lambda.
    def forward_lambda(self, in_lambda=None):
        return self.lambdas


"""
Imposing M(k) = M(k)hat * M(k-1)
"""


class SimpleStackModule(SimpleMaskModule):
    # REMEMBER the input Lambda used as an input and create the mask with it.
    def forward_lambda(self, in_lambda=None):
        return self.lambdas

    def forward_mask(self, lambda_prev=None, prev_mask=None, **kwargs):
        lambdas_next = self.forward_lambda(lambda_prev)
        dynamics_mask = self.create_mask(lambdas_next)
        if prev_mask is not None:
            dynamics_mask = dynamics_mask * prev_mask
        return dynamics_mask, lambdas_next


"""
The Lambda vectors are Normalized.
"""


class SimpleRotModule(SimpleStackModule):
    # REMEMBER the input Lambda used as an input and create the mask with it.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.dimVec > 2

    def forward_lambda(self, in_lambda=None):
        out_lambdas = F.normalize(self.lambdas, p=2, dim=1)
        return out_lambdas

    # def __call__(self, lambda_prev=None, prev_mask=None, **kwargs):
    #     self.own_mask = prev_mask  # If this is a Module of the first layer, this will be set to None, to be used by "encoder/decoder. "
    #     lambdas_next = self.forward_lambda(lambda_prev)
    #     dynamics_mask = self.create_mask(lambdas_next)
    #     if prev_mask is not None:
    #         dynamics_mask = dynamics_mask * prev_mask
    #     self.dynamics_mask = dynamics_mask  # THIS LINE IS REQUIRED AT ALL TIME

    #     return dynamics_mask, lambdas_next  # This will be used for dynamics.


"""
USING ROTATING FEATURES FOR THE PURPOSE OF BLOCK DIAGONALIZING
"""


class RotFeatureModule(SimpleMaskModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.dimVec > 2

        lambdainitial = torch.ones(self.dimRep, self.dimVec)
        lambdainitial = lambdainitial + torch.normal(
            torch.zeros_like(lambdainitial), 1.0
        )
        lambdainitial = F.normalize(lambdainitial, p=2, dim=1)
        self.lambdas = nn.Parameter(lambdainitial)

        opt = kwargs
        opt["rotation_dimensions"] = self.dimVec

        self.opt = OmegaConf.create(opt)
        modseq = nn.ModuleList()
        for k in range(self.opt.depth):
            modseq.append(rl.RotatingLinear(self.opt, self.dimRep, self.dimRep))

        self.rlnet = nn.Sequential(*modseq)

    def forward_lambda(self, in_lambda=None):
        if in_lambda is None:
            in_lambda = self.lambdas

        out_lambdas = self.rlnet(in_lambda[None])[0]
        # out_lambdas = F.normalize(out_lambdas, p=2, dim=1)
        return out_lambdas


"""
Decimation type Rotating Feature Prototype
"""


class RotFeatureMaskModule(SimpleMaskModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.dimVec > 2

        lambdainitial = torch.ones(self.dimRep, self.dimVec)
        lambdainitial = lambdainitial + torch.normal(
            torch.zeros_like(lambdainitial), 1.0
        )
        lambdainitial = F.normalize(lambdainitial, p=2, dim=1)
        # lambdainitial = torch.ones(self.dimRep, self.dimVec)
        # lambdainitial = torch.eye(self.dimRep, self.dimRep)
        # lambdainitial = lambdainitial[:, : self.dimVec]

        # lambdainitial = lambdainitial + torch.normal(
        #     torch.zeros_like(lambdainitial), 1.0
        # )
        # lambdainitial = F.normalize(lambdainitial, p=2, dim=1)
        self.lambdas = nn.Parameter(lambdainitial)

        opt = kwargs
        opt["rotation_dimensions"] = self.dimVec

        self.opt = OmegaConf.create(opt)
        modseq = nn.ModuleList()
        for k in range(self.opt.depth):
            modseq.append(rl.RotatingMaskLinear(self.opt, self.dimRep, self.dimRep))
        self.rlnet = nn.Sequential(*modseq)

    def forward_lambda(
        self,
        in_lambda=None,
        prev_mask=None,
    ):
        # if in_lambda is None:
        #     in_lambda = self.lambdas
        in_lambda = self.lambdas

        out_lambdas = in_lambda
        for layer in self.rlnet:
            out_lambdas = layer(out_lambdas[None], mask=prev_mask)[0]
        # out_lambdas = F.normalize(out_lambdas, p=2, dim=1)
        return out_lambdas

    def forward_mask(self, lambda_prev=None, prev_mask=None, **kwargs):
        lambdas_next = self.forward_lambda(lambda_prev, prev_mask=prev_mask)
        # print("DEBUG LM", lambdas_next)
        # pdb.set_trace()
        if prev_mask is None:
            dynamics_mask = self.create_mask(lambdas_next)
        else:
            dynamics_mask = self.create_mask(lambdas_next) * prev_mask
        return dynamics_mask, lambdas_next


class RotFeatureMindLessModule(SimpleMaskModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.dimVec > 2

        lambdainitial = torch.ones(self.dimRep, self.dimVec)
        lambdainitial = lambdainitial + torch.normal(
            torch.zeros_like(lambdainitial), 1.0
        )
        lambdainitial = F.normalize(lambdainitial, p=2, dim=1)
        self.lambdas = nn.Parameter(lambdainitial)

        opt = kwargs
        opt["rotation_dimensions"] = self.dimVec

        self.opt = OmegaConf.create(opt)
        modseq = nn.ModuleList()
        for k in range(self.opt.depth):
            modseq.append(rl.RotatingLinear(self.opt, self.dimVec, self.dimVec))

        self.rlnet = nn.Sequential(*modseq)

    def forward_lambda(self, in_lambda=None):
        if in_lambda is None:
            in_lambda = self.lambdas
        out_lambdas = self.rlnet(in_lambda[None])[0]
        return out_lambdas

    def forward_mask(self, lambda_prev=None, prev_mask=None, **kwargs):
        lambdas_next = self.forward_lambda(lambda_prev)
        dynamics_mask = self.create_mask(lambdas_next)
        if prev_mask is not None:
            dynamics_mask = dynamics_mask * prev_mask
        return dynamics_mask, lambdas_next
