import sys

sys.path.append("./")
sys.path.append("./module")

from torch import nn
import torch
from einops import rearrange, einsum
import math
import pdb
from module import mlp_new as mn
from module import mlp_lambda_mask as mlm
from module import RotatingLayers as rl
from omegaconf import OmegaConf


class MM_RotNet_AE(nn.Module):
    def __init__(
        self,
        dim_a,
        dim_m,
        depth=3,
        transition_model="LS",
        dim_data=128,
        hidden_dim=256,
        alignment=False,
        change_of_basis=False,
        predictive=True,
        gpu_id=0,
        activation="tanh",
        require_input_adapter=False,
        maskmat=None,
        no_mask=False,
    ):
        super().__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.depth = depth
        self.predictive = predictive
        self.dim_data = dim_data
        self.no_embed = True
        self.no_mask = no_mask

        self.Rot_opt = OmegaConf.create({"rotation_dimensions": self.dim_m})

        self.hidden_dim = hidden_dim
        self.require_input_adapter = require_input_adapter
        self.dim_latent = self.dim_m * self.dim_a

        if activation == "relu":
            self.activation_fxn = nn.ReLU()
        elif activation == "tanh":
            self.activation_fxn = nn.Tanh()
        elif activation == "sigmoid":
            self.activation_fxn = nn.Sigmoid()
        elif activation == "id":
            self.activation_fxn = nn.Identity()
        else:
            raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device if self.parameters() else None


class MM_RotEncoder(MM_RotNet_AE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        modseq = nn.ModuleList()

        dimlist = [self.hidden_dim] * (1 + self.depth)
        # This part differs from DecBase
        if (self.dim_data % self.dim_m) != 0:
            middledim = int(math.ceil(self.dim_data / self.dim_m) * self.dim_m)
            modseq.append(nn.Linear(self.dim_data, middledim))
            dimlist[0] = middledim
        else:
            dimlist[0] = self.dim_data
        dimlist[-1] = self.dim_latent
        self.dimlist = [int(dimlist[k] / self.dim_m) for k in range(len(dimlist))]
        for k in range(1, 1 + self.depth):
            if self.no_mask == True or self.dim_m == 0:
                modseq.append(nn.Linear(self.dimlist[k - 1], self.dimlist[k]))
            else:
                if k == self.depth:
                    # if True:
                    modseq.append(
                        rl.RotatingMaskLinear(
                            opt=self.Rot_opt,
                            in_features=self.dimlist[k - 1],
                            out_features=self.dimlist[k],
                        )
                    )
                else:
                    modseq.append(
                        mlm.MM_MaskFlatLinear(
                            in_dim=dimlist[k - 1],
                            out_dim=dimlist[k],
                            dim_m=self.dim_m,
                        )
                    )

            if k < self.depth:
                modseq.append(self.activation_fxn)
        self.net = nn.Sequential(*modseq)

    def forward(self, signal, mask=None):
        H = signal  # b d

        for layer in self.net:
            if isinstance(layer, rl.RotatingMaskLinear):
                H = layer(H, mask)
            elif isinstance(layer, mlm.MM_MaskFlatLinear):
                H = torch.reshape(H, (H.shape[0], -1))
                H = layer(H)
                H = torch.reshape(H, (H.shape[0], self.dim_m, -1))
            else:
                H = layer(H)
                H = torch.reshape(H, (H.shape[0], self.dim_m, -1))
        aaa = torch.sum(torch.sum(H**2, dim=1), axis=0)
        print(aaa)

        H = torch.reshape(H, (H.shape[0], self.dim_m, self.dim_a))

        pdb.set_trace()
        return H


class MM_RotDecoder(MM_RotNet_AE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dimlist = [self.hidden_dim] * (1 + self.depth)
        # This part differs from EncBase
        # dimlist[-1] = self.dim_data
        if (self.dim_data % self.dim_m) != 0:
            middledim = int(math.ceil(self.dim_data / self.dim_m) * self.dim_m)
            dimlist[-1] = middledim
        else:
            dimlist[-1] = self.dim_data
            middledim = None
        dimlist[0] = self.dim_latent
        modseq = nn.ModuleList()
        self.dimlist = [int(dimlist[k] / self.dim_m) for k in range(len(dimlist))]
        for k in range(1, 1 + self.depth):
            if self.no_mask == True or self.dim_m == 0:
                modseq.append(nn.Linear(self.dimlist[k - 1], self.dimlist[k]))
            else:
                if k == self.depth:
                    # if True:
                    modseq.append(
                        rl.RotatingMaskLinear(
                            opt=self.Rot_opt,
                            in_features=self.dimlist[k - 1],
                            out_features=self.dimlist[k],
                        )
                    )
                else:
                    modseq.append(
                        mlm.MM_MaskFlatLinear(
                            in_dim=dimlist[k - 1],
                            out_dim=dimlist[k],
                            dim_m=self.dim_m,
                        )
                    )

            if k < self.depth:
                modseq.append(self.activation_fxn)
        if (self.dim_data % self.dim_m) != 0:
            modseq.append(nn.Linear(middledim, self.dim_data))

        self.net = nn.Sequential(*modseq)

    def forward(self, xs, mask=None):
        H = xs
        for k in range(len(self.net)):
            layer = self.net[k]
            if k == len(self.net) - 1 and self.require_input_adapter is True:
                H = H.reshape(H.shape[0], -1)

            if isinstance(layer, rl.RotatingMaskLinear):
                H = layer(H, mask)
            elif isinstance(layer, mlm.MM_MaskFlatLinear):
                H = torch.reshape(H, (H.shape[0], -1))
                H = layer(H)
                H = torch.reshape(H, (H.shape[0], self.dim_m, -1))
            else:
                H = layer(H)

        return H
