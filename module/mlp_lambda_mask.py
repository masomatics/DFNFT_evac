import sys

sys.path.append("./")
sys.path.append("./module")

from torch import nn
import torch
from einops import rearrange, einsum
import math
import pdb
from utils import maskmodule as mm
from module import mlp_new as mn


class MM_MLP_AE(nn.Module):
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


class MM_MLPEncoder(MM_MLP_AE):
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
        for k in range(1, 1 + self.depth):
            if self.no_mask == True or self.dim_m == 0:
                modseq.append(nn.Linear(dimlist[k - 1], dimlist[k]))
            else:
                # modseq.append(nn.Linear(dimlist[k-1], dimlist[k]))
                modseq.append(
                    mn.MaskFlatLinear(
                        in_dim=dimlist[k - 1],
                        out_dim=dimlist[k],
                        maskmat=None,
                        dim_m=self.dim_m,
                    )
                )

            if k < self.depth:
                modseq.append(self.activation_fxn)
        self.net = nn.Sequential(*modseq)

    def forward(self, signal, mask):
        H = signal
        if not self.require_input_adapter:
            H = rearrange(signal, "... d m -> ... (d m)")

        for layer in self.net():
            if isinstance(layer, mn.MaskFlatLinear):
                H = layer(H, mask)
            else:
                H = layer(H)
        H = torch.reshape(H, (H.shape[0], self.dim_m, self.dim_a))
        return H


class MM_MLPDecoder(MM_MLP_AE):
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
        for k in range(1, 1 + self.depth):
            if self.no_mask == True or self.dim_m == 0:
                modseq.append(nn.Linear(dimlist[k - 1], dimlist[k]))
            else:
                modseq.append(
                    mn.MaskFlatLinear(
                        in_dim=dimlist[k - 1],
                        out_dim=dimlist[k],
                        maskmat=None,
                        dim_m=self.dim_m,
                    )
                )
                # modseq.append(nn.Linear(dimlist[k-1], dimlist[k]))

            if k < self.depth:
                modseq.append(self.activation_fxn)
        if (self.dim_data % self.dim_m) != 0:
            modseq.append(nn.Linear(middledim, self.dim_data))

        self.net = nn.Sequential(*modseq)

    def forward(self, xs, mask):
        H = xs.reshape([xs.shape[0], -1])
        for layer in self.net():
            if isinstance(layer, mn.MaskFlatLinear):
                H = layer(H, mask)
            else:
                H = layer(H)
        return H
