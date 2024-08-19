from torch import nn
import torch
from einops import rearrange, einsum
import math


class MaskFlatLinear(nn.Module):
    def __init__(
        self, dim_m: int, in_dim: int, out_dim: int, maskmat=None, initializer_range=0.1
    ):
        super().__init__()

        self.dim_m = dim_m
        self.in_dim = in_dim
        self.out_dim = out_dim
        assert self.out_dim % self.dim_m == 0
        assert self.in_dim % self.dim_m == 0

        self.dim_a_in = self.in_dim // self.dim_m
        self.dim_a_out = self.out_dim // self.dim_m
        self.maskmat = maskmat
        self.initializer_range = initializer_range
        self.dim_a_mask = nn.Parameter(
            torch.ones(self.dim_a_out, self.dim_a_in), requires_grad=False
        )

        # self.register_buffer('kronmask', kronmask)

        self.W = nn.Parameter(torch.zeros(self.out_dim, self.in_dim))
        self.b = nn.Parameter(torch.zeros(self.out_dim))

        torch.nn.init.normal_(self.W, std=self.initializer_range)
        torch.nn.init.normal_(self.b, std=self.initializer_range)

    def forward(self, x):
        if self.maskmat is not None:
            kronmask = torch.kron(self.maskmat, self.dim_a_mask)
            weight = self.W * kronmask
        else:
            weight = self.W
        x = einsum(weight, x, "... o i,  ... i -> ... o")
        x = x + self.b
        return x


class MLP_AE(nn.Module):
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
        self.maskmat = nn.Parameter(maskmat, requires_grad=False)
        self.hidden_dim = hidden_dim
        self.require_input_adapter = require_input_adapter
        self.dim_latent = self.dim_m * self.dim_a
        if torch.cuda.is_available():
            self.device = torch.device("cuda", gpu_id)
        else:
            self.device = torch.device("cpu")
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


class MLPEncoder(MLP_AE):
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
                    MaskFlatLinear(
                        in_dim=dimlist[k - 1],
                        out_dim=dimlist[k],
                        maskmat=self.maskmat,
                        dim_m=self.dim_m,
                    )
                )

            if k < self.depth:
                modseq.append(self.activation_fxn)
        self.net = nn.Sequential(*modseq)

    def forward(self, signal):
        xs = signal
        if not self.require_input_adapter:
            xs = rearrange(xs, "... d m -> ... (d m)")
        H = self.net(xs)
        H = torch.reshape(H, (H.shape[0], self.dim_m, self.dim_a))
        return H


class MLPDecoder(MLP_AE):
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
                    MaskFlatLinear(
                        in_dim=dimlist[k - 1],
                        out_dim=dimlist[k],
                        maskmat=self.maskmat,
                        dim_m=self.dim_m,
                    )
                )
                # modseq.append(nn.Linear(dimlist[k-1], dimlist[k]))

            if k < self.depth:
                modseq.append(self.activation_fxn)
        if (self.dim_data % self.dim_m) != 0:
            modseq.append(nn.Linear(middledim, self.dim_data))

        self.net = nn.Sequential(*modseq)

    def forward(self, xs):
        xs = xs.reshape([xs.shape[0], -1])
        H = self.net(xs)
        return H
