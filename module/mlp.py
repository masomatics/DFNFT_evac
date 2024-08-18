import numpy as np
import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
import pdb


class MLP(nn.Module):
    def __init__(
        self,
        in_dim=2,
        out_dim=3,
        depth=3,
        activation=nn.ELU,
        hidden_multiple=2,
        initmode="default",
        **kwargs,
    ):
        # super(MLP, self).__init__()
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = []
        self.depth = depth
        self.dospec = dospec
        for k in range(depth):
            if k == 0:
                dimin = in_dim
                dimout = int(in_dim * hidden_multiple)
            else:
                dimin = int(in_dim * hidden_multiple)
                dimout = int(in_dim * hidden_multiple)
            linlayer = nn.Linear(in_features=dimin, out_features=dimout)
            if initmode == "cond":
                initialize_linear(linlayer, thresh=2.0)
            elif initmode == "default":
                nn.init.orthogonal_(linlayer.weight.data)
                nn.init.uniform_(linlayer.bias.data)
            else:
                raise NotImplementedError

            self.layers.append(linlayer)
            self.layers.append(activation())
        linlayer = nn.Linear(in_features=dimout, out_features=self.out_dim)

        self.layers.append(linlayer)

        self.network = nn.Sequential(*self.layers)

    def __call__(self, x):
        x = x.flatten(start_dim=1)
        return self.network(x)


# Four layer MLP for encoder
class MLPEncBase(nn.Module):
    def __init__(
        self, dim_latent=128, dim_hidden=256, dim_data=128, act=nn.ReLU(), depth=3
    ):
        super().__init__()
        self.act = act
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        self.dim_data = dim_data
        self.lin1 = nn.Linear(self.dim_data, self.dim_hidden)
        self.lin2 = nn.Linear(self.dim_hidden, self.dim_latent)
        self.phi = nn.Sequential(
            # nn.LazyLinear(self.dim_hidden),
            self.lin1,
            self.act,
            # nn.LazyLinear(self.dim_hidden),
            self.lin2,
            self.act,
            # nn.LazyLinear(self.dim_latent)
            # nn.Linear(self.dim_hidden,self.dim_latent),
        )
        # self.linear = nn.LazyLinear(
        #    self.dim_latent) if self.dim_latent > 0 else lambda x: x
        self.linear = nn.Linear(self.dim_latent, self.dim_latent)

    def __call__(self, x):
        h = x
        h = self.phi(h)
        h = h.reshape(h.shape[0], -1)
        h = self.linear(h)
        return h


# Four layer MLP for encoder
class MLPDecBase(nn.Module):
    def __init__(
        self, dim_data=128, dim_latent=128, dim_hidden=256, act=nn.ReLU(), depth=3
    ):
        super().__init__()
        self.act = act
        self.dim_data = dim_data
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        # self.linear = nn.LazyLinear(self.dim_hidden)
        # self.linear = nn.Linear(self.dim_latent,self.dim_hidden)
        self.net = nn.Sequential(
            nn.Linear(self.dim_latent, self.dim_hidden),
            self.act,
            nn.Linear(self.dim_hidden, self.dim_hidden),
            self.act,
            # nn.LazyLinear(self.dim_data)
            nn.Linear(self.dim_hidden, self.dim_data),
        )

    def __call__(self, x):
        x = x.reshape([x.shape[0], -1])
        x = self.net(x)
        return x


class MLP_AE(nn.Module):
    def __init__(
        self,
        dim_a,
        dim_m,
        depth=3,
        transition_model="LS",
        dim_data=128,
        alignment=False,
        change_of_basis=False,
        predictive=True,
        gpu_id=0,
        activation="tanh",
        require_input_adapter=False,
        maskmat=None,
    ):
        super().__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.depth = depth
        self.predictive = predictive
        self.dim_data = dim_data
        self.no_embed = True
        self.maskmat = maskmat
        self.require_input_adapter = require_input_adapter
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
        else:
            raise NotImplementedError


class MLPEncoder(MLP_AE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enc = MLPEncBase(
            dim_data=self.dim_data,
            dim_latent=self.dim_a * self.dim_m,
            act=self.activation_fxn,
            depth=self.depth,
        )

    def _encode_base(self, xs, enc):
        H = enc(xs)
        return H

    def forward(self, signal):
        xs = signal
        H = self._encode_base(xs, self.enc)
        H = torch.reshape(H, (H.shape[0], self.dim_m, self.dim_a))
        # if hasattr(self, "change_of_basis"):
        #     H = H @ repeat(self.change_of_basis,
        #                    'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])
        return H


class MLPDecoder(MLP_AE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dec = MLPDecBase(
            dim_data=self.dim_data,
            dim_latent=self.dim_a * self.dim_m,
            act=self.activation_fxn,
            depth=self.depth,
        )

    def forward(self, H):
        # if hasattr(self, "change_of_basis"):
        #     H = H @ repeat(torch.linalg.inv(self.change_of_basis),
        #                    'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])
        if hasattr(self, "pidec"):
            # H = rearrange(H, 'n t d_s d_a -> (n t) d_a d_s')
            H = self.pidec(H)
        else:
            pass
        x_next_preds = self.dec(H)
        # x_next_preds = torch.reshape(
        #     x_next_preds, (n, t, *x_next_preds.shape[1:]))
        return x_next_preds


# import numpy as np
# import torch
# from torch import nn
# from einops.layers.torch import Rearrange
# from einops import repeat, rearrange, einsum
# import pdb
# import math


# class MaskFlatLinear(nn.Module):
#     def __init__(
#         self,
#         dim_m: int,
#         in_dim: int,
#         out_dim: int,
#         maskmat=None,
#         initializer_range=0.01,
#     ):
#         super().__init__()

#         self.dim_m = dim_m
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         assert self.out_dim % self.dim_m == 0
#         assert self.in_dim % self.dim_m == 0

#         self.dim_a_in = self.in_dim // self.dim_m
#         self.dim_a_out = self.out_dim // self.dim_m
#         # self.maskmat = maskmat
#         # self.maskmat.requires_grad = False
#         self.initializer_range = initializer_range
#         self.maskmat = maskmat
#         self.dim_a_mask = nn.Parameter(
#             torch.ones(self.dim_a_out, self.dim_a_in), requires_grad=False
#         )

#         # self.register_buffer('kronmask', kronmask)

#         self.W = nn.Parameter(torch.zeros(self.out_dim, self.in_dim))
#         self.b = nn.Parameter(torch.zeros(self.out_dim))

#         torch.nn.init.normal_(self.W, std=self.initializer_range)
#         torch.nn.init.normal_(self.b, std=self.initializer_range)

#     def forward(self, x):
#         weight = self.W
#         # if self.maskmat is not None:
#         #     kronmask = torch.kron(self.maskmat.to(self.W.device), self.dim_a_mask)
#         #     weight = self.W * kronmask
#         # print(self.maskmat[:2, :2])

#         x = einsum(weight, x, "... o i,  ... i -> ... o")
#         x = x + self.b
#         return x


# # Four layer MLP for encoder
# class MLPEncBase(nn.Module):
#     def __init__(
#         self,
#         dim_latent=128,
#         dim_hidden=256,
#         dim_data=128,
#         act=nn.ReLU(),
#         depth=3,
#         dim_m=0,
#         maskmat=None,
#     ):
#         super().__init__()
#         self.act = act
#         self.dim_hidden = dim_hidden
#         self.dim_latent = dim_latent
#         self.dim_data = dim_data
#         self.depth = depth

#         modseq = nn.ModuleList()
#         dimlist = [self.dim_hidden] * (1 + self.depth)
#         # This part differs from DecBase
#         if (self.dim_data % dim_m) != 0:
#             middledim = int(math.ceil(self.dim_data / dim_m) * dim_m)
#             modseq.append(nn.Linear(self.dim_data, middledim))
#             dimlist[0] = middledim
#         else:
#             dimlist[0] = self.dim_data
#         dimlist[-1] = self.dim_latent
#         for k in range(1, 1 + self.depth):
#             if maskmat is None or dim_m == 0:
#                 modseq.append(nn.Linear(dimlist[k - 1], dimlist[k]))
#             else:
#                 # modseq.append(nn.Linear(dimlist[k-1], dimlist[k]))
#                 modseq.append(
#                     MaskFlatLinear(
#                         in_dim=dimlist[k - 1],
#                         out_dim=dimlist[k],
#                         maskmat=maskmat,
#                         dim_m=dim_m,
#                     )
#                 )

#             if k < self.depth:
#                 modseq.append(self.act)

#         self.phi = nn.Sequential(*modseq)

#         # self.lin1 = nn.Linear(self.dim_data, self.dim_hidden)
#         # self.lin2 = nn.Linear(self.dim_hidden, self.dim_latent)
#         # self.linear = nn.Linear(self.dim_latent, self.dim_latent)
#         # self.phi = nn.Sequential(self.lin1, self.act, self.lin2, self.act, self.linear)

#     def __call__(self, x):
#         h = x
#         h = self.phi(h)
#         return h


# # Four layer MLP for encoder
# class MLPDecBase(nn.Module):
#     def __init__(
#         self,
#         dim_data=128,
#         dim_latent=128,
#         dim_hidden=256,
#         act=nn.ReLU(),
#         depth=3,
#         dim_m=0,
#         maskmat=None,
#     ):
#         super().__init__()
#         self.act = act
#         self.dim_data = dim_data
#         self.dim_hidden = dim_hidden
#         self.dim_latent = dim_latent
#         self.dim_m = dim_m
#         self.depth = depth

#         dimlist = [self.dim_hidden] * (1 + self.depth)
#         # This part differs from EncBase
#         # dimlist[-1] = self.dim_data
#         if (self.dim_data % dim_m) != 0:
#             middledim = int(math.ceil(self.dim_data / dim_m) * dim_m)
#             dimlist[-1] = middledim
#         else:
#             dimlist[-1] = self.dim_data
#             middledim = None
#         dimlist[0] = self.dim_latent
#         modseq = nn.ModuleList()
#         for k in range(1, 1 + self.depth):
#             if maskmat is None or dim_m == 0:
#                 modseq.append(nn.Linear(dimlist[k - 1], dimlist[k]))
#             else:
#                 modseq.append(
#                     MaskFlatLinear(
#                         in_dim=dimlist[k - 1],
#                         out_dim=dimlist[k],
#                         maskmat=maskmat,
#                         dim_m=dim_m,
#                     )
#                 )
#                 # modseq.append(nn.Linear(dimlist[k-1], dimlist[k]))

#             if k < self.depth:
#                 modseq.append(self.act)
#         if (self.dim_data % dim_m) != 0:
#             modseq.append(nn.Linear(middledim, self.dim_data))

#         self.net = nn.Sequential(*modseq)
#         # self.net = nn.Sequential(
#         #     nn.Linear(self.dim_latent, self.dim_hidden),
#         #     self.act,
#         #     nn.Linear(self.dim_hidden, self.dim_hidden),
#         #     self.act,
#         #     nn.Linear(self.dim_hidden, self.dim_data),
#         # )

#     def __call__(self, x):
#         x = x.reshape([x.shape[0], -1])
#         x = self.net(x)
#         return x


# class MLP_AE(nn.Module):
#     def __init__(
#         self,
#         dim_a,
#         dim_m,
#         depth=3,
#         transition_model="LS",
#         dim_data=128,
#         alignment=False,
#         change_of_basis=False,
#         predictive=True,
#         gpu_id=0,
#         activation="tanh",
#         require_input_adapter=False,
#         maskmat=None,
#     ):
#         super().__init__()
#         self.dim_a = dim_a
#         self.dim_m = dim_m
#         self.depth = depth
#         self.predictive = predictive
#         self.dim_data = dim_data
#         self.no_embed = True
#         # self.maskmat = nn.Parameter(maskmat, requires_grad=False)
#         self.maskmat = None
#         self.require_input_adapter = require_input_adapter
#         if torch.cuda.is_available():
#             self.device = torch.device("cuda", gpu_id)
#         else:
#             self.device = torch.device("cpu")
#         if activation == "relu":
#             self.activation_fxn = nn.ReLU()
#         elif activation == "tanh":
#             self.activation_fxn = nn.Tanh()
#         elif activation == "sigmoid":
#             self.activation_fxn = nn.Sigmoid()
#         else:
#             raise NotImplementedError


# class MLPEncoder(MLP_AE):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.enc = MLPEncBase(
#             dim_data=self.dim_data,
#             dim_latent=self.dim_a * self.dim_m,
#             act=self.activation_fxn,
#             depth=self.depth,
#             dim_m=self.dim_m,
#             maskmat=self.maskmat,
#         )

#     def _encode_base(self, xs, enc):
#         H = enc(xs)
#         return H

#     def forward(self, signal):
#         xs = signal
#         H = self._encode_base(xs, self.enc)
#         H = torch.reshape(H, (H.shape[0], self.dim_m, self.dim_a))
#         # if hasattr(self, "change_of_basis"):
#         #     H = H @ repeat(self.change_of_basis,
#         #                    'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])
#         return H


# class MLPDecoder(MLP_AE):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.dec = MLPDecBase(
#             dim_data=self.dim_data,
#             dim_latent=self.dim_a * self.dim_m,
#             act=self.activation_fxn,
#             depth=self.depth,
#             dim_m=self.dim_m,
#             maskmat=self.maskmat,
#         )

#     def forward(self, H):
#         # if hasattr(self, "change_of_basis"):
#         #     H = H @ repeat(torch.linalg.inv(self.change_of_basis),
#         #                    'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])
#         if hasattr(self, "pidec"):
#             # H = rearrange(H, 'n t d_s d_a -> (n t) d_a d_s')
#             H = self.pidec(H)
#         else:
#             pass
#         x_next_preds = self.dec(H)
#         # x_next_preds = torch.reshape(
#         #     x_next_preds, (n, t, *x_next_preds.shape[1:]))
#         return x_next_preds
