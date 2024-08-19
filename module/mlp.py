import numpy as np
import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
import pdb


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
        no_mask=True,
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
