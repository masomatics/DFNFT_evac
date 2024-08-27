import numpy as np
import torch
from torch import nn
from module.ulasmodules.resblock import Block, Conv1d1x1Block, Invertible_Resblock_Fc
import module.ulasmodules.resblock as rb
from einops.layers.torch import Rearrange
from einops import repeat
from torch.nn import functional as F
import copy
import pdb




class ResNetEncoder(nn.Module):
    def __init__(self,
                 dim_latent=1024,
                 k=1,
                 act=nn.ReLU(),
                 kernel_size=3,
                 n_blocks=3, 
                 img_size=32):
        super().__init__()
        self.k = k
        self.n_blocks= n_blocks
        self.img_size = img_size
        self.phi = nn.Sequential(
            nn.LazyConv2d(int(self.img_size * k), 3, 1, 1),
            *[Block(int(self.img_size * k) * (2 ** i), int(self.img_size * k) * (2 ** (i+1)), int(self.img_size * k) * (2 ** (i+1)),
                    resample='down', activation=act, kernel_size=kernel_size) for i in range(n_blocks)],
            nn.GroupNorm(min(self.img_size, int(self.img_size * k) * (2 ** n_blocks)),
                         int(self.img_size * k) * (2 ** n_blocks)),
            act)
        self.linear = nn.LazyLinear(
            dim_latent) if dim_latent > 0 else lambda x: x
        self.no_embed = True

    def __call__(self, x):
        h = x
        h = self.phi(h)
        h = h.reshape(h.shape[0], -1)
        h = self.linear(h)
        return h


class ResNetDecoder(nn.Module):
    def __init__(self, ch_x, k=1, act=nn.ReLU(), kernel_size=3, bottom_width=4, n_blocks=3,
    img_size=32):
        super().__init__()
        self.bottom_width = bottom_width
        self.n_blocks = n_blocks
        self.k = k
        self.img_size = img_size
        self.ch_x = ch_x
        self.kernel_size = kernel_size
        self.linear = nn.LazyLinear(int(self.img_size * self.k) * (2 ** n_blocks))
        self.net = nn.Sequential(
            *[Block(int(self.img_size * k) * (2 ** (i+1)), int(self.img_size * k) * (2 ** i), int(self.img_size * k) * (2 ** i),
                    resample='up', activation=act, kernel_size=self.kernel_size, posemb=True) for i in range(n_blocks-1, -1, -1)],
            nn.GroupNorm(min(self.img_size, int(self.img_size * k)), int(self.img_size * k)),
            act,
            nn.Conv2d(int(self.img_size * k), self.ch_x, 3, 1, 1)
        )

    def __call__(self, x):
        # print(1, x.shape)
        x = self.linear(x)
        # print(2, x.shape)
        x = repeat(x, 'n c -> n c h w', h=self.bottom_width, w=self.bottom_width)
        # print(3, x.shape)
        x = self.net(x)
        # print(4,x.shape)
        # pdb.set_trace()
        return x