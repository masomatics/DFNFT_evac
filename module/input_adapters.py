import torch
from einops import rearrange, repeat, einsum
from torch import nn


class vaniila_input_adapter(nn.Module):
    def __init__(self, **kwargs):
        super.__init__()

    def forward(self, x):
        return x

    def deforward(self, x):
        return x


class image2mlp_adapter(nn.Module):
    def __init__(self, height=32, width=32, ch=3, dim_data=128):
        super.__init__()

        self.height = height
        self.width = width
        self.ch = ch

        self.indim = height * width * ch
        self.simple_linear_adapt_in = self.linear(self.indim, dim_data)
        self.simple_linear_adapt_out = self.linear(self.indim, dim_data)

    def forward(self, x):
        x = rearrange(x, "n ... -> n (...)")
        x = self.simple_linear_adapt_in(x)
        return x

    def deforward(self, x):
        x = self.simple_linear_adapt_out(x)
        x = rearrange(x, "n, d ->  n h w c", h=self.height, w=self.width, c=self.ch)
        return x
