import torch
from einops import rearrange, repeat, einsum
from torch import nn


class vanilla_input_adapter(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

    def deforward(self, x):
        return x


class image2mlp_adapter(nn.Module):
    def __init__(self, height=32, width=32, ch=3, dim_data=128, **kwargs):
        super().__init__()

        self.height = height
        self.width = width
        self.ch = ch

        self.indim = height * width * ch
        self.simple_linear_adapt_in = nn.Linear(self.indim, dim_data)
        self.simple_linear_adapt_out = nn.Linear(dim_data, self.indim)

    def forward(self, x):
        x = rearrange(x, "n ... -> n (...)")
        x = self.simple_linear_adapt_in(x)
        return x

    def deforward(self, x):
        x = self.simple_linear_adapt_out(x)
        x = rearrange(
            x, "n (c h w) ->  n c h w", h=self.height, w=self.width, c=self.ch
        )
        return x
