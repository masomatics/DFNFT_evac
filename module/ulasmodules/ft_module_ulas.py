from module.ulasmodules import encdec
import torch
from einops.layers.torch import Rearrange
from einops import repeat
from einops import rearrange, repeat

from torch import nn
import encdec as ed
import pdb


class Msqae(nn.Module):
    def __init__(
        self,
        dim_a,
        dim_m,
        alignment=False,
        ch_x=3,
        k=1.0,
        kernel_size=3,
        change_of_basis=False,
        predictive=True,
        bottom_width=4,
        n_blocks=3,
        detachM=0,
        require_input_adapter=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.k = k
        self.kernel_size = kernel_size
        self.n_blocks = n_blocks
        self.ch_x = ch_x
        self.bottom_width = bottom_width
        self.no_embed = True
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.maskmat = kwargs["maskmat"]
        self.require_input_adapter = require_input_adapter


class ULASEncoder(Msqae):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enc = ed.ResNetEncoder(
            self.dim_a * self.dim_m,
            k=self.k,
            kernel_size=self.kernel_size,
            n_blocks=self.n_blocks,
        )

    def _encode_base(self, xs, enc):
        # Originall expecting batch, time, ch, h, w
        shape = xs.shape
        # (batch time), ch. h , w  (This part is knocked out)
        # x = torch.reshape(xs, (shape[0] * shape[1], *shape[2:]))
        H = enc(xs)
        # batch x time x flatten_dimen
        # H = torch.reshape( H, (shape[0], shape[1], *H.shape[1:]))
        H = torch.reshape(H, (shape[0], self.dim_m, self.dim_a))
        return H

    def forward(self, signal, mask=None):
        xs = signal  # (n t) c h w
        H = self._encode_base(xs, self.enc)
        # batch x time x flatten_dimen
        # H = torch.reshape(
        #     H, (H.shape[0], H.shape[1], self.dim_m, self.dim_a))
        # if hasattr(self, "change_of_basis"):
        #     H = H @ repeat(self.change_of_basis,
        #                    'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])
        # pdb.set_trace()
        return H  # (n t) m a


class ULASDecoder(Msqae):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dec = ed.ResNetDecoder(
            self.ch_x,
            k=self.k,
            kernel_size=self.kernel_size,
            bottom_width=self.bottom_width,
            n_blocks=self.n_blocks,
        )

    def forward(self, signal, mask=None):
        H = signal
        # if hasattr(self, "change_of_basis"):
        #     H = H @ repeat(torch.linalg.inv(self.change_of_basis),
        #                    'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])
        H = rearrange(H, "nt m a -> nt (m a)")
        x_next_preds = self.dec(H)
        if self.require_input_adapter == False:
            x_next_preds = torch.squeeze(x_next_preds, dim=1)
        return x_next_preds


class ULASEncoderLayer(ULASEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_Dimside = False

    def _encode_base(self, xs, enc):
        if len(xs.shape) == 3:
            xs = xs[:, None]

        shape = xs.shape
        H = enc(xs)
        if self.is_Dimside:
            H = torch.reshape(H, (shape[0], -1, self.dim_a))
        else:
            H = torch.reshape(H, (shape[0], self.dim_m, -1))

        return H


class ULASDecoderLayer(ULASDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_Dimside = False
