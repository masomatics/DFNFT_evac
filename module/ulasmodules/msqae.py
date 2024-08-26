import numpy as np
import torch
import torch.nn as nn
from models import dynamics_models
import torch.nn.utils.parametrize as P
from models.dynamics_models import LinearTensorDynamicsLSTSQ, MultiLinearTensorDynamicsLSTSQ, HigherOrderLinearTensorDynamicsLSTSQ, LinearTensorDynamicsLSTSQ_inner
from models.base_networks import ResNetEncoder, ResNetDecoder, Conv1d1x1Encoder, MLP_iResNet, LinearNet, MLP
from models import base_networks as bn
from einops import rearrange, repeat
from utils.clr import simclr
import pdb
from sklearn.metrics import r2_score
import copy
from utils import misc
import einops
import torch.autograd as autograd
from models import misc_mnet as mnet
from collections import OrderedDict


class SeqAELSTSQ(nn.Module):
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
            *args,
            **kwargs):
        super().__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.predictive = predictive
        self.enc = ResNetEncoder(
            dim_a*dim_m, k=k, kernel_size=kernel_size, n_blocks=n_blocks)
        self.dec = ResNetDecoder(
            ch_x, k=k, kernel_size=kernel_size, bottom_width=bottom_width, n_blocks=n_blocks)
        self.dynamics_model = LinearTensorDynamicsLSTSQ(alignment=alignment)
        self.detachM = detachM
        if change_of_basis:
            self.change_of_basis = nn.Parameter(
                torch.empty(dim_a, dim_a))
            nn.init.eye_(self.change_of_basis)

    def _encode_base(self, xs, enc):
        #batch, time, ch, h, w
        shape = xs.shape
        x = torch.reshape(xs, (shape[0] * shape[1], *shape[2:]))
        H = enc(x)
        #batch x time x flatten_dimen
        H = torch.reshape(
            H, (shape[0], shape[1], *H.shape[1:]))
        return H

    def encode(self, xs):
        H = self._encode_base(xs, self.enc)
        # batch x time x flatten_dimen
        H = torch.reshape(
            H, (H.shape[0], H.shape[1], self.dim_m, self.dim_a))
        if hasattr(self, "change_of_basis"):
            H = H @ repeat(self.change_of_basis,
                           'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])

        return H

    def phi(self, xs):
        return self._encode_base(xs, self.enc.phi)

    #WARNING: get_M is not directly used in the decoding process.
    #M is always computed in self.dynamics_model
    def get_M(self, xs):
        dyn_fn = self.dynamics_fn(xs)
        Mstar = dyn_fn.M
        if self.detachM == 1:
            Mstar = Mstar.detach()
        return Mstar

    def decode(self, H):
        if hasattr(self, "change_of_basis"):
            H = H @ repeat(torch.linalg.inv(self.change_of_basis),
                           'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])
        n, t = H.shape[:2]
        if hasattr(self, "pidec"):
            H = rearrange(H, 'n t d_s d_a -> (n t) d_a d_s')
            H = self.pidec(H)
        else:
            H = rearrange(H, 'n t d_s d_a -> (n t) (d_s d_a)')
        x_next_preds = self.dec(H)
        x_next_preds = torch.reshape(
            x_next_preds, (n, t, *x_next_preds.shape[1:]))
        return x_next_preds

    def dynamics_fn(self, xs, return_loss=False, fix_indices=None):
        H = self.encode(xs)
        return self.dynamics_model(H, return_loss=return_loss, fix_indices=fix_indices)

    def loss(self, xs, return_reg_loss=True, T_cond=2, reconst=False, regconfig={}):
        xs_cond = xs[:, :T_cond]
        xs_pred = self(xs_cond, return_reg_loss=return_reg_loss,
                       n_rolls=xs.shape[1] - T_cond, reconst=reconst, regconfig=regconfig)
        if return_reg_loss:
            xs_pred, reg_losses = xs_pred
            #losses = (loss_bd(dyn_fn.M, self.alignment),
            #          loss_orth(dyn_fn.M), loss_internal_T)
        if reconst:
            xs_target = xs
        else:
            xs_target = xs[:, T_cond:] if self.predictive else xs[:, 1:]

        loss = torch.mean(
            torch.sum((xs_target - torch.sigmoid(xs_pred)) ** 2, axis=[2, 3, 4]))
        return (loss, reg_losses) if return_reg_loss else loss

    def __call__(self, xs_cond, return_reg_loss=False, n_rolls=1, fix_indices=None, reconst=False,
                 regconfig = {}):
        # Encoded Latent. Num_ts x len_ts x  dim_m x dim_a

        H = self.encode(xs_cond)

        # ==Esitmate dynamics==  M IS APPLIED HERE
        ret = self.dynamics_model(
            H, return_loss=return_reg_loss, fix_indices=fix_indices, do_detach=self.detachM)
        if return_reg_loss:
            # fn is a map by M_star. Loss is the training external loss
            fn, losses = ret
        else:
            fn = ret

        if self.predictive:
            H_last = H[:, -1:]
            H_preds = [H] if reconst else []
            array = np.arange(n_rolls)

        else:
            H_last = H[:, :1]
            H_preds = [H[:, :1]] if reconst else []
            array = np.arange(xs_cond.shape[1] + n_rolls - 1)

        #repeated application of prediction
        for _ in array:
            H_last = fn(H_last)
            H_preds.append(H_last)

        H_preds = torch.cat(H_preds, axis=1)
        # Prediction in the observation space
        x_preds = self.decode(H_preds)

        if return_reg_loss:
            return x_preds, losses
        else:
            return x_preds