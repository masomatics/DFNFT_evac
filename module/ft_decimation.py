import sys

sys.path.append("./")
sys.path.append("./module")
import torch
from torch import nn

from matplotlib import pyplot as plt

from einops import rearrange, repeat, einsum
from misc import orthog_proj as op
import pdb
from module import dynamics as dyn


class NFT(nn.Module):
    def __init__(
        self,
        encoder: None,
        decoder: None,
        orth_proj=False,
        is_Dimside=False,
        require_input_adapter=False,
        **kwargs,
    ):
        super().__init__()
        self.is_Dimside = is_Dimside
        if encoder is not None:
            self.require_input_adapter = require_input_adapter
            self.encoder = encoder
            self.encoder.require_input_adapter = self.require_input_adapter
            self.decoder = decoder
            self.decoder.require_input_adapter = self.require_input_adapter

        if self.is_Dimside == True:
            self.dynamics = dyn.DynamicsDimSide()
        else:
            self.dynamics = dyn.Dynamics()
        self.orth_proj = orth_proj

    def latent_shift(self, insignal, n_rolls, mask, intervene_fxn=None):
        # determine the regressor on H0, H1
        self.dynamics._compute_M(insignal[:, :2], mask, orth_proj=self.orth_proj)

        terminal_latent = insignal[:, [0]]  # H0
        latent_preds = [terminal_latent]
        for k in range(n_rolls):
            shifted_latent = self.dynamics(
                terminal_latent, intervene_fxn=intervene_fxn
            )  # Hk+2
            terminal_latent = shifted_latent
            latent_preds.append(shifted_latent)
        latent_preds = torch.concatenate(latent_preds, axis=1)  # H0, H1, H2, ...

        return latent_preds

    # NEEDS TO ALSO DEAL WITH PATCH INFO
    def do_encode(self, obs, embed=None, is_reshaped=False):
        # expect   N T C H W
        if is_reshaped:
            obs_nt = obs  # Already in (b,t) format
        else:
            b, t = obs.shape[0], obs.shape[1]
            obs_nt = rearrange(obs, "b t ... -> (b t) ...")

        latent_bt = self.encoder(signal=obs_nt)  # batch numtokens dim
        bt, n, d = latent_bt.shape
        latent = rearrange(latent_bt, "(b t) n d -> b t n d", b=b)
        return latent

    def do_decode(self, latent, batchsize, do_reshape=True):
        # expect input N T dim
        latent = rearrange(latent, "n t ... -> (n t) ...")

        obshat_batched = self.decoder(latent)  # (N T) obshape
        if do_reshape:
            obshat = rearrange(obshat_batched, "(n t) ... -> n t ...", n=batchsize)
        else:
            obshat = obshat_batched
        return obshat

    def __call__(self, obs, n_rolls=1):
        batchsize, t = obs.shape[0], obs.shape[1]
        assert t > 1
        latent = self.do_encode(obs)  # b t n a
        # print(latent[0, 0, :5], "ForDebug LAT")
        # print(latent[0, 1, :5], "ForDebug LAT")

        latent_preds = self.shift_latent(latent, n_rolls=n_rolls)
        # determine the regressor on H0, H1
        # self.dynamics._compute_M(latent[:, :2])
        # # print(self.dynamics.M[0, 0], "FOR Debug M")
        # latent_preds = [latent[:, [0]], latent[:, [1]]]
        # terminal_latent = latent[:, [1]]  # H1

        # for k in range(n_rolls - 1):
        #     shifted_latent = self.dynamics(terminal_latent)  # H1+k
        #     terminal_latent = shifted_latent
        #     latent_preds.append(shifted_latent)
        # latent_preds = torch.concatenate(latent_preds, axis=1)  # H0, H1, H2hat, ...
        # print(f""" {latent_preds[0,0,0]}, Debug Latent Pred0""")
        # print(f""" {latent_preds[0,1,0]}, Debug Latent Pred1""")
        # print(f""" {latent_preds[0,2,0]}, Debug Latent Pred2""")

        predicted = self.do_decode(latent_preds, batchsize=batchsize)  # X0, X1, X2, ...
        # print(predicted[0, -1, :5], "Debug XPred")

        return predicted

    def shift_latent(self, latent, n_rolls=1):
        # determine the regressor on H0, H1
        self.dynamics._compute_M(latent[:, :2])
        # print(self.dynamics.M[0, 0], "FOR Debug M")
        latent_preds = [latent[:, [0]], latent[:, [1]]]
        terminal_latent = latent[:, [1]]  # H1

        for k in range(n_rolls - 1):
            shifted_latent = self.dynamics(terminal_latent)  # H1+k
            terminal_latent = shifted_latent
            latent_preds.append(shifted_latent)
        latent_preds = torch.concatenate(latent_preds, axis=1)  # H0, H1, H2hat, ...
        return latent_preds

    def loss(self, obstuple, n_rolls=1):
        predinput = obstuple[:, :-1]  # X0 X1
        predfuture = self(predinput, n_rolls)  # X0hat X1hat X2hat
        # print(predfuture[0, -1, :5])
        # print(predfuture[0, 1:, :5])
        # pdb.set_trace()

        predloss = torch.mean(
            torch.sum((obstuple - predfuture) ** 2, axis=tuple(range(2, obstuple.ndim)))
        )

        # predloss = dyn._mse(
        #     obstuple[:, 1:], predfuture[:, 1:]
        # )  # d([X1hat, X1], [X2hat, X2])

        loss = {"all_loss": predloss}
        loss["intermediate"] = torch.tensor([0.0])
        loss["predloss"] = predloss

        return loss

    def autoencode(self, obs):
        batchsize, t = obs.shape[0], obs.shape[1]
        latent = self.do_encode(obs)  # b t n a
        pred = self.do_decode(latent, batchsize=batchsize)
        return pred

    def evaluate(self, evalseq, writer, device):
        b, t = evalseq.shape[0], evalseq.shape[1]
        initialpair = evalseq[:, :2].to(device)
        rolllength = t - 1
        predicted = self(initialpair, n_rolls=rolllength).detach()
        print("""!!! Visualization Rendered!!! """)
        self.visualize(evalseq, predicted, writer)

    def visualize(self, evalseq, predicted, writer):
        predicted = predicted[0].to("cpu")
        evalseq = evalseq[0].to("cpu")
        # Prediction at -1
        plt.figure(figsize=(20, 10))
        plt.plot(evalseq[-1], label="gt")
        plt.plot(predicted[-1], label="pred")
        plt.legend()

        writer.add_figure("gt vs predicted", plt.gcf())


class DFNFT(NFT):
    def __init__(self, nftlist: list[NFT], owndecoders: list):
        super().__init__(encoder=None, decoder=None)
        self.owndecoders = nn.ModuleList(owndecoders)
        self.nftlayers = nn.ModuleList(nftlist)
        self.depth = len(self.nftlayers)
        self.terminal_dynamics = nftlist[-1].dynamics

        # Turning on the input_adapter for the first layer of NFT
        self.owndecoders.require_input_adapter = True
        assert self.nftlayers[0].require_input_adapter == True

    def do_encode(self, obs):
        latent = obs
        latents = []
        for k in range(self.depth):
            latent = self.nftlayers[k].do_encode(obs)
            latents.append(latent)
        return latents

    def do_decode(self, latent, layer_idx_from_bottom=0):
        # exepcts N T dim
        batchsize = latent.shape[0]
        latent = rearrange(latent, "n t ... -> (n t) ...")
        for j in range(layer_idx_from_bottom, self.depth):
            loc = self.depth - (j + 1)
            latent = self.owndecoders[loc](latent)
        obshat = latent
        obshat = rearrange(obshat, "(n t) ... -> n t ...", n=batchsize)
        return obshat

    def __call__(self, obs, n_rolls=1):
        batchsize, t = obs.shape[0], obs.shape[1]
        assert t > 1
        latents = self.do_encode(obs)  # b t n a
        # print(latent[0, 0, :5], "ForDebug LAT")
        # print(latent[0, 1, :5], "ForDebug LAT")

        latent_preds = []
        for k in range(self.depth):
            # determine the regressor on H0, H1
            latent = latents[k]
            latent_pred = self.nftlayers[k].shift_latent(latent, n_rolls=n_rolls)
            latent_preds.append(latent_pred)

        infer_pred = latent_preds[-1]
        intermediate_pred = latent_preds[-1]
        for j in range(self.depth - 1, -1, -1):
            infer_pred = self.owndecoders[j](infer_pred)

        return infer_pred
