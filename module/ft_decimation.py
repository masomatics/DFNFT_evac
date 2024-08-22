import sys

sys.path.append("./")
sys.path.append("./module")
import torch
from torch import nn

from matplotlib import pyplot as plt

from einops import rearrange
from module import dynamics as dyn


class NFT(nn.Module):
    def __init__(
        self,
        encoder: list,
        decoder: list,
        orth_proj=False,
        is_Dimside=False,
        require_input_adapter=False,
        **kwargs,
    ):
        assert len(encoder) == 1
        assert len(decoder) == 1

        super().__init__()
        self.is_Dimside = is_Dimside
        self.require_input_adapter = require_input_adapter
        self.encoder = encoder[0]
        self.encoder.require_input_adapter = self.require_input_adapter
        self.decoder = decoder[0]
        self.decoder.require_input_adapter = self.require_input_adapter

        if self.is_Dimside:
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
    def do_encode(self, obs, is_reshaped=False):
        # expect   N T C H W
        if is_reshaped:
            obs_nt = obs  # Already in (b,t) format
        else:
            b, _ = obs.shape[0], obs.shape[1]
            obs_nt = rearrange(obs, "b t ... -> (b t) ...")

        latent_bt = self.encoder(signal=obs_nt)  # batch numtokens dim
        latent = rearrange(latent_bt, "(b t) ... -> b t ...", b=b)
        return latent

    def do_decode(self, latent, batch_size, do_reshape=True):
        # expect input N T dim
        latent = rearrange(latent, "b t ... -> (b t) ...")

        obshat_batched = self.decoder(latent)  # (N T) obshape
        if do_reshape:
            obshat = rearrange(obshat_batched, "(b t) ... -> b t ...", b=batch_size)
        else:
            obshat = obshat_batched
        return obshat

    def __call__(self, obs, n_rolls=1):
        batch_size, t = obs.shape[0], obs.shape[1]
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

        predicted = self.do_decode(
            latent_preds, batch_size=batch_size
        )  # X0, X1, X2, ...
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

        pred_loss = torch.mean(
            torch.sum((obstuple - predfuture) ** 2, axis=tuple(range(2, obstuple.ndim)))
        )

        # pred_loss = dyn._mse(
        #     obstuple[:, 1:], predfuture[:, 1:]
        # )  # d([X1hat, X1], [X2hat, X2])

        loss = {"all_loss": pred_loss}
        loss["intermediate_loss"] = torch.tensor([0.0])
        loss["pred_loss"] = pred_loss

        return loss

    def autoencode(self, obs):
        batch_size, _ = obs.shape[0], obs.shape[1]
        latent = self.do_encode(obs)  # b t n a
        pred = self.do_decode(latent, batch_size=batch_size)
        return pred

    def evaluate(self, evalseq, writer, device):
        _, t = evalseq.shape[0], evalseq.shape[1]
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
    def __init__(self, nft_list: list[NFT], own_decoders: list):
        super().__init__(encoder=None, decoder=None)
        self.own_decoders = nn.ModuleList(own_decoders)
        self.nftlayers = nn.ModuleList(nft_list)
        self.depth = len(self.nftlayers)
        self.terminal_dynamics = nft_list[-1].dynamics

        # Turning on the input_adapter for the first layer of NFT
        self.own_decoders.require_input_adapter = True
        assert self.nftlayers[0].require_input_adapte

    def do_encode(self, obs):
        latent = obs
        latents = []
        for k in range(self.depth):
            latent = self.nftlayers[k].do_encode(obs)
            latents.append(latent)
        return latents

    def do_decode(self, latent, layer_idx_from_bottom=0):
        # exepcts N T dim
        batch_size = latent.shape[0]
        latent = rearrange(latent, "n t ... -> (n t) ...")
        for j in range(layer_idx_from_bottom, self.depth):
            loc = self.depth - (j + 1)
            latent = self.own_decoders[loc](latent)
        obshat = latent
        obshat = rearrange(obshat, "(n t) ... -> n t ...", n=batch_size)
        return obshat

    def __call__(self, obs, n_rolls=1):
        batch_size, t = obs.shape[0], obs.shape[1]
        assert t > 1
        latents = self.do_encode(obs)  # b t n a
        # print(latent[0, 0, :5], "ForDebug LAT")
        # print(latent[0, 1, :5], "ForDebug LAT")

        for k in range(self.depth):
            # determine the regressor on H0, H1
            latent = latents[k]
            self.dynamics._compute_M(latent[:, :2])
            # print(self.dynamics.M[0, 0], "FOR Debug M")
            latent_preds = [latent[:, [0]], latent[:, [1]]]
            terminal_latent = latent[:, [1]]  # H1

            for k in range(n_rolls - 1):
                shifted_latent = self.dynamics(terminal_latent)  # H1+k
                terminal_latent = shifted_latent
                latent_preds.append(shifted_latent)
            latent_preds = torch.concatenate(latent_preds, axis=1)  # H0, H1, H2hat, ...
            # print(f""" {latent_preds[0,0,0]}, Debug Latent Pred0""")
            # print(f""" {latent_preds[0,1,0]}, Debug Latent Pred1""")
            # print(f""" {latent_preds[0,2,0]}, Debug Latent Pred2""")

            predicted = self.do_decode(
                latent_preds, batch_size=batch_size
            )  # X0, X1, X2, ...
