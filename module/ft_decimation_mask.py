import sys

sys.path.append("./")
sys.path.append("./module")
import torch
from torch import nn

from matplotlib import pyplot as plt

from einops import rearrange, repeat, einsum
from misc import orthog_proj as op
from misc import loss_helper as lh
import pdb
from module import dynamics as dyn
from misc import maskmodule as mm


class NFT(nn.Module):
    def __init__(
        self,
        encoder: None,
        decoder: None,
        orth_proj=False,
        is_Dimside=False,
        require_input_adapter=False,
        dynamics_mask=None,
        **kwargs,
    ):
        super().__init__()
        self.depth = 1
        self.is_Dimside = is_Dimside
        if encoder is not None:
            self.require_input_adapter = require_input_adapter
            self.encoder = encoder
            self.encoder.require_input_adapter = self.require_input_adapter
            self.decoder = decoder
            self.decoder.require_input_adapter = self.require_input_adapter
            self.PLambdaNet = mm.SimpleMaskModule(dimRep=self.encoder.dim_m, dimVec=1)
        if self.is_Dimside == True:
            self.dynamics = dyn.DynamicsDimSide()
        else:
            self.dynamics = dyn.Dynamics()
        self.orth_proj = orth_proj

    @property
    def device(self):
        # Return the device of the first parameter
        return next(self.parameters()).device

    # NEEDS TO ALSO DEAL WITH PATCH INFO
    def do_encode(self, obs, embed=None):
        # expect   N T C H W
        b, t = obs.shape[0], obs.shape[1]
        obs_nt = rearrange(obs, "b t ... -> (b t) ...")
        latent_bt = self.encoder(signal=obs_nt)  # batch numtokens dim
        bt, n, d = latent_bt.shape
        latent = rearrange(latent_bt, "(b t) n d -> b t n d", b=b)
        return latent

    def do_decode(self, latent, do_reshape=True):
        # expect input N T dim
        batchsize = latent.shape[0]
        latent = rearrange(latent, "n t ... -> (n t) ...")
        obshat_batched = self.decoder(latent)  # (N T) obshape
        if do_reshape:
            obshat = rearrange(obshat_batched, "(n t) ... -> n t ...", n=batchsize)
        else:
            obshat = obshat_batched
        return obshat

    def __call__(self, obs, n_rolls=1, mask=None):
        self.dynamic_mask = self.PLambdaNet.create_mask()
        batchsize, t = obs.shape[0], obs.shape[1]
        assert t > 1
        if self.require_input_adapter == True:
            mask = torch.ones(self.encoder.dim_m, self.encoder.dim_m).to(obs.device)
        latent = self.do_encode(obs, mask)  # b t n a

        latent_preds = self.shift_latent(
            latent, n_rolls=n_rolls, mask=self.dynamic_mask
        )

        predicted = self.do_decode(latent_preds)  # X0, X1, X2, ...
        # print(predicted[0, -1, :5], "Debug XPred")

        return predicted

    def shift_latent(self, latent, n_rolls=1, mask=None, noise=0):
        # determine the regressor on H0, H1,  adding noise to H1 when deemed necessary
        noise_placeholder = torch.zeros_like(latent).to(latent.device)
        noise_placeholder[:, 1] = noise
        latent = latent + noise_placeholder
        self.dynamics._compute_M(latent[:, :2], mask, orth_proj=self.orth_proj)
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
        loss = {}
        loss["intermediate"] = torch.tensor([0.0]).to(predfuture.device)
        loss["predloss"] = predloss
        loss["blockness"] = 0.1 * torch.trace(
            self.PLambdaNet.get_laplacian(self.dynamic_mask)
        )
        all_loss = 0
        for key in loss:
            all_loss = all_loss + loss[key]
        loss["all_loss"] = all_loss

        return loss

    def autoencode(self, obs):
        batchsize, t = obs.shape[0], obs.shape[1]
        latent = self.do_encode(obs)  # b t n a
        pred = self.do_decode(latent, batchsize=batchsize)
        return pred

    def evaluate(self, evalseq, writer, device, step):
        b, t = evalseq.shape[0], evalseq.shape[1]
        initialpair = evalseq[:, :2].to(device)
        rolllength = t - 1
        predicted = self(initialpair, n_rolls=rolllength)
        predicted = predicted.detach()
        print("""!!! Visualization Rendered!!! """)
        self.visualize(evalseq, predicted, writer, step)

    def visualize(self, evalseq, predicted, writer, step):
        predicted = predicted[0].to("cpu")
        evalseq = evalseq[0].to("cpu")
        # Prediction at -1
        plt.figure(figsize=(20, 10))
        plt.plot(evalseq[-1], label="gt")
        plt.plot(predicted[-1], label="pred")
        plt.legend()
        writer.add_figure(
            f"""gt vs predicted t={len(evalseq)}""", plt.gcf(), global_step=step
        )

        timeidx = 1
        plt.figure(figsize=(20, 10))
        plt.plot(evalseq[timeidx], label="gt")
        plt.plot(predicted[timeidx], label="pred")
        plt.legend()
        writer.add_figure(
            f"""gt vs predicted t={timeidx}""", plt.gcf(), global_step=step
        )


class NFTImages(NFT):
    def _mse(self, obstuple, predfuture):
        errval = torch.mean(
            torch.sum((obstuple - predfuture) ** 2, axis=tuple(range(2, obstuple.ndim)))
        )
        return errval

    def visualize(self, evalseq, predicted, writer, iteridx):
        predicted = predicted[0].to("cpu")
        evalseq = evalseq[0].to("cpu")

        plt.figure(figsize=(20, 10))
        for k in range(len(predicted)):
            plt.subplot(2, len(evalseq), k + 1)
            plt.imshow(evalseq[k].permute([1, 2, 0]).clamp(0, 1.0))
            plt.subplot(2, len(predicted), k + 1 + len(evalseq))
            plt.imshow(predicted[k].permute([1, 2, 0]).clamp(0, 1.0))
        evalloss = self._mse(evalseq, predicted)
        print(torch.min(predicted), torch.max(predicted))
        plt.suptitle(f"""loss mse: {evalloss}""")

        writer.add_figure("gt vs predicted", plt.gcf(), global_step=iteridx)


class DFNFT(NFT):
    def __init__(self, nftlist: list[NFT], owndecoders: list, **kwargs):
        super().__init__(encoder=None, decoder=None)
        self.owndecoders = nn.ModuleList(owndecoders)
        self.nftlayers = nn.ModuleList(nftlist)
        self.depth = len(self.nftlayers)
        self.terminal_dynamics = nftlist[-1].dynamics
        self.device = self.nftlayers[0].device

        self.experimental_mode = False
        for key in kwargs:
            self.experimental_mode = True
            print("EXPERIMENTAL MODE ACTIVATED!" * 10)
            setattr(self, key, kwargs[key])

        # Turning on the input_adapter for the first layer of NFT
        self.owndecoders.require_input_adapter = True
        assert self.nftlayers[0].require_input_adapter == True

    def do_encode(self, obs):
        latent = obs
        latents = []
        for k in range(self.depth):
            latent = self.nftlayers[k].do_encode(latent)
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

    def intermediate_decode(self, kth_latent, layer_idx_from_bottom=0):
        # if layer_idx_from_bottom = 1 and depth=3, then it shall evaluate nftlayers[1]
        batchsize = kth_latent.shape[0]
        kth_latent = rearrange(kth_latent, "n t ... -> (n t) ...")
        kth_latent = self.owndecoders[(self.depth - 1) - layer_idx_from_bottom](
            kth_latent
        )
        kplus1th_latent = rearrange(kth_latent, "(n t) ... -> n t ...", n=batchsize)

        ##### EXPERIMENTALLLLLL!!!!!!
        # kplus1th_latent = self.nftlayers[
        #     (self.depth - 1) - layer_idx_from_bottom
        # ].do_decode(kth_latent)
        return kplus1th_latent

    def evaluate(self, evalseq, writer, device, step):
        b, t = evalseq.shape[0], evalseq.shape[1]
        initialpair = evalseq[:, :2].to(device)
        rolllength = t - 1
        predicted, _, _ = self(initialpair, n_rolls=rolllength)
        predicted = predicted.detach()
        print("""!!! Visualization Rendered!!! """)
        self.visualize(evalseq, predicted, writer, step)

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
            # Use the next mask
            maskidx = min(self.depth - 1, k + 1)
            mask_k = self.nftlayers[maskidx].encoder.maskmat
            """
            EXPERIMENTAL!!!!  Adding NOISE TO LATENT TO BE REGRESSED WHEN NOT at BOTTOM!  FROM HERE
            """
            if (
                hasattr(self, "use_zero_layer_mask")
                and self.use_zero_layer_mask == True
            ):
                mask_k = self.nftlayers[-1].dynamics_mask.to(latent.device)

            noise = torch.normal(
                mean=torch.zeros(size=latent[:, 1].shape), std=0.0 * (k == 0)
            ).to(latent.device)
            noise.requires_grad = False
            latent_pred = self.nftlayers[k].shift_latent(
                latent, n_rolls=n_rolls, mask=mask_k, noise=noise
            )
            latent_preds.append(latent_pred)
            """
            EXPERIMENTAL!!!!  FROM HERE
            """

        intermediate_obs_preds = []
        for k in range(1, self.depth):
            kplus1latent_pred = self.intermediate_decode(
                latent_preds[k], layer_idx_from_bottom=k
            )
            intermediate_obs_preds.append(kplus1latent_pred)

        infer_pred = self.do_decode(latent_preds[-1])

        return infer_pred, latent_preds, intermediate_obs_preds

    def lossfxn(self, target, pred):
        errval = torch.mean(
            torch.sum((target - pred) ** 2, axis=tuple(range(2, target.ndim)))
        )
        return errval

    def loss(self, obstuple, n_rolls=1):
        # predinput = obstuple[:, :-1]  # X0 X1   <--Call uses 0 and 1, this is unncessary
        predinput = obstuple[:, :2]
        # With the understanding that Z0 = Phi_0(X), Z^{-1} = X,
        # Xhat(t+1),   [Z^{0}(t+1) Z^{1}(t+1),..., Z^{depth}(t+1)] ,
        # [hatZ^{-1}(t+1), hatZ^{0}(t+1), ..., Z^{depth-1}(t+1) ]
        predfuture, latent_preds, intermediate_preds = self(predinput, n_rolls)

        # [X, Z^{0}(t+1) Z^{1}(t+1),..., Z^{depth-1}(t+1)] to be compared against
        # [hatZ^{-1}(t+1), hatZ^{0}(t+1), ..., Z^{depth-1}(t+1) ]
        targets = [obstuple] + latent_preds[:-1]

        intermediate_loss = torch.tensor([0.0]).to(predfuture.device)
        for k in range(self.depth):
            if k == self.depth - 1:
                predloss = self.lossfxn(obstuple, predfuture)
            else:
                intermediate_loss = intermediate_loss + self.lossfxn(
                    targets[k], intermediate_preds[k]
                )
            # EXPERIMENTAL. Making as many M0 as possible to Eye by Lasso Loss
            if self.experimental_mode == True and k == 0:
                LassoStrength = self.lasso_strength
                intermediate_strength = self.intermediate_strength
                eyes = torch.eye(self.nftlayers[0].dynamics.M.shape[-1])[None, :]
                eyes = eyes.to(self.nftlayers[0].dynamics.M.device)
                if hasattr(self, "use_mean") and self.use_mean == True:
                    meanMat = torch.mean(
                        self.nftlayers[0].dynamics.M, axis=0, keepdim=True
                    )
                    matrixL0DeltaNorms = torch.sum(
                        (self.nftlayers[0].dynamics.M - meanMat) ** 2, axis=[-1, -2]
                    )
                elif hasattr(self, "use_delta") and self.use_delta == True:
                    matrixL0DeltaNorms = lh.tensors_sparseloss(
                        self.nftlayers[0].dynamics.M
                    )
                else:
                    matrixL0DeltaNorms = torch.sum(
                        (self.nftlayers[0].dynamics.M - eyes) ** 2, axis=[-1, -2]
                    )
                Lassoloss = torch.mean(matrixL0DeltaNorms)
                intermediate_loss = intermediate_strength * (
                    intermediate_loss + LassoStrength * Lassoloss
                )

                # noise (EXPERIMENTAL)
                # intermediate_loss = self.lossfxn(
                #     torch.mean(latent_preds[0], axis=-1, keepdims=True), latent_preds[0]
                # )
                # intermediate_loss = intermediate_loss + self.lossfxn(
                #     torch.mean(targets[k], axis=-1, keepdims=True),
                #     intermediate_preds[k],
                # )
                # ep = torch.normal(mean=torch.zeros(size=targets[k].shape), std=5.0).to(
                #     targets[k].device
                # )
                # intermediate_loss = intermediate_loss + self.lossfxn(
                #     targets[k] + ep, intermediate_preds[k]
                # )

        # predloss = dyn._mse(
        #     obstuple[:, 1:], predfuture[:, 1:]
        # )  # d([X1hat, X1], [X2hat, X2])

        loss = {"all_loss": predloss + intermediate_loss}
        loss["intermediate"] = intermediate_loss
        loss["predloss"] = predloss
        return loss
