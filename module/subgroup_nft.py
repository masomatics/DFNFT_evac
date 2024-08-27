from torch import nn, Tensor
from einops import rearrange
from module.dynamics import Dynamics
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class SubgroupNFT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        orth_proj: bool,
        require_input_adapter: bool,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.encoder.require_input_adapter = require_input_adapter
        self.decoder.require_input_adapter = require_input_adapter

        self.dynamics = Dynamics()
        self.orth_proj = orth_proj

    def encode(self, obs: Tensor) -> Tensor:
        """
        obs.shape=(B, T, ...)
        """
        if obs.ndim < 2:
            raise ValueError
        obs_bt = rearrange(obs, "b t ... -> (b t) ...")
        latent_bt = self.encoder(obs_bt)
        return rearrange(latent_bt, "(b t) ... -> b t ...", b=obs.shape[0])

    def decode(self, latent: Tensor) -> Tensor:
        """
        latent.shape=(B, T, ...)
        """
        latent_bt = rearrange(latent, "b t ... -> (b t) ...")
        pred_bt = self.decoder(latent_bt)
        return rearrange(pred_bt, "(b t) ... -> b t ...", b=latent.shape[0])

    def __call__(self, obs: Tensor, num_shifts: int = 1) -> Tensor:
        latent = self.encode(obs=obs)
        latent_regression = self.get_latent_regression(
            latent=latent, num_shifts=num_shifts
        )
        prediction = self.decode(latent=latent_regression)
        return prediction

    def get_latent_regression(self, latent, num_shifts: int = 1) -> Tensor:
        """
        Only latent[:, :2] matters.
        """
        self.dynamics._compute_M(H=latent[:, :2])  # determine the action matrix M
        latent_regression = [latent[:, [0]], latent[:, [1]]]
        for k in range(num_shifts - 1):
            next_latent = self.dynamics(latent_regression[-1])  # H_{k+1}
            latent_regression.append(next_latent)

        return torch.concatenate(
            latent_regression, axis=1
        )  # H_0, H_1, \hat{H_2}, \hat_{H_3}, ...

    def loss(self, obs: Tensor, num_shifts: int = 1) -> dict[str, Tensor]:
        prediction = self(obs, num_shifts)  # X_0, X_1, \hat{X_2}
        pred_loss = torch.mean(
            torch.sum((obs - prediction) ** 2, axis=tuple(range(2, obs.ndim)))
        )

        # self.dynamics.M has shape=(b, d, d)
        # d = self.dynamics.M.shape[1]
        # group_representation_loss = torch.mean(
        #     torch.sum(
        #         torch.abs(
        #             self.dynamics.M - torch.eye(d).to(self.dynamics.M.device)
        #         ),  # M[i] - I_d, shape=(b, d, d)
        #         dim=(1, 2),
        #     )
        # )

        return {
            "all_loss": pred_loss,
            "pred_loss": pred_loss,
            # "group_representation_loss": group_representation_loss,
            "intermediate_loss": Tensor([0.0]),
            # "ratio_of_identity_matrices": self.get_ratio_of_identity_matrices(),
        }

    def evaluate(
        self, obs: Tensor, writer: SummaryWriter, step: int, device: torch.device
    ) -> None:
        """mask = torch.ones(matsize, matsize, requires_grad=False)
        obs: shape=(B, T, ...)
        """
        num_shifts = obs.shape[1] - 1
        prediction = self(obs.to(device=device), num_shifts)

        # The first data in the given batch
        obs = obs[0].detach().to("cpu")
        prediction = prediction[0].detach().to("cpu")

        plt.figure(figsize=(20, 10))
        plt.plot(obs[0], label="gt")
        plt.plot(prediction[0], label="pred")
        plt.legend()
        writer.add_figure("initial gt vs predicted", plt.gcf(), global_step=step)

        plt.figure(figsize=(20, 10))
        plt.plot(obs[-1], label="gt")
        plt.plot(prediction[-1], label="pred")
        plt.legend()

        writer.add_figure("shifted gt vs predicted", plt.gcf(), global_step=step)

    # def get_ratio_of_identity_matrices(self, threshold: float = 0.1) -> Tensor:
    #     b, d, _ = self.dynamics.M.shape
    #     return torch.mean(
    #         (
    #             torch.sum(
    #                 torch.abs(
    #                     self.dynamics.M - torch.eye(d).to(self.dynamics.M.device)
    #                 ),  # M[i] - I_d, shape=(b, d, d)
    #                 dim=(1, 2),
    #             )
    #             <= threshold
    #         ).float()
    #     )
