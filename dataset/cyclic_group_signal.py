from torch.utils.data import Dataset
import math
import numpy as np
import random

import torch
from torch import Tensor
from typing import Callable


class CyclicGroupSignal(Dataset):
    def __init__(
        self,
        num_data: int = 5000,
        num_shifts: int = 3,
        num_freqs: int = 5,
        diffeo_of_circle: Callable[[float], float] = lambda t: t**3,
        group_param: tuple[int, int] = (2, 5),
        shift_label: bool = False,
    ) -> None:
        self.num_data = num_data
        self.num_shifts = num_shifts
        self.diffeo_of_circle = diffeo_of_circle
        self.group_param = group_param
        self.group_order = self.group_param[0] * self.group_param[1]

        self.num_sample_points = max(self.group_order * 10, 128)

        self.shift_label = shift_label
        random.seed(0)
        np.random.seed(0)

        self.fixed_freq = np.array(random.sample(range(self.group_order), num_freqs))
        # (num_freq, )
        self.freqs = np.array(
            [self.fixed_freq for _ in range(self.num_data)]
        )  # (num_data, num_freqs)
        assert self.freqs.shape == (self.num_data, num_freqs)

        coeff_list = []
        for _ in range(self.num_data):
            coeff = np.random.randn(num_freqs)
            coeff = coeff / np.linalg.norm(coeff)
            coeff_list.append(coeff)

        self.coeffs = np.array(coeff_list)  # (num_data, num_freqs)
        assert self.coeffs.shape == (self.num_data, num_freqs)

        # The initial value of the i-th signal (without diffeo) is f_i(t) = \sum_j self.coeffs[i][j] * sin(2*pi*t*self.freqs[i][j])

        self.gs = np.random.choice(self.group_order, num_data)
        # (num_data, ), gs[i] is a random element of G

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, i: int) -> Tensor:
        """
        return: shape=(self.num_shifts, self.num_samples)
        """
        freqs = self.freqs[i]  # num_freqs
        coeffs = self.coeffs[i]
        g = self.gs[i]
        g_action = g / self.group_order
        dt = 1.0 / self.num_sample_points

        sampling_t = np.arange(
            0, self.num_sample_points * dt, dt
        )  # observed time 0, 1/N, 2/N, ..., (N-1)/N

        shifted_signals = []
        sampling_t_after_diffeo = self.diffeo_of_circle(sampling_t)

        for s in range(self.num_shifts):
            sampling_t_after_diffeo_and_shift = (lambda t: t - g_action * s)(
                sampling_t_after_diffeo
            )

            signal = np.matmul(
                np.sin(np.outer(2 * np.pi * sampling_t_after_diffeo_and_shift, freqs)),
                coeffs,
            )  # (N, num_freqs) * (num_freqs, ) -> (N, )
            shifted_signals.append(torch.from_numpy(signal))

        return torch.stack(
            shifted_signals
        ), g_action * math.pi * 2 if self.shift_label else torch.stack(shifted_signals)


if __name__ == "__main__":
    dataset = CyclicGroupSignal()
    signals = dataset.__getitem__(0)
    print(signals.shape)
    print(signals[0][:5])
    print(signals[1][:5])
