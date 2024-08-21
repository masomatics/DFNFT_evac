# Data generation for neural fft
# Sequence of periodic functions on circle [0, 2\pi )
# shift is acted to a function
# Modified from seq_mnist.py

import numpy as np
import math

import torch


#  time points are nonlinearly transformed from the latent function, where the standard shifts are applied.
class ShiftedFreqFunNonLinear:
    def __init__(
        self,
        num_data=5000,  # Number of data
        num_sample_points=128,  # Number of observation points
        num_shifts=3,  # Time steps in a sequence
        shift_label=False,
        batchM_size=1,
        shift_range=2,
        max_shift=[0.0, 2 * math.pi / 2],  # range of shift action (in radian)
        shared_transition=False,
        rng=None,
        num_freqs=5,  # Number of selected frequecy to make a function
        ns=0.0,  # Noise level of additive Gaussian noise
        pow=3,
        shifts=None,
        freq_fix=False,
        freq_manual=[],
        freqseed=1,
        test=0,
        track_idx=False,
        smallfreqs_num=0,
        smallfreqs_strength=0,
    ):
        self.num_shifts = num_shifts
        self.num_data = num_data
        self.num_sample_points = num_sample_points
        self.ns = ns
        self.rng = rng if rng is not None else np.random
        self.num_freqs = num_freqs + smallfreqs_num
        self.shift_range = shift_range  # control the frequency range of data
        self.max_shift = max_shift
        self.shift_label = shift_label
        self.batchM_size = batchM_size
        self.shifts = shifts
        self.pow = pow
        self.freq_fix = freq_fix
        self.track_idx = track_idx
        self.smallfreqs_num = smallfreqs_num
        if smallfreqs_num > 0:
            self.smallfreqs_strength = smallfreqs_strength / smallfreqs_num
        else:
            self.smallfreqs_strength = 0

        if freq_fix:
            if freq_manual:
                assert (
                    len(freq_manual) == self.num_freqs
                ), f"{len(freq_manual)=}, {self.num_freqs=}"
                self.fixed_freqs = freq_manual
            else:
                np.random.seed(freqseed)
                self.fixed_freqs = np.random.randint(
                    0,
                    np.ceil(num_sample_points / (5 * self.shift_range)),
                    self.num_freqs,
                )
            print(self.fixed_freqs)
        else:
            print("random freqs")

        # dt = 1.0/N
        # obs_t = np.arange(0, N*dt, dt) # observed time  [0,1]/N
        # lat_t = np.power(obs_t, self.pow)      # observed time domain: lat_t = obs_t^pow

        # In this class (nonlinear), unlike Shifted_freqFun(), data set is not contained in the class, because the time point must be calculated by shifts.
        #   Instead, the frequencies and coefficients to make the latent functions are contained in the class
        # Generation of data (size: num_data)
        # fdata =[]
        freqs = []
        sin_coeffs = []
        cos_coeffs = []

        if test > 0:
            np.random.seed(test)
        print(test)

        for i in range(num_data):
            sin_coeff = np.random.randn(self.num_freqs)
            sin_coeff = sin_coeff / np.linalg.norm(sin_coeff)
            if self.smallfreqs_num > 0:
                sin_coeff[-self.smallfreqs_num :] = (
                    self.smallfreqs_strength * sin_coeff[-self.smallfreqs_num :]
                )
            sin_coeffs.append(sin_coeff)

            cos_coeff = np.random.randn(self.num_freqs)
            cos_coeff = cos_coeff / np.linalg.norm(cos_coeff)
            if self.smallfreqs_num > 0:
                cos_coeff[-self.smallfreqs_num :] = (
                    self.smallfreqs_strength * cos_coeff[-self.smallfreqs_num :]
                )
            cos_coeffs.append(cos_coeff)

            if freq_fix:
                freqs.append(self.fixed_freqs)
            else:
                fixed_freqss = np.random.randint(
                    0,
                    np.ceil(num_sample_points / (5 * self.shift_range)),
                    self.num_freqs,
                )  # randomly selected frequencies
                freqs.append(fixed_freqss)

        self.freqs = np.array(
            freqs
        )  # self.freqs:  num_data x num_freqs (double array) contains frequencoes to make the latent functions
        #  To make the function values use f =  np.matmul(np.sin(np.outer(2*np.pi*t,self.freqs[i,:])), self.sin_coeff) + np.matmul(np.cos(np.outer(2*np.pi*t,self.freqs[i,:])), self.cos_coeff) # function value at latent t

        self.sin_coeffs = np.array(sin_coeffs)
        self.cos_coeffs = np.array(cos_coeffs)

        #   In the batch, the same M(g) is used.  Set shuffle off in Dataloader.
        if self.shift_label and (self.shifts is None):
            shifts = []
            for i in range(num_data):
                if i % batchM_size == 0:
                    shift = self.rng.uniform(
                        self.max_shift[0], self.max_shift[1], size=1
                    )
                shifts.append(shift)
            self.shifts = np.array(shifts)  # shifts are given in radian [0, 2pi)

        # random shift
        # self.shift = self.max_shift[0] + np.random.rand(1) * (self.max_shift[1] - self.max_shift[0])

        self.shared_transition = shared_transition
        if self.shared_transition:
            self.init_shared_transition_parameters()

    def init_shared_transition_parameters(self):
        self.shared_shift = self.rng.uniform(
            self.max_shift[0], self.max_shift[1], size=1
        )

    def __len__(self):
        return self.num_data

    def __getitem__(self, i):  # i-th data and its shifts (T shifts)
        freq = self.freqs[i, :]
        dt = 1.0 / self.num_sample_points
        obs_t = np.arange(0, self.num_sample_points * dt, dt)  # observed time  [0,1]/N
        lat_t = np.power(obs_t, self.pow)  # observed time domain: lat_t = obs_t^pow

        if self.shift_label:
            shift = self.shifts[i]
        elif self.shared_transition:
            shift = self.shared_shift
        else:
            shift = self.rng.uniform(self.max_shift[0], self.max_shift[1], size=1)

        fvals = []
        for t in range(self.num_shifts):  # shift * t ( action up to T times)
            lat_t = np.power(obs_t, self.pow) - t * shift / (2 * math.pi)
            lat_t = lat_t + (lat_t < 0) * (1.0)

            fobs_t = np.matmul(
                np.sin(np.outer(2 * np.pi * lat_t, freq)), self.sin_coeffs[i]
            ) + np.matmul(
                np.cos(np.outer(2 * np.pi * lat_t, freq)), self.cos_coeffs[i]
            )  # fn_t(j) = f_lat((t_j)**pow - shift*t)
            fvals.append(torch.tensor(fobs_t))

        if self.shift_label:
            if self.track_idx:
                shift = (shift, i)
            return [
                torch.stack(fvals),
                shift,
            ]  # List (length T) of N-dim function values and shift
        else:
            fvals = torch.stack(fvals)
            return fvals  # List (length T) of N-dim function values and time
