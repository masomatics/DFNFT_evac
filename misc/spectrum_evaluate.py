import sys
import yaml

sys.path.append("./")
import os
import torch
from misc import yaml_util as yu
import numpy as np

sys.path.append("./dataset")
sys.path.append("./module")
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from misc import character_analysis as ca
from misc import loss_helper as lh
import pdb
import copy

import time

# Get the current time


def replace_lowhalf(M):
    sqM = copy.deepcopy(M)
    n, b, b = sqM.shape
    blocksize = b // 2
    zeroblock = torch.zeros(blocksize, blocksize)
    onesblock = torch.ones(blocksize, blocksize)
    maskup = torch.block_diag(onesblock, zeroblock).to(dtype=sqM.dtype)

    sqM = sqM * maskup
    return sqM


def replace_uphalf(M):
    sqM = copy.deepcopy(M)
    n, b, b = sqM.shape
    blocksize = b // 2
    zeroblock = torch.zeros(blocksize, blocksize)
    onesblock = torch.ones(blocksize, blocksize)
    masklow = torch.block_diag(zeroblock, onesblock).to(dtype=sqM.dtype)

    sqM = sqM * masklow
    return sqM


def plot_to_image(fig):
    fig.canvas.draw()
    # Convert to numpy array
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def spectrum(mymodel, myloader, mywriter, step, device):
    current_time = time.localtime()
    current_date = time.strftime("%Y-%m-%d-%H:%M", current_time)

    mydata = myloader.dataset
    Ms = {0: [], 1: []}
    shifts = []

    if mymodel.depth > 1:
        for k in range(10):
            evalseq, shift = next(iter(myloader))
            evalseq = evalseq[:, :2].to(mymodel.nftlayers[0].encoder.device)
            predicted = mymodel(evalseq, n_rolls=1)
            shifts.append(shift)
            Ms[0].append(mymodel.nftlayers[0].dynamics.M)
            Ms[1].append(mymodel.nftlayers[1].dynamics.M)

        shifts = torch.concatenate(shifts)
        Ms[0] = torch.concatenate(Ms[0]).detach()
        Ms[1] = torch.concatenate(Ms[1]).detach()
        print(Ms[0].shape)
        Mup = replace_lowhalf(Ms[1].to("cpu"))
        Mbot = replace_uphalf(Ms[1].to("cpu"))

        myfreqs = np.array(mydata.freqsel)
        maxfreq = np.max(myfreqs)

        targfreq, prods0 = ca.inner_prod(
            Ms[0].to(shifts.device), shifts, maxfreq=maxfreq, bins=maxfreq + 1
        )
        targfreq, prods1 = ca.inner_prod(
            Ms[1].to(shifts.device), shifts, maxfreq=maxfreq, bins=maxfreq + 1
        )
        targfreq, prodsUp = ca.inner_prod(
            Mup.to(shifts.device), shifts, maxfreq=maxfreq, bins=maxfreq + 1
        )
        targfreq, prodsBot = ca.inner_prod(
            Mbot.to(shifts.device), shifts, maxfreq=maxfreq, bins=maxfreq + 1
        )

        plt.figure()
        deltas = ca.deltafxn(targfreq, mydata.freqsel) * 2
        plt.plot(
            targfreq, deltas, alpha=0.2, label="gt:" + str(np.where(deltas > 1.0)[0])
        )
        plt.plot(
            targfreq,
            prods0,
            label="pred0:" + str(np.where(prods0 > 1.0)[0]),
            alpha=0.8,
        )
        plt.plot(
            targfreq,
            prods1,
            label="pred1:" + str(np.where(prods1 > 1.0)[0]),
            alpha=0.8,
        )
        plt.legend()
        plt.title(str(np.sort(myfreqs)) + current_date)

        mywriter.add_figure("P0 vs P1", plt.gcf(), global_step=step)

        # plt_image = plot_to_image(plt)
        # # Log the image to TensorBoard
        # mywriter.add_image("P0 vs P1", plt_image, global_step=step, dataformats="HWC")

        plt.figure()
        plt.plot(targfreq, deltas, label="gt", alpha=0.2)
        plt.plot(
            targfreq,
            prodsUp,
            label="predUP:" + str(np.where(prodsUp > 1.0)[0]),
            alpha=0.8,
        )
        plt.plot(
            targfreq,
            prodsBot,
            label="predBot:" + str(np.where(prodsBot > 1.0)[0]),
            alpha=0.8,
        )
        plt.legend()
        plt.title(str(np.sort(myfreqs)) + current_date)

        mywriter.add_figure("UP and Bottom", plt.gcf(), global_step=step)

    else:
        for k in range(10):
            evalseq, shift = next(iter(myloader))
            evalseq = evalseq[:, :2].to(mymodel.nftlayers[0].encoder.device)
            predicted = mymodel(evalseq, n_rolls=1)
            shifts.append(shift)
            Ms[0].append(mymodel.nftlayers[0].dynamics.M)

        shifts = torch.concatenate(shifts)
        Ms[0] = torch.concatenate(Ms[0]).detach()

        myfreqs = np.array(mydata.freqsel)
        maxfreq = np.max(myfreqs)

        targfreq, prods0 = ca.inner_prod(
            Ms[0].to(shifts.device), shifts, maxfreq=maxfreq, bins=maxfreq + 1
        )

        plt.figure()
        deltas = ca.deltafxn(targfreq, mydata.freqsel) * 2
        plt.plot(
            targfreq, deltas, alpha=0.2, label="gt:" + str(np.where(deltas > 1.0)[0])
        )
        targfreq, prods0 = ca.inner_prod(
            Ms[0].to(shifts.device), shifts, maxfreq=maxfreq, bins=maxfreq + 1
        )
        plt.plot(
            targfreq,
            prods0,
            label="pred0:" + str(np.where(prods0 > 1.0)[0]),
            alpha=0.8,
        )
        mywriter.add_figure("P0 vs GT", plt.gcf(), global_step=step)

    plt.figure()
    matrixL0DeltaNorms = lh.tensors_sparseloss(mymodel.nftlayers[0].dynamics.M)
    plt.imshow(matrixL0DeltaNorms.to("cpu").detach())
    mywriter.add_figure("Matrix Variety, Batch x Batch", plt.gcf(), global_step=step)

    # plt_image = plot_to_image(plt)
    # # Log the image to TensorBoard
    # mywriter.add_image("Up and Bottom", plt_image, global_step=step, dataformats="HWC")
