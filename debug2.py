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
from module import ft_decimation as ftd

import time


datname = "OneDsignal_c8mimic_lowpow"
modelname = "Plambda_OneD_RotFeat1layer_ver3"

trainname = "faster"
with open(f"""./cfg_data/{datname}.yaml""", "rb") as f:
    cfg_data = yaml.safe_load(f)

cfg_data["args"]["shift_label"] = True
mydata = yu.load_component(cfg_data)

expname = f"""{datname}_{modelname}_{trainname}"""

exppath = os.path.join("./dnftresult", expname)

mymodelpath = f"""{exppath}/model.pt"""
mymodel = torch.load(mymodelpath)


myloader = DataLoader(
    mydata,
    batch_size=32,
    shuffle=True,
    num_workers=2,
)


Ms = {0: [], 1: []}
shifts = []

if hasattr(mymodel, "nftlayers"):
    mynft = mymodel.nftlayers[0]
else:
    mynft = mymodel

if hasattr(mydata, "nfreq"):
    for k in range(10):
        evalseq, shift = next(iter(myloader))
        evalseq = evalseq[:, :2].to(mynft.encoder.device)
        predicted = mymodel(evalseq, n_rolls=1)
        shifts.append(shift)
        Ms[0].append(mynft.dynamics.M)

    shifts = torch.concatenate(shifts)
    Ms[0] = torch.concatenate(Ms[0]).detach()

    plt.figure()

    matrixMeanshape = torch.mean(torch.abs(Ms[0].detach()).to("cpu"), axis=0)

    plt.imshow(matrixMeanshape)

    plt.savefig("./figures/debugMeanfig.jpg")

    myfreqs = np.array(mydata.freqsel)
    maxfreq = np.max(myfreqs)

    targfreq, prods0 = ca.inner_prod(
        Ms[0].to(shifts.device), shifts, maxfreq=maxfreq, bins=maxfreq + 1
    )

    plt.figure()
    deltas = ca.deltafxn(targfreq, mydata.freqsel) * 2
    plt.plot(
        targfreq,
        deltas,
        alpha=0.2,
        label="gt:" + str(np.where(deltas > 1.0)[0]),
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

    plt.savefig("./figures/debugfig.jpg")
