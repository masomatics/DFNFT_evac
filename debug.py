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
import pdb
import copy


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


datname = "OneDsignal_c8mimic"
# datname = 'OneDCyclic'
modelname = "fordebug"
# modelname = 'mask2Stacklayer'
# modelname = 'mlp1layer_nonDim'
# modelname = 'mask1layer'

trainname = "baseline"
with open(f"""./cfg_data/{datname}.yaml""", "rb") as f:
    cfg_data = yaml.safe_load(f)

cfg_data["args"]["shift_label"] = True
mydata = yu.load_component(cfg_data)

myloader = DataLoader(
    mydata,
    batch_size=20,
    shuffle=True,
    num_workers=1,
)


seq, shift = mydata[0]
seq = seq[None, :]
print(seq.shape)

expname = f"""{datname}_{modelname}_{trainname}"""

exppath = os.path.join("./dnftresult", expname)
if not os.path.exists(exppath):
    raise NotImplementedError
mymodelpath = f"""{exppath}/model.pt"""
mymodel = torch.load(mymodelpath)
mymodel = mymodel.to(0)


import time

# Get the current time
current_time = time.localtime()

# Format the time to only include the date (Year-Month-Day)
current_date = time.strftime("%Y-%m-%d-%H:%M", current_time)

Ms = {0: [], 1: []}
shifts = []
for k in range(100):
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
plt.plot(targfreq, deltas, alpha=0.2, label="gt:" + str(np.where(deltas > 1.0)[0]))
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

plt.savefig("./Notebooks/debug0.jpg")

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

plt.savefig("./Notebooks/debugUpBot.jpg")
