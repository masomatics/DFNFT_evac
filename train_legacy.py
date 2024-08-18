import os
import sys

sys.path.append("../")
import torch
from torch.utils.data import DataLoader
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy
import numpy as np

from module import ft_decimation as ftd
import pdb
from misc import yaml_util as yu


def main():
    modelname = "fordebug"
    datname = "OneDsignal"
    trainname = "baseline"
    mode = "_".join([datname, modelname, trainname])

    with open(f"""./cfg_data/{datname}.yaml""", "rb") as f:
        cfg_data = yaml.safe_load(f)
    with open(f"""./cfg_train/{trainname}.yaml""", "rb") as f:
        cfg_train = yaml.safe_load(f)

    configs = {}
    configs["train"] = cfg_train
    configs["data"] = cfg_data
    configs["expname"] = mode

    trainer = Trainer(configs)
    trainer.train()


class Trainer(object):
    def __init__(self, configs):
        self.configs = configs
        self.dtype = torch.float64

        self.writerlocation = "./dnftresult/vanillaX"
        self._set_train()
        self._set_model()
        self._set_data()
        self._set_loader()
        self._set_optimizer()
        self.writer = SummaryWriter(self.writerlocation)

        print(f"""Using device {self.device}""")
        configs["train"]["device"] = 1

        trainer = Trainer(configs)
        trainer.train()

    def _set_train(self):
        for attr in self.configs["train"].keys():
            setattr(self, attr, self.configs["train"][attr])

    def _set_data(self):
        torch.manual_seed(0)
        np.random.seed(42)

        data_args = self.configs["data"]
        # self.data = SequentialMNIST_double(**data_args)
        self.data = yu.load_component(data_args)
        print(f"""Using the datatype {self.data} """)
        eval_data_args = copy.deepcopy(data_args)
        eval_data_args["args"]["T"] = self.evalT
        self.eval_data = yu.load_component(eval_data_args)

    def _set_model(self):
        from module import legacy_seqae as LS

        self.nftmodel = LS.SeqAETSQmlp(
            dim_data=128, dim_m=16, dim_a=10, predictive=True
        )

    def train(self):
        self.nftmodel.train().to(dtype=self.dtype).to(self.device)
        self.itermax = 10000
        for iteridx in tqdm(range(self.itermax)):
            self.iter = iteridx
            seqs = next(iter(self.loader)).to(dtype=self.dtype).to(self.device)
            # seqs:   N T instance_shape
            n, t = seqs.shape[:2]
            trainT = self.trainT
            trainseqs = seqs[:, :trainT]
            loss, loss_dict = self.nftmodel.loss(
                seqs,
                T_cond=2,
                return_reg_loss=True,
                reconst=True,
            )


if __name__ == "__main__":
    main()
