import os
import sys
import random
from itertools import cycle

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
# from module import ft_module_from_dimside as ftdim


def main():
    # modename
    modelname = "mask1layer"
    # modelname = "fordebug"
    datname = "OneDsignal"
    trainname = "baseline"
    mode = "_".join([datname, modelname, trainname])

    with open(f"""./cfg_model/{modelname}.yaml""", "rb") as f:
        cfg_model = yaml.safe_load(f)
    with open(f"""./cfg_data/{datname}.yaml""", "rb") as f:
        cfg_data = yaml.safe_load(f)
    with open(f"""./cfg_train/{trainname}.yaml""", "rb") as f:
        cfg_train = yaml.safe_load(f)

    myseed = cfg_train["seed"]

    torch.manual_seed(myseed)
    configs = {}
    configs["train"] = cfg_train
    configs["model"] = cfg_model
    configs["data"] = cfg_data
    configs["expname"] = mode

    configs["train"]["device"] = 6

    trainer = DF_Trainer(configs)
    trainer.train()


class DF_Trainer(object):
    def __init__(self, configs):
        seed = 1
        self.seed = seed
        torch.cuda.empty_cache()
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.configs = configs
        self.dtype = torch.float64

        self.writerlocation = "./dnftresult/vanillaX"
        self._set_train()
        self._set_models()
        self._set_data()
        self._set_loader()
        self._set_optimizer()
        self.writer = SummaryWriter(self.writerlocation)

        print(f"""Using device {self.device}""")

    def _set_train(self):
        for attr in self.configs["train"].keys():
            setattr(self, attr, self.configs["train"][attr])

    def _set_data(self):
        data_args = self.configs["data"]
        # self.data = SequentialMNIST_double(**data_args)
        self.data = yu.load_component(data_args)
        print(f"""Using the datatype {self.data} """)
        eval_data_args = copy.deepcopy(data_args)
        eval_data_args["args"]["T"] = self.evalT
        self.eval_data = yu.load_component(eval_data_args)

    def _set_loader(self):
        torch.manual_seed(self.seed)

        self.loader = DataLoader(
            self.data,
            batch_size=self.configs["train"]["batchsize"],
            shuffle=True,
            num_workers=2,
        )

    def _set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.nftmodel.parameters(), lr=self.lr)

    def report(self, value=None, name=None):
        if self.iter % self.report_freq == 0:
            self.writer.add_scalar(name, value, self.iter)

    def create_configs(self, depth, base_args):
        cfglist = []
        for k in range(depth):
            encdec_cfg = []
            common_cfg = copy.deepcopy(base_args)
            if k == 0:
                common_cfg["require_input_adapter"] = True
            else:
                common_cfg["require_input_adapter"] = False

            cfglist.append(common_cfg)
        return cfglist

    def _set_models(self):
        cfg_model = self.configs["model"]
        enc_class = yu.load_component_fxn(cfg_model["encmodel"])
        dec_class = yu.load_component_fxn(cfg_model["decmodel"])

        model_args = cfg_model["modelargs"]
        nft_args = cfg_model["nftargs"]
        mask = self.create_masks(model_args)
        enc1 = enc_class(**model_args, maskmat=mask)
        dec1 = dec_class(**model_args, maskmat=mask)
        self.nftmodel = ftd.NFT(
            encoder=[enc1], decoder=[dec1], require_input_adapter=True, **nft_args
        )
        self.writerlocation = f"""./dnftresult/{self.configs['expname']}"""
        self.configs["data"]["args"]["T"] = self.trainT

        print(f"""Work will be saved at {self.writerlocation}""")

    def create_masks(self, model_args, layer=0):
        matsize = model_args["dim_m"]
        if layer == 0:
            mask = torch.ones(matsize, matsize, requires_grad=False)
        else:
            blocksize = matsize // 2
            block = torch.ones(blocksize, blocksize, requires_grad=False)
            mask = torch.block_diag(block, block)
        return mask

    def train(self):
        self.nftmodel.train().to(dtype=self.dtype).to(self.device)
        loader_iter = cycle(self.loader)
        for iteridx in tqdm(range(self.itermax)):
            self.iter = iteridx
            [seqs, label] = next(loader_iter)
            seqs = seqs.to(dtype=self.dtype).to(self.device)
            # print(f"""Debug label is {label}""")
            # print(seqs[0, 0, :10])
            # print(seqs[0, 1, :10])

            # print(self.nftmodel.encoder.enc.lin1.bias[:10])
            # seqs:   N T instance_shape
            n, t = seqs.shape[:2]
            trainT = self.trainT
            trainseqs = seqs[:, :trainT]

            rollnum = trainT - 1
            self.optimizer.zero_grad()
            loss = self.nftmodel.loss(trainseqs, n_rolls=rollnum)

            loss["predloss"].backward()
            self.optimizer.step()

            all_loss = loss["all_loss"].item()
            intermediate = loss["intermediate"].item()
            predloss = loss["predloss"].item()

            loss_metrics = {
                "all_loss": all_loss,
                "predloss": predloss,
                "intermediate": intermediate,
            }

            for key in loss_metrics.keys():
                self.report(value=loss_metrics[key], name=key)

            if self.iter % self.save_freq == 0:
                print(loss_metrics["all_loss"])
                try:
                    torch.save(self.nftmodel, f"""{self.writerlocation}/model.pt""")
                except:
                    print("torch.nn.utils.parametrize is probably invoked.")
                    torch.save(
                        self.nftmodel.state_dict(), f"{self.writerlocation}/model.pt"
                    )
                print(f"""Model saved at {self.writerlocation}/model.pt""")
                self.evaluate()
                print(f"""Model evaluated""")

    def evaluate(self):
        self.nftmodel = self.nftmodel.eval()
        evalseq, label = self.eval_data[1]
        evalseq = (evalseq.to(dtype=self.dtype))[None, :]
        self.nftmodel.evaluate(evalseq, self.writer, device=self.device)
        self.nftmodel = self.nftmodel.train()


if __name__ == "__main__":
    main()
