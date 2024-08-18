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
# from module import ft_module_from_dimside as ftdim


def main():
    # modename
    modelname = "fordebug"
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
    np.random.seed(42)
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
        torch.manual_seed(0)
        np.random.seed(42)

        data_args = self.configs["data"]
        # self.data = SequentialMNIST_double(**data_args)
        self.data = yu.load_component(data_args)
        print(f"""Using the datatype {self.data} """)
        eval_data_args = copy.deepcopy(data_args)
        eval_data_args["args"]["T"] = self.evalT
        self.eval_data = yu.load_component(eval_data_args)

    def _set_loader(self):
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
        if "nftmodel" in self.configs["model"].keys():
            DFNFTclass = yu.load_component_fxn(self.configs["model"]["nftmodel"])

        else:
            DFNFTclass = ftd.DFNFT

        if "reload" in self.configs["model"].keys():
            expname = self.configs["model"]["reload"]
            exppath = os.path.join("./dnftresult", expname)
            if not os.path.exists(exppath):
                raise NotImplementedError
            mymodelpath = f"""{exppath}/model.pt"""
            self.nftmodel = torch.load(mymodelpath)
            print(f"""Begin restarting from the model of {mymodelpath}. """)
            print("WARNING : We ARE RELOADING A MODEL" * 10)

        else:
            cfg_model = self.configs["model"]

            enc_class = yu.load_component_fxn(cfg_model["encmodel"])
            dec_class = yu.load_component_fxn(cfg_model["decmodel"])

            model_args = cfg_model["modelargs"]
            nft_args = cfg_model["nftargs"]

            if "num_register_tokens" in model_args.keys():
                matsize = model_args["num_register_tokens"]
            else:
                matsize = (
                    model_args["dim_a"]
                    if nft_args["is_Dimside"] == True
                    else model_args["dim_m"]
                )

            cfglist = self.create_configs(nft_args["depth"], model_args)
            dec_cfglist = copy.deepcopy(cfglist)
            if "," in str(model_args["depth"]):
                depthlist = model_args["depth"].split(",")
                for k in range(len(cfglist)):
                    cfglist[k]["depth"] = int(depthlist[k])
                    dec_cfglist[k]["depth"] = int(depthlist[k])

            if {"dim_m", "dim_a"}.issubset(model_args):
                for k in range(1, len(cfglist)):
                    cfglist[k]["dim_data"] = (
                        cfglist[k - 1]["dim_m"] * cfglist[k - 1]["dim_a"]
                    )
                    dec_cfglist[k]["dim_data"] = (
                        cfglist[k - 1]["dim_m"] * cfglist[k - 1]["dim_a"]
                    )
            nft_list = []
            owndec_list = []

            for k in range(nft_args["depth"]):
                enc1 = enc_class(**cfglist[k])
                dec1 = dec_class(**dec_cfglist[k])
                dec1a = dec_class(**dec_cfglist[k])
                nft1 = ftd.NFT(encoder=enc1, decoder=dec1, **nft_args)
                nft_list.append(nft1)
                owndec_list.append(dec1a)

            # self.nftmodel = DFNFTclass(nftlist=nft_list, owndecoders=owndec_list)
        self.nftmodel = nft_list[0]
        self.writerlocation = f"""./dnftresult/{self.configs['expname']}"""
        self.configs["data"]["args"]["T"] = self.trainT

        print(f"""Work will be saved at {self.writerlocation}""")

    def train(self):
        self.nftmodel.train().to(dtype=self.dtype).to(self.device)
        for iteridx in tqdm(range(self.itermax)):
            self.iter = iteridx
            seqs = next(iter(self.loader)).to(dtype=self.dtype).to(self.device)
            # seqs:   N T instance_shape
            n, t = seqs.shape[:2]
            trainT = self.trainT
            trainseqs = seqs[:, :trainT]

            rollnum = trainT - 1
            self.optimizer.zero_grad()
            loss = self.nftmodel.loss(trainseqs, n_rolls=rollnum)

            loss["all_loss"].backward()
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
        evalseq = (self.eval_data[1].to(dtype=self.dtype))[None, :]
        self.nftmodel.evaluate(evalseq, self.writer, device=self.device)
        self.nftmodel = self.nftmodel.train()


if __name__ == "__main__":
    main()
