import importlib

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from dataset import MNISTDataModule
from train import LitMNIST


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    torch.backends.cudnn.benchmark = False
    torch.random.manual_seed(cfg.trainer.random_seed)

    print("Config: ", cfg)

    # モデルのロード
    model = importlib.import_module(f"networks.{cfg.network.name}").get_model(cfg)
    model_file = f"{cfg.data.save_dir}/{cfg.network.name}.pth"
    model.load_state_dict(torch.load(model_file, weights_only=True))
    model.eval()

    # テストデータの準備
    data_module = MNISTDataModule(cfg.data.data_dir, cfg.trainer.batch_size)
    data_module.setup()
    test_loader = data_module.test_dataloader()

    # テスト用のLightningModule
    lit_model = LitMNIST(model)

    # テストの実行
    trainer = pl.Trainer(
        logger=pl.loggers.WandbLogger(
            project="mnist_project",
            name=f"{cfg.network.name}_{cfg.trainer.random_seed}",
        ),
    )
    trainer.test(lit_model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
