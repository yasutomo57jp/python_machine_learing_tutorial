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
from networks import MNISTModel2
from train import LitMNIST


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    torch.backends.cudnn.benchmark = False
    torch.random.manual_seed(cfg.trainer.random_seed)

    # モデルのロード
    model = MNISTModel2(cfg)
    model.load_state_dict(torch.load(cfg.networks.save_path, weights_only=True))
    model.eval()

    # テストデータの準備
    data_module = MNISTDataModule(cfg.data_dir, cfg.trainer.batch_size)
    data_module.setup()
    test_loader = data_module.test_dataloader()

    # テスト用のLightningModule
    lit_model = LitMNIST(model)

    # テストの実行
    trainer = pl.Trainer(logger=pl.loggers.WandbLogger(project="mnist_project"))
    trainer.test(lit_model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
