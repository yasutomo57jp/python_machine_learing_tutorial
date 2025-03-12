import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from dataset import MNISTDataModule
from networks import MNISTModel, MNISTModel2


class LitMNIST(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).item() / (len(y) * 1.0)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).item() / (len(y) * 1.0)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).item() / (len(y) * 1.0)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "test_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def on_epoch_end(self):
        optimizer = self.optimizers()
        self.log(
            "learning_rate", optimizer.param_groups[0]["lr"], on_epoch=True, logger=True
        )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False

    # Data
    data_module = MNISTDataModule()

    # Model
    # model = MNISTModel()
    model = MNISTModel2()

    # Loss
    loss_module = LitMNIST(model)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=5, logger=pl.loggers.WandbLogger(project="mnist_project")
    )
    trainer.fit(loss_module, data_module)

    # Save the model
    torch.save(model.state_dict(), "mnist_model.pth")
