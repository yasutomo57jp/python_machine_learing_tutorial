# Pytorch Lightningの利用

MNISTでのpytorch lightningを使った学習コードのサンプルです．


## train.py

LightningModuleを継承したクラスを定義し，そのクラスを使って学習を行います．
```python
class LitMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
```

学習に使うデータセットを定義します．．
```python
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./", batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

    def setup(self, stage=None):
        dataset = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)
```

プログラムのメイン部分です．
```python
def main():
    torch.backends.cudnn.benchmark = False

    # データセットの読み込み
    data_module = MNISTDataModule()

    # モデルの定義
    model = LitMNIST()

    # 学習（ここでは5エポックだけ学習）
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model, data_module)

    # 学習済みモデルの保存
    torch.save(model.state_dict(), "mnist_model.pth")


if __name__ == "__main__":
    main()
```

## train2.py


学習の仕方と，モデルの定義を分離した例です．
```python
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return F.log_softmax(x, dim=1)
```

```python
class LitMNIST(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        以下同様
```

mainでは，モデルをインスタンス化したうえでlightning moduleへ渡しています．
```python
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False

    # Data
    data_module = MNISTDataModule()

    # Model
    model = MNISTModel()

    # Loss
    loss_module = LitMNIST(model)
```
