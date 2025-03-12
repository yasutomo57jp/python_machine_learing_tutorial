from torch import nn
from torch.nn import functional as F


class MNISTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layer_1 = nn.Linear(28 * 28, cfg.network.fc1.ch)
        self.layer_2 = nn.Linear(cfg.network.fc1.ch, cfg.network.fc2.ch)
        self.layer_3 = nn.Linear(cfg.network.fc2.ch, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return F.log_softmax(x, dim=1)


def get_model(cfg):
    return MNISTModel(cfg)
