import sparsechem as sc
import torch
from torch import nn

class Trunk(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.trunk = sc.SparseInputNet(conf)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) in [sc.SparseLinear]:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, X):
        return self.trunk(X)


class TrunkAndHead(torch.nn.Module):
    def __init__(self, trunk, conf):
        super().__init__()

        self.trunk = trunk

        self.head = nn.Sequential(
            sc.MiddleNet(conf),
            sc.LastNet(conf),
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) in [nn.Linear, sc.SparseLinear]:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, X):
        X = self.trunk(X)
        return self.head(X)
