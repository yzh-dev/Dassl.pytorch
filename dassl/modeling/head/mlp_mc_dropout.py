import functools
import torch.nn as nn
from torch import Tensor

import torch.nn.functional as F


from .build import HEAD_REGISTRY


class MCDropout(nn.Dropout):

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, True, self.inplace)


class MLP(nn.Module):

    def __init__(
        self,
        in_features=2048,
        hidden_layers=[],
        activation='relu',
        bn=True,
        dropout=0.
    ):
        super().__init__()
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]

        assert len(hidden_layers) > 0
        self.out_features = hidden_layers[-1]

        mlp = []

        if activation == 'relu':
            act_fn = functools.partial(nn.ReLU, inplace=True)
        elif activation == 'leaky_relu':
            act_fn = functools.partial(nn.LeakyReLU, inplace=True)
        else:
            raise NotImplementedError

        for hidden_dim in hidden_layers:
            mlp += [nn.Linear(in_features, hidden_dim)]
            if bn:
                mlp += [nn.BatchNorm1d(hidden_dim)]
            mlp += [act_fn()]
            if dropout > 0:
                mlp += [MCDropout(dropout)]
            in_features = hidden_dim

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)


@HEAD_REGISTRY.register()
def mlp_mc_dropout(**kwargs):
    return MLP(**kwargs)
