"""
model.py - Simple PINN MLP for TL prediction.
"""

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


class PINN(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128, num_layers=6, out_dim=1,
                 activation="tanh", dropout=0.0):
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation, nn.Tanh)

        layers = [nn.Linear(in_dim, hidden_dim), act_cls()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, coords):
        return self.net(coords)
