from typing import List

import torch.nn as nn
from torch import Tensor

from .utils import get_activation_module


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        use_norm: bool = False,
        activation: str = "ReLU",
    ) -> None:
        """
        Args:
            in_dim (int): Size of each input sample.
            out_dim (int): Size of the desired output sample.
            dropout (float): Dropout value. Defaults to 0.0.
            use_norm (bool): If True, batch-normalization is applied before activation.
                Defaults to False.
            activation (str): Activation function. `activation` should be
                attributes of `torch.nn` or :func:`~deep_table.nn.layers.activation`.
                Defaults to "ReLU".
        """
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim) if use_norm else nn.Identity()
        self.fc = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_module(activation)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(self.dropout(self.activation(self.norm(x))))
        return x


class SimpleMLPLayer(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, hidden_dim: int, activation: str = "ReLU"
    ) -> None:
        """
        Args:
            in_dim (int): Size of each input sample.
            out_dim (int): Size of the desired output sample.
            hidden_dim (int): Size of the dimension of a hidden layer.
            activation (str): Activation function. `activation` should be
                attributes of `torch.nn` or :func:`~deep_table.nn.layers.activation`.
                Defaults to "ReLU".
        """
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.dense = DenseBlock(hidden_dim, out_dim, activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        return self.dense(x)


class MLP(nn.Module):
    def __init__(
        self,
        dims: List[int],
        activation: str = "ReLU",
    ) -> None:
        """
        Args:
            dims (list[int]): Sizes of each sample. The first value denotes size of input sample
                and the last one denotes size of output sample.
            activation (str): Activation function. `activation` should be
                attributes of `torch.nn` or :func:`~deep_table.nn.layers.activation`.
        """
        super().__init__()
        self.mlp = nn.ModuleList()
        dim_pairs = zip(dims[:-1], dims[1:])

        mlp = []
        for idx, (dim_in, dim_out) in enumerate(dim_pairs):
            is_last = idx == (len(dims) - 1)
            mlp.extend(
                [
                    nn.Linear(dim_in, dim_out),
                    get_activation_module(activation)()
                    if not is_last
                    else nn.Identity(),
                ]
            )
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        return x
