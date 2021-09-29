from typing import Optional

import torch.nn as nn
from torch import Tensor

from deep_table.nn.encoders.embedding.layer.base import BaseEmbeddingLayer
from deep_table.nn.layers.utils import get_activation_module


class ContinuousEmbedding(BaseEmbeddingLayer):
    def __init__(
        self,
        n_features: int,
        d_token: int,
        bias: bool,
        initialization: str = "uniform",
        activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(n_features, d_token))
        self.bias = nn.Parameter(Tensor(n_features, d_token)) if bias else None
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                self._apply_initialization(parameter, d_token, initialization)
        self.activation = get_activation_module(activation)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        x = self.activation(x)
        return x
