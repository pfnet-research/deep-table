from typing import Optional

import torch.nn as nn
from torch import Tensor

from deep_table.nn.encoders.embedding.layer.base import BaseEmbeddingLayer
from deep_table.nn.layers.utils import get_activation_module


class CategoricalEmbedding(BaseEmbeddingLayer):
    def __init__(
        self,
        num_categories: int,
        d_token: int,
        bias: bool,
        num_categorical_features: Optional[int] = None,
        initialization: str = "uniform",
        activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert d_token > 0
        if bias:
            assert num_categorical_features is not None  # to add bias in feature-wise

        self.embeddings = nn.Embedding(num_categories, d_token)
        if bias:
            self.bias = nn.Parameter(Tensor(num_categorical_features, d_token))
        else:
            self.bias = None

        for parameter in [self.embeddings.weight, self.bias]:
            if parameter is not None:
                self._apply_initialization(parameter, d_token, initialization)
        self.activation = get_activation_module(activation)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.embeddings(x)
        if self.bias is not None:
            x = x + self.bias[None]
        x = self.activation(x)
        return x
