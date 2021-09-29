import torch
import torch.nn as nn
from torch import Tensor

from deep_table.nn.encoders.embedding.layer.base import BaseEmbeddingLayer


class AppendCLSToken(BaseEmbeddingLayer):
    def __init__(self, d_token: int, initialization: str = "uniform") -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(d_token))
        self._apply_initialization(self.weight, d_token, initialization)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3
        return torch.cat([x, self.weight.view(1, 1, -1).repeat(len(x), 1, 1)], dim=1)
