import math

import torch.nn as nn
from torch import Tensor


class BaseEmbeddingLayer(nn.Module):
    def _apply_initialization(self, x: Tensor, d: int, method: str) -> None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if method == "uniform":
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif method == "normal":
            nn.init.normal_(x, std=d_sqrt_inv)
        else:
            raise ValueError(f"initialization: {method} is not implemented")
