from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


def mixup(x: Tensor, x_b: Tensor, alpha: float) -> Tensor:
    x = x * alpha + x_b * (1 - alpha)
    return x


class Mixup(nn.Module):
    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.alpha = alpha

    def forward(self, *inputs: Tensor) -> Tuple[Tensor]:
        batch_size = inputs[0].size(0)
        shuffled_idx = torch.randint(0, batch_size, (batch_size,))
        outputs = tuple(mixup(x, x[shuffled_idx], self.alpha) for x in inputs)
        return outputs
