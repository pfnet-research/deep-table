from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from deep_table.augmentation.utils import generate_mask, masking


class Cutmix(nn.Module):
    """Cutmix module for augmentation

    Examples:
        >>> input = torch.tensor([
                [1, 2, 3],
                [4, 5, 6]
            ])
        >>> Cutmix(mask_prob=0.5)(input)
        (tensor([[1, 5, 6], [1, 2, 6]]),)
    """

    def __init__(self, mask_prob: float = 0.1):
        """
        Args:
            mask_prob (float): Probability for masking.
                Defaults to 0.1.
        """
        super().__init__()
        self.mask_prob = mask_prob

    def forward(self, *inputs: Tensor) -> Tuple[Tensor]:
        batch_size = inputs[0].size(0)
        device = inputs[0].device

        shuffled_idx = torch.randint(0, batch_size, (batch_size,))
        outputs = tuple(
            masking(
                x,
                x[shuffled_idx],
                generate_mask(batch_size, x.size(1), mask_prob=self.mask_prob).to(
                    device
                ),
            )
            for x in inputs
        )
        return outputs
