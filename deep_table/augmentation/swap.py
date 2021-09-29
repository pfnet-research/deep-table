from abc import abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from deep_table.augmentation.utils import generate_mask, masking


class BaseSwap(nn.Module):
    def __init__(
        self,
        prob: float = 0.2,
        overlap: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        Args:
            prob (float): The probability for swapping. If `prob` is 0.2,
                20 percents of values are swapped.
            overlap (bool): If True, the same element can appear multiple times.
                Otherwise, one element can appear only once at most.
            generator (torch.Generator, optional): Generator for random modules.
                If set, you can produce reproductive result. Defaults to None.
        """
        super().__init__()
        self.prob = prob
        self.overlap = overlap
        self.generator = generator

    def random_idx(self, size: int) -> Tensor:
        if self.overlap:
            return torch.randint(0, size, (size,), generator=self.generator)
        else:
            return torch.randperm(size, generator=self.generator)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Simple wrapper for `_forward`.
        If `x` is None or [], return tuple of None.
        Otherwise, `x` is converted in `_forward`.
        """
        if x is None or x == []:
            return None, None
        else:
            return self._forward(x)

    @abstractmethod
    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): Input tensor for swapping.

        Returns:
            x (Tensor): Swapped tensor.
            mask (Tensor): Mask tensor. Each value is torch.int64 and
                the value is 1 for swapped and 0 for not swapped.
        """
        pass


class RowSwap(BaseSwap):
    """Swapping with values in other rows.

    For swapping, the value is changed into the value in the
    same column but the other rows.
    """

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = x.size(0)
        num_features = x.size(1)
        device = x.device

        x_shuffled = x[self.random_idx(x.size(0))]

        mask = generate_mask(batch_size, num_features, mask_prob=self.prob).to(device)
        x = masking(x, x_shuffled, mask)
        return x, mask


class ColumnSwap(BaseSwap):
    """Swapping with values in other columns.

    For swapping, the value is changed into the value in the
    same row but the other columns.
    """

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        num_features = x.size(1)
        device = x.device

        x_shuffled = x.view(-1)[self.random_idx(x.view(-1).size(0))].view(x.size())

        mask = (
            generate_mask(num_features, mask_prob=self.prob)
            .to(device)
            .unsqueeze(0)
            .expand(x.size())
        )
        x = masking(x, x_shuffled, mask)
        return x, mask


class RandomizedSwap(BaseSwap):
    """Swapping with values in other rows and other columns.

    The value is changed into the value of the ramdom sample in
    other rows and other columns.
    """

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = x.size(0)
        num_features = x.size(1)
        device = x.device

        x_shuffled = x.view(-1)[self.random_idx(x.view(-1).size(0))].view(x.size())

        mask = generate_mask(batch_size, num_features, mask_prob=self.prob).to(device)
        x = masking(x, x_shuffled, mask)
        return x, mask


class EmpiricalSwap(BaseSwap):
    """Empirical swapping

    Returns:
        tuple[Tensor, Tensoe]: The first value is swapped value and
            the second value is mask which contains 1 or 0.

    Examples:
    >>> inpt = torch.tensor([
    >>>     [1, 2, 3],
    >>>     [4, 5, 6]
    >>> ])
    >>> EmpiricalSwap(prob=0.2)(inpt)
    (tensor([[1, 5, 6], [4, 5, 6]]),
     tensor([[1, 0, 0], [1, 1, 0]]))
    """

    def __init__(
        self, prob: float = 0.2, generator: Optional[torch.Generator] = None
    ) -> None:
        """
        Args:
            prob (float): The probability for swapping. If `prob` is 0.2,
                20 percents of values are swapped.
            generator (torch.Generator, optional): Generator for random modules.
                If set, you can produce reproductive result. Defaults to None.
        """
        super(EmpiricalSwap, self).__init__(prob, overlap=True)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = x.size(0)
        num_features = x.size(1)
        device = x.device

        x_shuffled = []
        for i in range(num_features):
            x_shuffled.append(x[self.random_idx(batch_size), i])
        x_shuffled = torch.stack(x_shuffled, 1)

        mask = generate_mask(batch_size, num_features, mask_prob=self.prob).to(device)
        x = masking(x, x_shuffled, mask)
        return x, mask
