import torch
from torch import Tensor


def generate_mask(*size: int, mask_prob: float) -> Tensor:
    """
    Args:
        *size (int): Shape of the expected values.
        mask_prob (float): The probability of masking.

    Returns:
        Tensor: If True, these features should be masked.
            Otherwise, these features should be not changed.

    Examples:
        >>> generate_mask(4, mask_prob=0.5)
        tensor([0, 1, 1, 0])
    """
    mask = (torch.rand(*size) > mask_prob).long()
    return mask


def masking(x: Tensor, x_mask: Tensor, mask: Tensor) -> Tensor:
    """
    Args:
        x (Tensor):
        x_mask (Tensor):
        mask (Tensor):

    Returns:
        Tensor: Masked deatures.
        .. math:: x * mask + x_mask * (1 - mask).

    Examples:
        >>> import torch
        >>> x = torch.tensor([[1, 2, 3]])
        >>> x_mask = torch.tensor([[0, 0, 0]])
        >>> mask = torch.tensor([[0.5, 0.5, 0.5]])
        >>> masking(x, x_mask, mask)
        tensor([[0.5000, 1.0000, 1.5000]])
    """
    if x.dim() == 3:
        mask = mask.unsqueeze(-1)
    elif x.dim() > 3:
        raise ValueError(f"{x.dim()}D tensor is invalid for masking")
    return x * mask + x_mask * (1 - mask)
