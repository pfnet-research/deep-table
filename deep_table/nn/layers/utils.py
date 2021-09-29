from logging import getLogger
from typing import Callable, Optional

import torch.nn as nn
import torch.nn.functional as F

from . import activation as A

logger = getLogger(__name__)


def get_activation_module(activation: Optional[str] = None) -> nn.Module:
    if activation is None:
        logger.warning("`activation` is not set. `nn.Identity` would be used instead.")
        return nn.Identity

    elif hasattr(A, activation):
        return getattr(A, activation)

    elif hasattr(nn, activation):
        return getattr(nn, activation)

    else:
        raise ValueError(
            f"{activation} must be implemented in torch.nn or"
            ":func:`~deep_table.nn.layers.activation`."
        )


def get_activation_fn(activation: str = "relu") -> Callable:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
