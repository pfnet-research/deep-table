from abc import ABCMeta, abstractmethod
from typing import Optional

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor


class BaseBackbone(nn.Module, metaclass=ABCMeta):
    """Base class for Backbone
    Parent class of each Backbone module.
    `dim_out` and `forward` functions need to be implemented.
    """

    @property
    @abstractmethod
    def dim_out(self, is_pretrain: bool = False) -> int:
        """Dimension of forward output.

        Args:
            is_pretrain (bool): When True, Backbone module is used in pretraining.
        """
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Pass the input through Backbone.

        Args:
            x (Tensor): the embedded 3D tensor (batch_size, num_features, dim_embed).

        Returns:
            Tensor: 2D or 3D tensor.
        """
        pass

    @classmethod
    def make(
        cls, num_features, dim_embed, use_cls, config: Optional[DictConfig] = None
    ) -> "BaseBackbone":
        if config is not None and config.get("args", False):
            return cls(
                num_features=num_features,
                dim_embed=dim_embed,
                use_cls=use_cls,
                **config.args,
            )
        else:
            return cls(
                num_features=num_features,
                dim_embed=dim_embed,
                use_cls=use_cls,
            )
