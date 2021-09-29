from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from deep_table.nn.encoders.embedding.layer.append_cls import AppendCLSToken


class BaseEmbedding(nn.Module, metaclass=ABCMeta):
    """Base class for Embedding
    Each embedding module should inherit this class.
    """

    def __init__(
        self,
        num_continuous_features: int,
        num_categorical_features: int,
        num_categories: int,
        is_in_backbone_continuous: bool,
        is_in_backbone_categorical: bool,
        dim_in_backbone: Tuple[int],
        dim_skip_backbone: int,
        dim_embed: int,
        use_cls: bool = True,
    ) -> None:
        """
        Args:
            num_continuous_features (int): The number of continuous features (columns).
            num_categorical_features (int): The number of categorical features (columns).
            num_categories (int): Sum of categories in the dataset.
                When `num_categorical_features == 2` and each feature has 5 categories,
                `num_categories` should be 10.
            is_in_backbone_continuous (bool): When True, continuous features are
                fed into Backbone after embedded.
            is_in_backbone_categorical (bool): When True, categorical features are
                fed into Backbone after embedded.
            dim_in_backbone (tuple[int]): Dimension of input tensor for Backbone.
            dim_skip_backbone (int): Dimension of tensor which is not processed in `Backbone`.
            dim_embed (int): Dimension of embedding.
            use_cls (bool): When True, [CLS] token is added to the embedding.
                Defaults to True.
        """
        super().__init__()
        assert num_continuous_features or num_categorical_features
        self.use_cls = use_cls
        self.num_continuous_features = num_continuous_features
        self.num_categorical_features = num_categorical_features
        self.num_categories = num_categories
        self.dim_embed = dim_embed
        self.is_in_backbone = {
            "continuous": is_in_backbone_continuous,
            "categorical": is_in_backbone_categorical,
        }
        self.dim_in_backbone = dim_in_backbone
        self.dim_skip_backbone = dim_skip_backbone
        self._append_cls = AppendCLSToken(dim_embed) if use_cls else nn.Identity()

    @abstractmethod
    def continuous_embedding(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Features of continious columns. It has the shape
                (Batch size) * (Number of continuous columns).

        Returns:
            embedding (Tensor): Embedded tensor for continuous feature.
        """
        pass

    def _continuous_embedding(self, x: Optional[Tensor]) -> Optional[Tensor]:
        if x is None or x == []:
            return None
        else:
            return self.continuous_embedding(x)

    @abstractmethod
    def categorical_embedding(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Features of cateorical columns. It has the shape
                (Batch size) * (Number of categorical columns).

        Returns:
            embedding (Tensor): Embedded tensor for categorical feature
        """
        pass

    def _categorical_embedding(self, x: Optional[Tensor]) -> Optional[Tensor]:
        if x is None or x == []:
            return None
        else:
            return self.categorical_embedding(x)

    @classmethod
    def make(cls, config: Optional[DictConfig] = None, **kwargs):
        if config is not None and config.get("args", False):
            return cls(**config.args, **kwargs)
        else:
            return cls(dim_embed=16, **kwargs)

    def con_cat_embedding(self, x: Dict[str, Tensor]) -> Tensor:
        embed = {
            "continuous": self._continuous_embedding(x["continuous"]),
            "categorical": self._categorical_embedding(x["categorical"]),
        }
        in_backbone = []
        skip_backbone = []
        for key, value in self.is_in_backbone.items():
            if embed[key] is None:
                continue
            if value:
                in_backbone.append(embed[key])
            else:
                skip_backbone.append(embed[key])

        in_backbone = torch.cat(in_backbone, dim=1) if len(in_backbone) else None
        skip_backbone = torch.cat(skip_backbone, dim=1) if len(skip_backbone) else None
        embed = {"in_backbone": in_backbone, "skip_backbone": skip_backbone}
        return embed

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.con_cat_embedding(x)
        if x["in_backbone"] is not None:
            x["in_backbone"] = self.append_cls(x["in_backbone"])
        return x

    def append_cls(self, x: Optional[Tensor]) -> Optional[Tensor]:
        if x is None:
            return x
        else:
            return self._append_cls(x)
