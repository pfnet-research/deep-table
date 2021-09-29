"""
TabTransformer: Tabular Data Modeling Using Contextual Embeddings
https://arxiv.org/abs/2012.06678
"""

from typing import Optional

import torch.nn as nn
from torch import Tensor

from deep_table.nn.encoders.embedding.base import BaseEmbedding
from deep_table.nn.encoders.embedding.layer.categorical_embedding import (
    CategoricalEmbedding,
)


class TabTransformerEmbedding(BaseEmbedding):
    """Embedding method used in TabTransformer.

    The categorical features are embedded by `CategoricalEmbedding`.
    On the other hand, the continuous features are converted by layer normalization.
    """

    def __init__(
        self,
        num_continuous_features: int,
        num_categorical_features: int,
        num_categories: int,
        dim_embed: int,
        use_cls: bool = True,
        initialization: str = "uniform",
        activation: Optional[str] = None,
    ) -> None:
        super().__init__(
            num_continuous_features=num_continuous_features,
            num_categorical_features=num_categorical_features,
            num_categories=num_categories,
            dim_embed=dim_embed,
            is_in_backbone_continuous=False,
            is_in_backbone_categorical=True,
            dim_in_backbone=(num_categorical_features + int(use_cls), dim_embed),
            dim_skip_backbone=num_continuous_features,
            use_cls=use_cls,
        )
        self.con_embedding = nn.LayerNorm(num_continuous_features)
        self.cat_embedding = (
            CategoricalEmbedding(
                num_categorical_features=num_categorical_features,
                num_categories=num_categories,
                d_token=dim_embed,
                bias=True,
                initialization=initialization,
                activation=activation,
            )
            if num_categorical_features
            else None
        )

    def continuous_embedding(self, x: Tensor) -> Tensor:
        return self.con_embedding(x)

    def categorical_embedding(self, x: Tensor) -> Tensor:
        return self.cat_embedding(x)
