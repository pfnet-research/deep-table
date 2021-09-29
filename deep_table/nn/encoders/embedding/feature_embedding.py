from typing import Optional

from torch import Tensor

from deep_table.nn.encoders.embedding.base import BaseEmbedding
from deep_table.nn.encoders.embedding.layer.categorical_embedding import (
    CategoricalEmbedding,
)
from deep_table.nn.encoders.embedding.layer.continuous_embedding import (
    ContinuousEmbedding,
)


class FeatureEmbedding(BaseEmbedding):
    """All features (continuous and catecorigal) are embedded by `ContinousEmbedding`
    and `CategoricalEmbedding`.
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
            is_in_backbone_continuous=True,
            is_in_backbone_categorical=True,
            dim_in_backbone=(
                num_categorical_features + num_continuous_features + int(use_cls),
                dim_embed,
            ),
            dim_skip_backbone=0,
            use_cls=use_cls,
        )
        self.con_embedding = ContinuousEmbedding(
            n_features=num_continuous_features,
            d_token=dim_embed,
            bias=True,
            initialization=initialization,
            activation=activation,
        )

        self.cat_embedding = CategoricalEmbedding(
            num_categorical_features=num_categorical_features,
            num_categories=num_categories,
            d_token=dim_embed,
            bias=True,
            initialization=initialization,
            activation=activation,
        )

    def continuous_embedding(self, x: Tensor) -> Tensor:
        return self.con_embedding(x)

    def categorical_embedding(self, x: Tensor) -> Tensor:
        return self.cat_embedding(x)
