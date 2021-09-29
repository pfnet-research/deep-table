"""
Revisiting Deep Learning Models for Tabular Data
https://arxiv.org/abs/2106.11959
"""

import logging
from typing import Optional

import torch.nn as nn
from torch import Tensor

from deep_table.nn.encoders.backbone.base import BaseBackbone
from deep_table.nn.layers.transformer import TransformerEncoderLayer

logger = logging.getLogger(__name__)


class FTTransformerBackbone(BaseBackbone):
    def __init__(
        self,
        num_features: int,
        dim_embed: int,
        use_cls: bool = True,
        n_blocks: int = 3,
        n_heads: int = 4,
        dim_head: Optional[int] = None,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        """
        Args:
            num_features (int)
            dim_embed (int)
            use_cls (bool): Defaults to True.
            n_blocks (int): Defaults to 3.
            n_heads (int): Defaults to 4.
            dim_head (int, optional)
            dim_feedforward (int): Defaults to 256.
            dropout (float): Defaults to 0.1.
            activation (str): {"relu", "gelu"}. Defaults to "relu".
        """
        super().__init__()
        self.dim_embed = dim_embed
        self.num_features = num_features
        self.use_cls = use_cls
        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=dim_embed,
                    n_heads=n_heads,
                    dim_head=dim_head,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(n_blocks)
            ]
        )

    def dim_out(self, is_pretrain: bool = False) -> int:
        if not is_pretrain and self.use_cls:
            return self.dim_embed
        else:
            return self.num_features * self.dim_embed

    def forward(self, x: Tensor) -> Tensor:
        for transformer in self.transformer:
            x = transformer(x)
        return x
