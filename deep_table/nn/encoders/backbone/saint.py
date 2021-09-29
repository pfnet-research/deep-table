from copy import deepcopy

import torch.nn as nn
from torch import Tensor

from deep_table.nn.encoders.backbone.base import BaseBackbone
from deep_table.nn.layers.transformer import ColRowTransformer


class SAINTBackbone(BaseBackbone):
    """Backbone model used in SAINT paper.

    References:
        G. Somepalli, M. Goldblum, A. Schwarzschild, C. B. Bruss and T. Goldstein,
        “SAINT: Improved Neural Networks for Tabular Data via Row Attention and
        Contrastive Pre-Training,” ArXiv:2106.01342 [cs.LG], 2021.
        <https://arxiv.org/abs/2106.01342>
    """

    def __init__(
        self,
        num_features: int,
        use_cls: bool = False,
        dim_embed: int = 16,
        n_layers: int = 6,
        n_heads: int = 8,
        dim_selfattn_head: int = 16,
        dim_intersample_head: int = 64,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        attn_type: str = "colrow",
    ) -> None:
        super().__init__()
        self.dim_embed = dim_embed
        self.num_features = num_features
        self.use_cls = use_cls
        saint_layer = ColRowTransformer(
            dim_embed,
            n_heads,
            dim_selfattn_head,
            dim_intersample_head,
            self.num_features,
            dim_feedforward,
            dropout,
            attn_type,
        )
        self.saint = nn.ModuleList([deepcopy(saint_layer) for _ in range(n_layers)])

    def dim_out(self, is_pretrain: bool = False) -> int:
        if not is_pretrain and self.use_cls:
            return self.dim_embed
        else:
            return self.num_features * self.dim_embed

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.saint:
            x = layer(x)
        return x
