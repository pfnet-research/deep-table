import torch.nn as nn
from torch import Tensor

from deep_table.nn.encoders.backbone.base import BaseBackbone
from deep_table.nn.layers.mlp import DenseBlock


class MLPBackbone(BaseBackbone):
    def __init__(
        self,
        num_features: int,
        dim_embed: int,
        use_cls: bool = False,
        dim_max: int = 512,
        dim_out: int = 64,
        num_layers: int = 4,
        activation: str = "ReLU",
        dropout: float = 0.1,
        use_norm: bool = False,
    ) -> None:
        super(MLPBackbone, self).__init__()
        assert num_layers > 1
        self._dim_out = dim_out
        dim_input = num_features * dim_embed
        reduce_rate = (dim_max - dim_out) // (num_layers - 1)
        dim_hiddn_list = [dim_max - reduce_rate * i for i in range(num_layers - 1)]
        dim_hiddn_list.append(dim_out)

        dense_layers = []
        dense_layers.append(nn.Linear(dim_input, dim_hiddn_list[0]))
        for i in range(num_layers - 1):
            dense_layers.append(
                DenseBlock(
                    dim_hiddn_list[i],
                    dim_hiddn_list[i + 1],
                    dropout,
                    use_norm,
                    activation,
                )
            )
        self.dense_layers = nn.Sequential(*dense_layers)

    def dim_out(self, is_pretrain: bool = False) -> int:
        return self._dim_out

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        x = self.dense_layers(x)
        return x
