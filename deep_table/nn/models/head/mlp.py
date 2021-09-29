from typing import Dict

import torch.nn as nn
from torch import Tensor

from deep_table.nn.encoders.encoder import Encoder
from deep_table.nn.models.base import BaseModel


class MLPHeadModel(BaseModel):
    """MLP head model for training or fine-tuning"""

    def __init__(
        self,
        encoder: Encoder,
        dim_out: int,
        dim_hidden: int = 256,
        **kwargs,
    ) -> None:
        """
        Args:
            encoder (`Encoder`): Encoder used in training. It contains `Embedding` and `Backbone`.
            dim_out (int): Output size. If task is binary classification, `dim_out` should be 1.
            dim_hidden (int): Dimension of the intermediate layer. Defaults to 256.
        """
        self.save_hyperparameters(ignore="encoder")
        super(MLPHeadModel, self).__init__(encoder, **kwargs)

    def _build_network(self) -> None:
        dim_representation = self.encoder.dim_out(is_pretrain=False)

        self.mlp = nn.Sequential(
            nn.Linear(dim_representation, self.hparams.dim_hidden),
            nn.ReLU(),
            nn.Linear(self.hparams.dim_hidden, self.hparams.dim_out),
        )

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        x = self.encoder(x)
        x = self.mlp(x)
        return x
