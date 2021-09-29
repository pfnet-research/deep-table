from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from deep_table.augmentation import ColumnSwap, RandomizedSwap, RowSwap
from deep_table.nn.encoders.encoder import Encoder
from deep_table.nn.layers.mlp import SimpleMLPLayer
from deep_table.nn.models.base import BaseModel


class DenoisingPretrainModel(BaseModel):
    """Denoising Auto Encoder model.

    References:
        1st place - turn your data into DAEta | Kaggle,
        <https://www.kaggle.com/springmanndaniel/1st-place-turn-your-data-into-daeta>
    """

    def __init__(
        self,
        encoder: Encoder,
        dim_feature_estimator: int = 256,
        prob_row: float = 0.2,
        prob_column: float = 0.2,
        prob_randomized: float = 0.2,
        **kwargs,
    ) -> None:
        """
        Args:
            encoder (`Encoder`): Encoder module used in training.
            dim_feature_estimator (int): Size of dimension in the hidden layer of MLP.
                The model predicts the original features using this MLP module.
                Defaults to 256.
            prob_row (float):
                Probability of applying row-wise swapping. The value is changed into the
                value in the same column but the other rows. Defaults to 0.2.
            prob_column (float):
                Probability of applying column-wise swapping. The value is changed into the
                value in the same row but the other columns. Defaults to 0.2.
            prob_randomized (float):
                Probability of applying randomized-swapping. The value is changed into
                the value in other rows and other columns. Defaults to 0.2.
        """
        self.save_hyperparameters(ignore="encoder")
        super(DenoisingPretrainModel, self).__init__(encoder, **kwargs)

    def _build_network(self) -> None:
        self.row_swap = RowSwap(self.hparams.prob_row)
        self.col_swap = ColumnSwap(self.hparams.prob_column)
        self.randomized_swap = RandomizedSwap(self.hparams.prob_randomized)

        self.num_categorical_features = self.encoder.num_categorical_features
        self.num_continuous_features = self.encoder.num_continuous_features
        self.num_categories = self.encoder.num_categories
        num_features = self.num_categorical_features + self.num_continuous_features
        dim_representation = self.encoder.dim_out(is_pretrain=True)

        layers = []
        for i in range(num_features):
            if i < self.num_continuous_features:
                layers.append(
                    SimpleMLPLayer(
                        in_dim=dim_representation,
                        out_dim=1,
                        hidden_dim=self.hparams.dim_feature_estimator,
                    )
                )
            else:
                layers.append(
                    SimpleMLPLayer(
                        in_dim=dim_representation,
                        out_dim=self.num_categories,
                        hidden_dim=self.hparams.dim_feature_estimator,
                    )
                )

        self.feature_estimators = nn.ModuleList(layers)

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x["continuous"] = self.randomized_swap(
            self.col_swap(self.row_swap(x["continuous"])[0])[0]
        )[0]
        x["categorical"] = self.row_swap(x["categorical"])[0]

        x = self.encoder(x, is_pretrain=True)
        x_reconstructed = [mlp(x) for mlp in self.feature_estimators]
        if self.num_continuous_features == 0:
            con_reconstructed = None
        else:
            con_reconstructed = torch.stack(
                x_reconstructed[: self.num_continuous_features], dim=1
            )
        if self.num_categorical_features == 0:
            cat_reconstructed = None
        else:
            cat_reconstructed = torch.stack(
                x_reconstructed[self.num_continuous_features :], dim=1
            )
        return {
            "con_reconstructed": con_reconstructed,
            "cat_reconstructed": cat_reconstructed,
        }

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        outputs = self(batch)
        loss = self.calculate_loss(batch, outputs, tag="train")
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        outputs = self(batch)
        _ = self.calculate_loss(batch, outputs, tag="val")

    def _setup_loss(self) -> None:
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def calculate_loss(
        self, batch: Dict[str, Tensor], outputs: Dict[str, Tensor], tag: str = "train"
    ) -> Tensor:
        loss = 0
        if outputs["con_reconstructed"] is not None:
            loss = loss + self.mse_loss(
                outputs["con_reconstructed"].squeeze(), batch["continuous"]
            )
        for i in range(self.num_categorical_features):
            loss = loss + self.cross_entropy_loss(
                outputs["cat_reconstructed"][:, i], batch["categorical"][:, i]
            )

        self.log(
            f"{tag}_loss",
            loss,
            on_epoch=(tag == "val"),
            on_step=(tag == "train"),
            logger=True,
            prog_bar=True,
        )
        return loss
