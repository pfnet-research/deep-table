from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from deep_table.augmentation import EmpiricalSwap
from deep_table.nn.encoders import Encoder
from deep_table.nn.layers.mlp import SimpleMLPLayer
from deep_table.nn.models.base import BaseModel


class VIMEPretrainModel(BaseModel):
    """VIME pretrain model

    References:
        R. Houthooft, X. Chen, Y. Duan, J. Schulman, F. De Turck and P. Abbeel,
        ”VIME: Variational Information Maximizing Exploration,” NeurIPS 2016,
        vol. 29, pp. 1109-1117, 2016. <https://arxiv.org/abs/1605.09674>
    """

    def __init__(
        self,
        encoder: Encoder,
        dim_feature_estimator: int = 256,
        dim_mask_estimator: int = 256,
        mask_prob: float = 0.1,
        alpha: float = 2.0,
        **kwargs,
    ) -> None:
        """
        Args:
            encoder (`Encoder`): Encoder module used in training.
            dim_feature_estimator (int): Size of dimension in the hidden layer of MLP.
                The model predicts the original features using this MLP module.
            dim_mask_estimator (int): Size of dimension in the hidden layer of MLP.
                The model predicts where masked using this MLP module.
            mask_prob (float): Probability of masking features. Each masked feature is
                swapped to the value of another row. For pretraining, this model is
                predicting which values are swapped. Defaults to 0.3.
            alpha (float): Importance of `reconstruction_loss`. The loss is calculated by
                loss = `mask_predict_loss` + `alpha` * `reconstruction_loss`
        """

        self.save_hyperparameters(ignore="encoder")
        super(VIMEPretrainModel, self).__init__(encoder, **kwargs)

    def _build_network(self) -> None:
        self.swap_noise = EmpiricalSwap(self.hparams.mask_prob)

        self.num_categorical_features = self.encoder.num_categorical_features
        self.num_continuous_features = self.encoder.num_continuous_features
        self.num_categories = self.encoder.num_categories
        num_features = self.num_categorical_features + self.num_continuous_features
        dim_representation = self.encoder.dim_out(is_pretrain=True)

        self.mask_estimator = SimpleMLPLayer(
            in_dim=dim_representation,
            out_dim=num_features,
            hidden_dim=self.hparams.dim_mask_estimator,
        )

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
        x["continuous"], mask_con = self.swap_noise(x["continuous"])
        x["categorical"], mask_cat = self.swap_noise(x["categorical"])
        mask = []
        for m in [mask_con, mask_cat]:
            if m is not None:
                mask.append(m)
        mask = torch.cat(mask, dim=1)
        x = self.encoder(x, is_pretrain=True)

        mask_hat = self.mask_estimator(x)
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
            "mask": mask,
            "mask_hat": mask_hat,
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
        self.binary_cross_entropy_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def calculate_loss(
        self, batch: Dict[str, Tensor], outputs: Dict[str, Tensor], tag: str = "train"
    ) -> Tensor:
        mask_predict_loss = self.binary_cross_entropy_loss(
            outputs["mask_hat"], outputs["mask"].float()
        )
        reconstruction_loss = 0
        if outputs["con_reconstructed"] is not None:
            reconstruction_loss = reconstruction_loss + self.mse_loss(
                outputs["con_reconstructed"].squeeze(), batch["continuous"]
            )
        for i in range(self.num_categorical_features):
            reconstruction_loss = reconstruction_loss + self.cross_entropy_loss(
                outputs["cat_reconstructed"][:, i], batch["categorical"][:, i]
            )
        loss = mask_predict_loss + self.hparams.alpha * reconstruction_loss

        self.log(
            f"{tag}_mask_predict_loss",
            mask_predict_loss,
            on_epoch=(tag == "val"),
            on_step=(tag == "train"),
            logger=True,
            prog_bar=True,
        )
        self.log(
            f"{tag}_reconstruction_loss",
            reconstruction_loss,
            on_epoch=(tag == "val"),
            on_step=(tag == "train"),
            logger=True,
            prog_bar=True,
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
