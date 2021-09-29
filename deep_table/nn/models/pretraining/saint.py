from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from deep_table.augmentation import Cutmix, Mixup
from deep_table.nn.layers.mlp import SimpleMLPLayer
from deep_table.nn.models.base import BaseModel
from deep_table.nn.models.loss import InfoNCELoss


class SAINTPretrainModel(BaseModel):
    """SAINT pretrain model.

    References:
        G. Somepalli, M. Goldblum, A. Schwarzschild, C. B. Bruss and T. Goldstein,
        “SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training,”
        ArXiv:2106.01342 [cs.LG], 2021. <https://arxiv.org/abs/2106.01342>
    """

    def __init__(
        self,
        encoder,
        dim_embed: int = 16,
        mask_prob: float = 0.1,
        alpha: float = 0.9,
        temp_param: float = 0.7,
        lambda_: float = 10,
        dim_z: int = 2,
        dim_g_hidden: int = 256,
        dim_mlp_hidden: int = 256,
        **kwargs,
    ) -> None:
        """
        Args:
            encoder (`Encoder`): Encoder module used in training.
            dim_feature_estimator (int): Size of dimension in the hidden layer of MLP.
                The model predicts the original features using this MLP module.
                Defaults to 256.
            prob_row (float):
                Probability of applying row-wise swapping.
                Defaults to 0.2.
            prob_column (float):
                Probability of applying column-wise swapping.
        """
        self.save_hyperparameters(ignore="encoder")
        super(SAINTPretrainModel, self).__init__(encoder, **kwargs)

    def _build_network(self) -> None:
        self.cutmix = Cutmix(self.hparams.mask_prob)
        self.mixup = Mixup(self.hparams.alpha)

        self.num_categorical_features = self.encoder.num_categorical_features
        self.num_continuous_features = self.encoder.num_continuous_features
        self.num_categories = self.encoder.num_categories
        self.num_features = (
            self.num_categorical_features + self.num_continuous_features + 1
        )
        dim_representation = self.encoder.dim_out(is_pretrain=True)

        self.g1 = SimpleMLPLayer(
            in_dim=dim_representation,
            out_dim=self.hparams.dim_z,
            hidden_dim=self.hparams.dim_g_hidden,
        )
        self.g2 = SimpleMLPLayer(
            in_dim=dim_representation,
            out_dim=self.hparams.dim_z,
            hidden_dim=self.hparams.dim_g_hidden,
        )

        layers = []
        for i in range(self.num_features):
            if i < self.num_continuous_features:
                layers.append(
                    SimpleMLPLayer(
                        in_dim=dim_representation,
                        out_dim=1,
                        hidden_dim=self.hparams.dim_mlp_hidden,
                    )
                )
            else:
                layers.append(
                    SimpleMLPLayer(
                        in_dim=dim_representation,
                        out_dim=self.num_categories,
                        hidden_dim=self.hparams.dim_mlp_hidden,
                    )
                )

        self.feature_wise_mlp = nn.ModuleList(layers)

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x_origin = self._encoder(x)
        x_noisy = self._encoder(x, add_noise=True)

        if x_origin.dim() == 3:
            x_origin = x_origin.flatten(1)
            x_noisy = x_noisy.flatten(1)
        else:
            assert x_origin.dim() == 2

        z_origin = self.g1(x_origin)
        z_noisy = self.g2(x_noisy)
        x_reconstructed = [mlp(x_noisy) for mlp in self.feature_wise_mlp]
        return {
            "z_origin": z_origin,
            "z_noisy": z_noisy,
            "con_reconstructed": torch.stack(
                x_reconstructed[: self.num_continuous_features], 1
            )
            if self.num_continuous_features > 0
            else None,
            "cat_reconstructed": torch.stack(
                x_reconstructed[self.num_continuous_features :], 1
            )
            if self.num_categorical_features > 0
            else None,
        }

    def _encoder(self, x: Dict[str, Tensor], add_noise: bool = False) -> Tensor:
        if add_noise:
            if (self.num_continuous_features > 0) and (
                self.num_categorical_features > 0
            ):
                x_cat, x_con = self.cutmix(x["categorical"], x["continuous"])
                x = {"categorical": x_cat, "continuous": x_con}
            elif self.num_continuous_features > 0:
                x["continuous"] = self.cutmix(x["continuous"])[0]
            elif self.num_categorical_features > 0:
                x["categorical"] = self.cutmix(x["categorical"])[0]
            x = self.encoder.forward_embedding(x)
            x["in_backbone"] = self.mixup(x["in_backbone"])[0]
        else:
            x = self.encoder.forward_embedding(x)
        x = self.encoder.forward_backbone(x, is_pretrain=True)
        return x

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        outputs = self(batch)
        loss = self.calculate_loss(batch, outputs, tag="train")
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        outputs = self(batch)
        _ = self.calculate_loss(batch, outputs, tag="val")

    def _setup_loss(self) -> None:
        self.contranstive_loss = InfoNCELoss()
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def calculate_loss(
        self, batch: Dict[str, Tensor], outputs: Dict[str, Tensor], tag: str = "train"
    ) -> Tensor:
        contrastive_loss = self.contranstive_loss(
            outputs["z_origin"], outputs["z_noisy"], self.hparams.temp_param
        )
        denoising_loss = 0
        if outputs["con_reconstructed"] is not None:
            denoising_loss = denoising_loss + self.mse_loss(
                outputs["con_reconstructed"].squeeze(), batch["continuous"]
            )
        for i in range(self.num_categorical_features):
            denoising_loss = denoising_loss + self.cross_entropy_loss(
                outputs["cat_reconstructed"][:, i], batch["categorical"][:, i]
            )
        loss = contrastive_loss + self.hparams.lambda_ * denoising_loss

        self.log(
            f"{tag}_contrastive_loss",
            contrastive_loss,
            on_epoch=(tag == "valid"),
            on_step=(tag == "train"),
            logger=True,
            prog_bar=True,
        )
        self.log(
            f"{tag}_denoising_loss",
            denoising_loss,
            on_epoch=(tag == "valid"),
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
