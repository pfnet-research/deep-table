import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from deep_table.augmentation.swap import RowSwap
from deep_table.nn.encoders.encoder import Encoder
from deep_table.nn.models.base import BaseModel

logger = logging.getLogger(__name__)


class TabTransformerPretrainModel(BaseModel):
    """TabTransformer-RTD pretraining

    References:
         X. Huang and A. Khetan, M. Cvitkovic and Z. Karnin,
         “TabTransformer: Tabular Data Modeling Using Contextual Embeddings,”
         ArXiv: 2012.06678 [cs.LG], 2020. <https://arxiv.org/abs/2012.06678>
    """

    def __init__(
        self,
        encoder: Encoder,
        mask_prob: float = 0.3,
        **kwargs,
    ) -> None:
        """
        Args:
            encoder (Encoder): Encoder module used in training.
            mask_prob (float): Probability of masking features. Each masked feature is
                swapped to the value of another row. For pretraining, this model is
                predicting which values are swapped. Defaults to 0.3.
        """
        self.mask_prob = mask_prob
        super().__init__(encoder, **kwargs)

    def _build_network(self) -> None:
        self.rowswap = RowSwap(prob=self.mask_prob, overlap=True)
        dim_representation = self.encoder.dim_out(is_pretrain=True)
        num_prediction = (
            self.encoder.num_continuous_features + self.encoder.num_categorical_features
        )
        self.linear_pretrain = nn.ModuleList(
            [nn.Linear(dim_representation, 1) for _ in range(num_prediction)]
        )

    def _augment(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if x is not None or x == []:
            x, mask = self.rowswap(x)
        else:
            x = None
            mask = None
        return x, mask

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        # augment
        x["continuous"], mask_con = self._augment(x["continuous"])
        x["categorical"], mask_cat = self._augment(x["categorical"])

        mask = []
        for m in [mask_con, mask_cat]:
            if m is not None:
                mask.append(m)
        mask = torch.cat(mask, dim=-1)

        out_backbone = self.encoder(x, is_pretrain=True)
        x_mask_pred = torch.cat(
            [lin(out_backbone) for lin in self.linear_pretrain], dim=-1
        )
        return {"pred_mask": x_mask_pred, "mask": mask}

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        outputs = self(batch)
        loss = self.calculate_loss(batch, outputs, tag="train")
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        outputs = self(batch)
        _ = self.calculate_loss(batch, outputs, tag="val")

    def _setup_loss(self) -> None:
        self.bce_loss = nn.BCEWithLogitsLoss()

    def calculate_loss(
        self, batch: Dict[str, Tensor], outputs: Dict[str, Tensor], tag: str = "train"
    ) -> Tensor:
        mask = outputs["mask"].float()
        mask = mask
        pred_mask = outputs["pred_mask"]
        loss = torch.mean(
            torch.stack(
                [
                    self.bce_loss(pred_mask[:, i], mask[:, i])
                    for i in range(mask.size(1))
                ],
                dim=0,
            )
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
