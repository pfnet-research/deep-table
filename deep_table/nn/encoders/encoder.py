from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from deep_table.nn.encoders.backbone import BaseBackbone
from deep_table.nn.encoders.embedding import BaseEmbedding


class Encoder(nn.Module):
    def __init__(
        self,
        config_encoder: DictConfig,
        embedding: BaseEmbedding,
        backbone: BaseBackbone,
    ) -> None:
        """
        Args:
            config_encoder (DictConfig): Settings of `Encoder`.
            embedding (BaseEmbedding): Instance of `Embedding`. Input features
                are tokenized or simply normalized in this module.
            backbone (BaseBackbone): Instance of `Backbone`. Tokenized features
                are processed in this module.
        """
        super().__init__()
        self.num_continuous_features = config_encoder.num_continuous_features
        self.num_categorical_features = config_encoder.num_categorical_features
        self.num_categories = config_encoder.num_categories
        self.embedding = embedding
        self.backbone = backbone

    def _choose_from_out_backbone(
        self, x: Optional[Tensor], is_pretrain: bool = False
    ) -> Optional[Tensor]:
        if x is None:
            return None
        if x.dim() == 3 and self.embedding.use_cls and not is_pretrain:
            return x[:, -1, :]
        else:
            return x.flatten(1)

    def forward(self, x: Dict[str, Tensor], is_pretrain: bool = False) -> Tensor:
        x = self.forward_embedding(x)
        x = self.forward_backbone(x, is_pretrain)
        return x

    def forward_embedding(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.embedding(x)
        return x

    def forward_backbone(
        self, x: Dict[str, Tensor], is_pretrain: bool = False
    ) -> Tensor:
        x_out_backbone = self.backbone(x["in_backbone"])
        x_skip_backbone = x["skip_backbone"]

        out_extractor = []
        x_out_backbone = self._choose_from_out_backbone(x_out_backbone, is_pretrain)
        if x_out_backbone is not None:
            out_extractor.append(x_out_backbone)

        if x_skip_backbone is not None:
            out_extractor.append(x_skip_backbone)

        out_extractor = torch.cat(out_extractor, dim=1)
        return out_extractor

    def dim_out(self, is_pretrain: bool) -> int:
        if self.embedding.dim_in_backbone[0] == int(
            self.embedding.use_cls
        ):  # if backbone is not used
            backbone_dim_out = 0
        else:
            backbone_dim_out = self.backbone.dim_out(is_pretrain=is_pretrain)

        return backbone_dim_out + self.embedding.dim_skip_backbone
