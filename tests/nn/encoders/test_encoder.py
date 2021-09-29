import torch
from omegaconf import OmegaConf

from deep_table.nn.encoders import Encoder
from deep_table.nn.encoders.backbone import BaseBackbone
from deep_table.nn.encoders.embedding import BaseEmbedding


def test_encoder(mocker, dummy_datamodule):
    dummy_data = dummy_datamodule.dataloader("train").__iter__().next()
    encoder_config = OmegaConf.create(
        {
            "num_categories": 2,
            "num_categorical_features": 1,
            "num_continuous_features": 1,
        }
    )

    embedding = mocker.Mock(spec=BaseEmbedding)
    embedding.use_cls = True
    embedding.return_value = {"in_backbone": None, "skip_backbone": None}

    backbone = mocker.Mock(spec=BaseBackbone)
    backbone.return_value = torch.randn(1, 2, 3)

    encoder = Encoder(encoder_config, embedding, backbone)
    assert isinstance(encoder(dummy_data), torch.Tensor)
