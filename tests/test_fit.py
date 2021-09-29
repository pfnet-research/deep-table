import pytest
import torch
from omegaconf import OmegaConf

from deep_table.estimators.base import Estimator

backbone_name = ["MLPBackbone", "FTTransformerBackbone", "SAINTBackbone"]
embedding_name = ["FeatureEmbedding", "TabTransformerEmbedding"]
model_name = [
    "MLPHeadModel",
    "SAINTPretrainModel",
    "VIMEPretrainModel",
    "DenoisingPretrainModel",
    "TabTransformerPretrainModel",
]


@pytest.mark.slow
@pytest.mark.parametrize("backbone_name", backbone_name)
@pytest.mark.parametrize("embedding_name", embedding_name)
@pytest.mark.parametrize("model_name", model_name)
def test_all_fit(embedding_name, backbone_name, model_name, dummy_datamodule):
    """Tests all combinations of the implemented models"""

    if embedding_name == "TabTransformerEmbedding":
        if backbone_name != "FTTransformerBackbone":
            return
    config = OmegaConf.create({})
    OmegaConf.set_struct(config, False)

    OmegaConf.update(config, "encoder.embedding.name", embedding_name)
    OmegaConf.update(config, "encoder.backbone.name", backbone_name)
    OmegaConf.update(config, "estimator.model_args.name", model_name)

    estimator = Estimator(
        encoder_config=config.encoder,
        trainer_config=None,
        model_config=config.estimator.model_args,
    )
    estimator.fit(datamodule=dummy_datamodule)
