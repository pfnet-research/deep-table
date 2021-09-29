import pytest
from omegaconf import OmegaConf


@pytest.fixture(scope="session")
def encoder_config():
    return OmegaConf.create(
        {
            "embedding": {
                "name": "FeatureEmbedding",
            },
            "backbone": {"name": "MLPBackbone"},
        }
    )


@pytest.fixture(scope="session")
def model_config():
    return OmegaConf.create(
        {
            "name": "MLPHeadModel",
        }
    )
