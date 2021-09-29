import pytest

from deep_table.nn.encoders import embedding as E


@pytest.mark.parametrize("name", ["FeatureEmbedding", "TabTransformerEmbedding"])
def test_embedding(name, dummy_datamodule):
    embedding = getattr(E, name).make(
        num_categorical_features=1,
        num_continuous_features=1,
        num_categories=2,
    )
    dummy_data = dummy_datamodule.dataloader("train").__iter__().next()
    out = embedding(dummy_data)
    assert "skip_backbone" in out.keys()
    assert "in_backbone" in out.keys()
