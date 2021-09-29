import numpy as np
import pytest
import torch

from deep_table.nn.encoders import backbone as B


@pytest.mark.parametrize(
    "name", ["FTTransformerBackbone", "MLPBackbone", "SAINTBackbone"]
)
def test_embedding(name):
    num_features = 2
    dim_embed = 16
    backbone = getattr(B, name).make(
        num_features=num_features, dim_embed=dim_embed, use_cls=True
    )
    dummy_data = torch.randn((1, num_features, dim_embed))
    out = backbone(dummy_data)
    assert isinstance(out, torch.Tensor)
