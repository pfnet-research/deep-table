import pytest

from deep_table.nn.encoders.encoder import Encoder
from deep_table.nn.models.base import BaseModel
from deep_table.nn.models.utils import get_model_callable


@pytest.fixture()
def encoder(mocker):
    encoder = mocker.Mock(spc=Encoder)
    mocker.patch.object(encoder, "num_continuous_features", 1)
    mocker.patch.object(encoder, "num_categorical_features", 1)
    mocker.patch.object(encoder, "num_categories", 2)
    mocker.patch.object(encoder, "dim_out", return_value=1)
    return encoder


def test_basemodel(encoder, mocker):
    basemodel = mocker.Mock(BaseModel)
    mocker.patch.object(
        basemodel, "_set_default_loss", side_effect=BaseModel._set_default_loss
    )

    # Test `_set_default_loss`
    for t in ["binary", "multiclass", "regression"]:
        basemodel._set_default_loss(basemodel, task=t)

    with pytest.raises(AttributeError):
        basemodel._set_default_loss(basemodel, task="multi-regiression")


@pytest.mark.parametrize(
    "name",
    [
        "SAINTPretrainModel",
        "VIMEPretrainModel",
        "DenoisingPretrainModel",
        "TabTransformerPretrainModel",
        "MLPHeadModel",
    ],
)
def test_models(name, encoder, mocker):
    get_model_callable(name).make(encoder, task="binary", dim_out=1)
