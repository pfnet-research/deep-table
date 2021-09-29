import torch

from deep_table.estimators.base import Estimator


def test_estimator(encoder_config, model_config, dummy_datamodule):
    estimator = Estimator(encoder_config, model_config)
    estimator.fit(datamodule=dummy_datamodule)
    predict = estimator.predict(dummy_datamodule.dataloader(split="test"))
    assert isinstance(predict, torch.Tensor)
    assert estimator.n_parameters > 0
