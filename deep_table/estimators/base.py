import logging
import time
from typing import Dict, Optional, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.base import BaseEstimator
from torch import Tensor
from torch.utils.data import DataLoader

from deep_table.data.data_module import TabularDatamodule
from deep_table.nn.encoders import Encoder, backbone, embedding
from deep_table.nn.encoders.backbone.base import BaseBackbone
from deep_table.nn.encoders.embedding.base import BaseEmbedding
from deep_table.nn.models import BaseModel
from deep_table.nn.models.utils import get_model_callable

logger = logging.getLogger(__name__)


def _logger_elapsed_time(name: str, step: int = 0):
    def inner(func):
        def wrap(self, *args, **kwargs):
            start = time.time()
            ret_val = func(self, *args, **kwargs)
            end = time.time()
            elapsed_time = end - start
            self.log({name: elapsed_time}, step)
            return ret_val

        return wrap

    return inner


class Estimator(BaseEstimator):
    """Estimator for pretraining/fine-tuning or simply training.

    Examples:
        from omegaconf import OmegaConf

        encoder_config = OmegaConf.create({
            "embedding": {
                "name": "FeatureEmbedding",
            },
            "backbone": {
                "name": "MLPBackbone",
            },
        })

        model_config = OmegaConf.create({
            "name": "MLPHeadModel"
        })

        trainer_config = OmegaConf.create({
            "max_epochs": 100,
            "gpus": 1,
        })

        estimator = Estimator(
            encoder_config,
            model_config,
            trainer_config,
        )

        estimator.fit(datamodule)
        predict = estimator.predict(datamodule.dataloader(split="test"))
    """

    def __init__(
        self,
        encoder_config: DictConfig,
        model_config: DictConfig,
        trainer_config: Optional[DictConfig] = None,
        early_stop_round: Optional[int] = None,
        monitor: str = "val_loss",
        mode: str = "min",
        logger: Optional[pl.loggers.base.LightningLoggerBase] = None,
        custom_embedding: Optional[BaseEmbedding] = None,
        custom_backbone: Optional[BaseBackbone] = None,
        custom_model: Optional[BaseModel] = None,
    ) -> None:
        """
        Args:
            encoder_config (DictConfig): Encoder settings. See `Estimator.__doc__` for more details.
            model_config (DictConfig): Model settings. See `Estimator.__doc__` for more details.
            trainer_config (DictConfig, optional): 'trainer' settings (gpus, max_epochs...)
            early_stop_round (int, optional): Patience of early stopping.
            monitor (str): Value used for early stopping. Defaults to "val_loss".
            mode (str): One of "min", "max". If "min", Training will be stopped when `monitor`
                stops decreasing. If "max", Training will be stopped when `monitor`
                stops increasing. Defaults to "min".
            logger (pl.loggers.base.LightningLoggerBase, optional): If None,
                model is trained without logging. Defaults to None.
            custom_embedding (BaseEmbedding, optional): Custom Embedding module.
                If set, `encoder_config.embedding.name` would be overrided.
                Defaults to None.
            custom_backbone (BaseBackbone, optional): Custom Backbone module.
                If set, `encoder_config.backbone.name` would be overrided.
                Defaults to None.
            custom_model (BaseModel, optional): Custom Model module.
                If set, `model_config` would be overrided. Defaults to None.
        """
        self.encoder_config = encoder_config
        self.trainer_config = trainer_config
        self.model_config = model_config
        self.logger = logger
        if early_stop_round is not None:
            self._prepare_callback(early_stop_round, monitor, mode)
        else:
            self.callbacks = None
        self.custom_embedding = custom_embedding
        self.custom_backbone = custom_backbone
        self.custom_model = custom_model

    def _prepare_model(
        self,
        from_pretrained: Optional[Union[BaseModel, "Estimator"]] = None,
        task: str = "binary",
        dim_out: int = 1,
    ) -> None:
        if from_pretrained is not None:
            self.encoder = from_pretrained.encoder
        else:
            if self.custom_embedding is not None:
                _embedding_cls = self.custom_embedding
            else:
                _embedding_name = self.encoder_config.embedding.name
                _embedding_cls = getattr(embedding, _embedding_name)

            _embedding = _embedding_cls.make(
                self.encoder_config.embedding,
                num_continuous_features=self.encoder_config.num_continuous_features,
                num_categorical_features=self.encoder_config.num_categorical_features,
                num_categories=self.encoder_config.num_categories,
            )

            if self.custom_backbone is not None:
                _backbone_cls = self.custom_backbone
            else:
                _backbone_name = self.encoder_config.backbone.name
                _backbone_cls = getattr(backbone, _backbone_name)

            _backbone = _backbone_cls.make(
                num_features=_embedding.dim_in_backbone[0],
                dim_embed=_embedding.dim_in_backbone[1],
                use_cls=_embedding.use_cls,
                config=self.encoder_config.backbone,
            )

            self.encoder = Encoder(
                config_encoder=self.encoder_config,
                embedding=_embedding,
                backbone=_backbone,
            )

        if self.custom_model is not None:
            _model = self.custom_model
        else:
            _model = get_model_callable(self.model_config.name)
        self.model = _model.make(
            self.encoder, self.model_config, task=task, dim_out=dim_out
        )

    def _prepare_callback(self, early_stop_round: int, monitor: str, mode: str) -> None:
        early_stop_callback = EarlyStopping(
            monitor=monitor, patience=early_stop_round, verbose=True, mode=mode
        )
        self.callbacks = [early_stop_callback]

    def _prepare_trainer(self) -> None:
        """ """
        if self.trainer_config is not None:
            self.trainer = pl.Trainer(
                logger=self.logger,
                callbacks=self.callbacks,
                max_epochs=self.trainer_config.get("max_epochs", 1),
                gpus=self.trainer_config.get("gpus", 0),
                fast_dev_run=self.trainer_config.get("fast_dev_run", False),
            )
        else:
            self.trainer = pl.Trainer(
                logger=self.logger,
                callbacks=self.callbacks,
                max_epochs=1,
                gpus=1 if torch.cuda.is_available() else 0,
                fast_dev_run=True,
            )

    def _pre_fit(
        self,
        datamodule: TabularDatamodule,
        from_pretrained: Optional[Union[BaseModel, "Estimator"]] = None,
    ) -> None:
        self._set_params_from_datamodule(datamodule)
        self._prepare_trainer()
        self._prepare_model(
            from_pretrained, task=datamodule.task, dim_out=datamodule.dim_out
        )
        self.log({"trainable_params": self.n_parameters}, step=0)

    def _set_params_from_datamodule(self, datamodule: TabularDatamodule) -> None:
        update_config = OmegaConf.create(
            {
                "num_continuous_features": datamodule.num_continuous_features,
                "num_categorical_features": datamodule.num_categorical_features,
                "num_categories": datamodule.num_categories,
            }
        )
        update_config.update(self.encoder_config)
        self.encoder_config = update_config

    def _set_seed(self, seed):
        if seed is not None:
            seed_everything(seed)
        elif self.trainer_config is not None and self.trainer_config.get("seed", False):
            seed_everything(self.trainer_config.seed)
        else:
            seed_everything(0)

    @_logger_elapsed_time(name="time_fit", step=0)
    def fit(
        self,
        datamodule: TabularDatamodule,
        seed: Optional[int] = None,
        from_pretrained: Optional[Union[BaseModel, "Estimator"]] = None,
    ) -> None:
        """
        Args:
            datamodule (TabularDatamodule)
            seed (int, optional): Defaults to None.
            from_pretrained (BaseModel, `Estimator`): If set, the weights of
                `Encoder` are set to the instance. Defaults to None.
        """
        self._pre_fit(datamodule, from_pretrained)
        self._set_seed(seed)
        self.model.train()
        self.trainer.fit(
            self.model,
            datamodule.dataloader("train"),
            datamodule.dataloader("val") if datamodule.val is not None else None,
        )
        logger.info("Training the model completed...")

    def predict(self, dataloader: Union[DataLoader, LightningDataModule]) -> Tensor:
        """Uses the trained model to predict on new data and return as a dataframe

        Args:
            dataloader (DataLoader or LightningDataModule)

        Returns:
            Tensor: Prediction for the given dataloader.
        """
        pred = self.trainer.predict(model=self.model, dataloaders=dataloader)
        return torch.cat(pred, 0)

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Args:
          metrics (dict[str, float])
          step (int, optional): Step logged with `metrics`. Defaults to None.
        """
        if self.logger is None:
            logger.info(
                f"self.logger is not set. Continue without logging: {metrics.keys()}"
            )
            return
        self.logger.log_metrics(metrics, step)
        self.logger.save()

    @property
    def n_parameters(self) -> int:
        """Trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def load_from_checkpoint(self, path: str) -> None:
        """
        Args:
          path (str): The path to checkpoints
        """
        if torch.cuda.is_available():
            state = torch.load(path)["state_dict"]
        else:
            state = torch.load(path, map_location=torch.device("cpu"))["state_dict"]
        self.encoder.load_state_dict(state, strict=False)
        self.model.load_state_dict(state, strict=False)
