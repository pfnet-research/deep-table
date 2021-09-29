import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from deep_table.nn.encoders.encoder import Encoder

logger = logging.getLogger(__name__)


class BaseModel(pl.LightningModule, metaclass=ABCMeta):
    def __init__(
        self,
        encoder: Encoder,
        task: Optional[str] = "binary",
        loss: Optional[str] = None,
        learning_rate: float = 1e-3,
        optimizer: str = "Adam",
        optimizer_params: Dict[str, Any] = {},
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[Dict[str, Any]] = {},
        lr_scheduler_monitor_metric: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            encoder (`Encoder`): Encoder used in training. It contains `Embedding` and `Backbone`.
            task (str): If the loss function is not implemented, loss function is set to
                the default corresponding to `task`. `task` should be one of "binary", "regression", "multitask".
                Defaults to "binary".
            loss (str, optional): Loss function used in training. `loss` should be
                an attribute in `torch.nn`. When `loss` is None, loss function corresponding to task is used.
                Defaults to None.
            learning_rate (float): Defaults to 1e-3.
            optimizer (str): `optimizer` should be an attribute in `torch.nn`. Defaults to "Adam".
            optimizer_params (dict[str, Any]): Defaults to {}.
            lr_scheduler (str, optional): Scheduler of learning rate. It must be an attribute
                of `torch.optim.lr_scheduler`. Defaults to None.
            lr_scheduler_params (dict[str, Any]): Paramerters of learning rate scheduler.
                Defaults to {}.
            lr_scheduler_monitor_metric (str, optional): Monitor value of learning rate scheduler.
                Defaults to None.
        """
        super().__init__()
        self.save_hyperparameters(ignore="encoder")
        self.encoder = encoder
        self._build_network()
        self._setup_loss()

    @abstractmethod
    def _build_network(self):
        pass

    @abstractmethod
    def forward(self, x: Dict):
        """Pass the input through Embedding -> AppendClsToken (optional) ->
        Backbone -> Head or Decoder. The output of the Embedding is 3D,
        and that of the Backbone is 2D or 3D.

        Args:
            x (Dict): the dict of continuous and categorical inputs
        """
        pass

    def training_step(self, batch, batch_idx):
        y = batch["target"]
        y_hat = self(batch)
        loss = self.calculate_loss(y, y_hat, tag="train")
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["target"]
        y_hat = self(batch)
        _ = self.calculate_loss(y, y_hat, tag="val")
        return y_hat, y

    def test_step(self, batch, batch_idx):
        y = batch["target"]
        y_hat = self(batch)
        _ = self.calculate_loss(y, y_hat, tag="test")
        return y_hat, y

    def _setup_loss(self):
        if self.hparams.get("loss", None) is not None:
            try:
                self.loss = getattr(nn, self.hparams.loss)()
            except AttributeError as e:
                logger.error(
                    f"{self.hparams.loss} is not a valid loss defined in the torch.nn module"
                )
                raise e
        else:
            self._set_default_loss(self.hparams.task)

    def _set_default_loss(self, task: str):
        if task == "binary":
            self.loss = nn.BCEWithLogitsLoss()
        elif task == "multiclass":
            self.loss = nn.CrossEntropyLoss()
        elif task == "regression":
            self.loss = nn.MSELoss()
        else:
            raise AttributeError(
                f"{task} is not a valid task."
                "Should choose a task from ['binary', 'multiclass', 'regression']."
            )

    def calculate_loss(self, y, y_hat, tag):
        computed_loss = self.loss(y_hat.squeeze(), y.squeeze())
        self.log(
            f"{tag}_loss",
            computed_loss,
            on_epoch=(tag == "val"),
            on_step=(tag == "train"),
            logger=True,
            prog_bar=True,
        )
        return computed_loss

    def configure_optimizers(self):
        try:
            self._optimizer = getattr(torch.optim, self.hparams.optimizer)
            opt = self._optimizer(
                self.parameters(),
                lr=self.hparams.learning_rate,
                **self.hparams.optimizer_params
                if hasattr(self.hparams, "optimizer_params")
                else {},
            )
        except AttributeError as e:
            logger.error(
                f"{self.hparams.optimizer} is not a valid optimizer"
                "defined in `torch.optim`"
            )
            raise e

        if self.hparams.lr_scheduler is not None:
            try:
                self._lr_scheduler = getattr(
                    torch.optim.lr_scheduler, self.hparams.lr_scheduler
                )
            except AttributeError as e:
                logger.error(
                    f"{self.hparams.lr_scheduler} is not a valid learning "
                    "rate scheduler defined in `torch.optim.lr_scheduler`"
                )
                raise e

            if isinstance(self._lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                return {
                    "optimizer": opt,
                    "lr_scheduler": self._lr_scheduler(
                        opt, **self.hparams.lr_scheduler_params
                    ),
                }
            else:
                return {
                    "optimizer": opt,
                    "lr_scheduler": self._lr_scheduler(
                        opt, **self.hparams.lr_scheduler_params
                    ),
                    "monitor": self.hparams.lr_scheduler_monitor_metric,
                }
        else:
            return opt

    @classmethod
    def make(
        cls, encoder, model_config: Optional[DictConfig] = None, **kwargs
    ) -> "BaseModel":
        if model_config is not None:
            return cls(encoder, **model_config, **kwargs)
        else:
            return cls(encoder, **kwargs)
