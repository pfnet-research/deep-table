import os
import pprint
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor

from deep_table.configs import read_config
from deep_table.configs.optuna.utils import (
    make_suggest_config,
    merge_config_optuna_params,
)
from deep_table.data import datasets
from deep_table.data.data_module import TabularDatamodule
from deep_table.estimators.base import Estimator
from deep_table.utils import get_scores

pp = pprint.PrettyPrinter(width=41, indent=4)
dataset_dir = Path("data")


def _print_configs(config: DictConfig) -> None:
    def _pprint_dictconfig(conf: DictConfig):
        pp.pprint(OmegaConf.to_container(conf))

    _pprint_dictconfig(config.encoder)
    if config.pretrainer.enable:
        _pprint_dictconfig(config.pretrainer)
    _pprint_dictconfig(config.estimator)


def objective(
    config: DictConfig,
    datamodule: TabularDatamodule,
    val_target: Union[pd.DataFrame, np.ndarray, Tensor],
    test_target: Union[pd.DataFrame, np.ndarray, Tensor],
    target_columns: Sequence[str],
    task: str,
):
    def update_suggest_config(trial: optuna.trial.Trial) -> DictConfig:
        suggest_config = make_suggest_config(trial, config.optuna.parameters)
        OmegaConf.set_struct(config, False)
        merge_config = OmegaConf.unsafe_merge(config, suggest_config)
        return merge_config

    def objective_fn(trial: optuna.trial.Trial) -> float:
        config = update_suggest_config(trial)

        _print_configs(config)

        # Pre-training
        if config.pretrainer.enable:
            logger = CSVLogger(save_dir=config.logger.save_dir)
            pretrainer = Estimator(
                encoder_config=config.encoder,
                trainer_config=config.pretrainer.trainer_args,
                model_config=config.pretrainer.model_args,
                early_stop_round=config.pretrainer.trainer_args.early_stop_round,
                logger=logger,
            )
            pretrainer.fit(datamodule=datamodule)

        # Fine-tuning
        logger = CSVLogger(
            save_dir=config.logger.save_dir,
        )
        estimator = Estimator(
            encoder_config=config.encoder,
            trainer_config=config.estimator.trainer_args,
            model_config=config.estimator.model_args,
            early_stop_round=config.estimator.trainer_args.early_stop_round,
            logger=logger,
        )
        estimator.fit(
            datamodule=datamodule,
            from_pretrained=pretrainer if config.pretrainer.enable else None,
        )

        # Prediction for val data
        pred = estimator.predict(datamodule.dataloader("val"))
        val_scores = get_scores(
            pred=pred,
            target=val_target,
            task=task,
            prefix="val",
        )
        estimator.log(val_scores, step=0)
        objective_loss = val_scores["val_" + config.optuna.objective_loss]

        # Prediction for test data
        pred = estimator.predict(datamodule.dataloader("test"))
        test_scores = get_scores(
            pred=pred,
            target=test_target,
            task=task,
            prefix="test",
        )
        estimator.log(test_scores, step=0)

        return objective_loss

    return objective_fn


def make_config(path_config: str = "config.json") -> DictConfig:
    basedir = Path(os.path.dirname(__file__))
    config = read_config(basedir / path_config)
    config = merge_config_optuna_params(
        config,
        os.path.join(
            basedir,
            "jsons/nn/encoders/embedding/",
            config.encoder.embedding.name + ".json",
        ),
        os.path.join(
            basedir,
            "jsons/nn/encoders/backbone/",
            config.encoder.backbone.name + ".json",
        ),
        os.path.join(
            basedir, "jsons/nn/models/head", config.estimator.model_args.name + ".json"
        ),
        os.path.join(basedir, "jsons/estimator_model_args.json"),
    )
    return config


def prepare_dataset(
    config: DictConfig
) -> Tuple[TabularDatamodule, np.ndarray, np.ndarray]:
    dataset = getattr(datasets, config.dataset)(root=dataset_dir)
    dataframes = dataset.processed_dataframes(**config.dataframe_args)

    datamodule = TabularDatamodule(
        train=dataframes["train"],
        val=dataframes["val"],
        test=dataframes["test"],
        task=dataset.task,
        dim_out=dataset.dim_out,
        categorical_columns=dataset.categorical_columns,
        continuous_columns=dataset.continuous_columns,
        target=dataset.target_columns,
        num_categories=dataset.num_categories(),
        **config.datamodule_args,
    )

    val_target = dataframes["val"][dataset.target_columns].to_numpy()
    test_target = dataframes["test"][dataset.target_columns].to_numpy()

    return datamodule, val_target, test_target


def train_with_optuna(
    config: DictConfig,
    datamodule: TabularDatamodule,
    val_target: np.ndarray,
    test_target=np.ndarray,
):
    os.makedirs(config.logger.save_dir, exist_ok=True)
    study = optuna.create_study(
        direction=config.optuna.create_study.direction,
        storage=f"sqlite:///{config.logger.save_dir}/experiment.sqlite",
        study_name=config.optuna.create_study.study_name,
        load_if_exists=True,
    )

    study.optimize(
        func=objective(
            config=config,
            datamodule=datamodule,
            val_target=val_target,
            test_target=test_target,
            target_columns=datamodule.target,
            task=datamodule.task,
        ),
        n_trials=config.optuna.n_trials,
    )
    best_trial = study.best_trial
    print(best_trial)


if __name__ == "__main__":
    config = make_config()
    datamodule, val_target, test_target = prepare_dataset(config)
    train_with_optuna(config, datamodule, val_target, test_target)
