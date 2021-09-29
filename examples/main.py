import os
from pathlib import Path
from typing import Union

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


def objective(
    config: DictConfig,
    datamodule: TabularDatamodule,
    target: Union[pd.DataFrame, np.ndarray, Tensor],
    task: str,
):
    def update_suggest_config(trial: optuna.trial.Trial) -> DictConfig:
        suggest_config = make_suggest_config(trial, config.optuna.parameters)
        OmegaConf.set_struct(config, False)
        merge_config = OmegaConf.unsafe_merge(config, suggest_config)
        return merge_config

    def objective_fn(trial):
        config = update_suggest_config(trial)
        print(config)
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
        pred = estimator.predict(datamodule.dataloader("val"))
        scores = get_scores(
            pred=pred,
            target=dataframes["val"][dataset.target_columns].to_numpy(),
            task=task,
        )
        val_scores = {"val_" + str(key): val for key, val in scores.items()}
        estimator.log(val_scores, step=0)
        objective_loss = val_scores["val_" + config.optuna.objective_loss]

        # TEST SCORE
        pred = estimator.predict(datamodule.dataloader("test"))
        test_scores = get_scores(
            pred=pred,
            target=dataframes["test"][dataset.target_columns].to_numpy(),
            task=task,
        )
        test_scores = {"test_" + str(key): val for key, val in test_scores.items()}
        estimator.log(test_scores, step=0)
        return objective_loss

    return objective_fn


if __name__ == "__main__":
    basedir = Path(os.path.dirname(__file__))
    config = read_config(basedir / "config.json")
    config = merge_config_optuna_params(
        config,
        os.path.join(basedir, "jsons/nn/encoders/embedding/",
                     config.encoder.embedding.name+".json"),
        os.path.join(basedir, "jsons/nn/encoders/backbone/",
                     config.encoder.backbone.name+".json"),
        os.path.join(basedir, "jsons/nn/models/head",
                     config.estimator.model_args.name+".json"),
        os.path.join(basedir, "jsons/estimator_model_args.json")
    )

    dataset_dir = Path("data")

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
            target=dataframes["test"][dataset.target_columns].to_numpy(),
            task=dataset.task,
        ),
        n_trials=config.optuna.n_trials,
    )
    best_trial = study.best_trial
    print(best_trial)
