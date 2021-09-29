import optuna
from omegaconf import DictConfig, OmegaConf

from deep_table.configs import read_config
from deep_table.configs.optuna import _CONFIG_OPTUNA_ROOT as _ROOT

JSON_ROOT = _ROOT / "jsons"


def merge_config_optuna_params(
    base_config: DictConfig,
    embedding: str,
    backbone: str,
    head: str,
) -> DictConfig:
    """Utility for creating config.optuna.parameters.

    Each json file defines the range of hyper-parameters.
    """

    config = OmegaConf.create({})
    OmegaConf.set_struct(config, True)

    embedding_args = read_config(JSON_ROOT / "nn/encoders/embedding" / f"{embedding}.json")
    OmegaConf.update(
        config,
        "optuna.parameters.encoder.embedding.args",
        embedding_args,
        force_add=True,
    )

    backbone_args = read_config(JSON_ROOT / "nn/encoders/backbone" / f"{backbone}.json")
    OmegaConf.update(
        config, "optuna.parameters.encoder.backbone.args", backbone_args, force_add=True
    )

    estimator_model_args = read_config(JSON_ROOT / "estimator_model_args.json")
    estimator_model_args.update(read_config(JSON_ROOT / "nn/models/head/" / f"{head}.json"))
    OmegaConf.update(
        config,
        "optuna.parameters.estimator.model_args",
        estimator_model_args,
        force_add=True,
    )
    config = OmegaConf.merge(base_config, config)
    return config


def make_suggest_config(trial: optuna.trial.Trial, config: DictConfig) -> DictConfig:
    """Insert optuna.trial.Trial suggestion to DictConfig.

    Args:
        trial (`optuna.trial.Trial`)
        config (`DictConfig`): Config to update.

    Returns:
        DictConfig: Config updated by `optuna.trial.Trial`.
    """
    suggest_config = OmegaConf.create({})
    OmegaConf.set_struct(suggest_config, True)

    def dfs(trial, cur: DictConfig, suggest_config: DictConfig) -> None:
        for k, v in cur.items():
            if hasattr(v, "type"):
                suggest_func = getattr(trial, f"suggest_{v.type}")
                suggest_arg = suggest_func(name=k, **v.args)
                OmegaConf.update(suggest_config, k, suggest_arg, force_add=True)
            else:
                OmegaConf.update(suggest_config, k, {}, force_add=True)
                dfs(trial, cur[k], suggest_config[k])

    dfs(trial, config, suggest_config)
    return suggest_config
