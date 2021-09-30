import optuna
from omegaconf import DictConfig, OmegaConf

from deep_table.configs import read_config
from deep_table.configs.optuna import _CONFIG_OPTUNA_ROOT as _ROOT

JSON_ROOT = _ROOT / "jsons"


def update_config_(
    config_orig: DictConfig, path: str, update_key: str
) -> None:
    """Update `DictConfig` using config in `path`.

    This function is inplace computation.

    Args:
        config_orig (`DictConfig`): Original config that will be updated.
        path (str): Path to file. `path` is given to
            :func:`~deep_table.configs.io.read_config` and `config_orig`
            will be updated using this config.
        update_key (str): Updating config of `config_orig.update_key`.
    """
    config = read_config(path)
    config_update = OmegaConf.update(config_orig, update_key, config, force_add=True)
    return config_update


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
