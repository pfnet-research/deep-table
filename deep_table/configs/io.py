import os
import json

from omegaconf import DictConfig, OmegaConf


def read_config(filepath: str) -> DictConfig:
    """Utilities to read config.

    Args:
        filepath (str): Target path to the file.
            The file must be json or yaml format.

    Returns:
        DictConfig
    """
    config = None

    if not os.path.isfile(filepath):
        raise ValueError(f"{filepath} does not exist.")

    try:  # loading yaml file
        config = OmegaConf.load(filepath)
    except Exception:
        pass

    try:  # loading json file
        with open(filepath, "r") as f:
            config = OmegaConf.create(json.load(f))
    except Exception:
        pass

    if config is None:
        raise ValueError(f"filepath: {filepath} must be json or yaml file.")
    return config
