from typing import Callable

from deep_table.nn.models import head, pretraining


def get_model_callable(model_name: str) -> Callable:
    """
    Args:
        model_name (str): Model should implemented in :func:`~deep_table.nn.models.head`
        or :func:`~deep_table.nn.models.pretraining`.
    """
    if hasattr(head, model_name):
        return getattr(head, model_name)
    elif hasattr(pretraining, model_name):
        return getattr(pretraining, model_name)
    else:
        raise AttributeError(f"{model_name} does not exist in (head, pretraining)")
