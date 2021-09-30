from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from torch import Tensor
from torch.utils.data import DataLoader


def get_scores(
    pred: Union[Tensor, pd.Series, np.ndarray],
    target: Union[Tensor, pd.Series, np.ndarray, DataLoader],
    task: str,
    sigmoid: bool = True,
    softmax: bool = True,
    prefix: Optional[str] = None,
) -> Dict[str, float]:
    """
    Args:
        pred (Tensor or pd.Series or np.ndarray): Prediction values.
            These values are converted to numpy arrays.
        target (Tensor or pd.Series or np.ndarray or DataLoader): Ground truth Values.
            These values are converted to numpy arrays.
        task (str): One of "binary", "regression", "multitask".
        sigmoid (bool): If True, sigmoid fuction is applied to `pred`.
            Defaults to True.
        softmax (bool): If True, softmax fuction is applied to `pred`.
            Defaults to True.
        prefix (str, optional): Prefix for the keys. If `task` is "binary",
            the returned value is {"`prefix`_AUC", ...}.

    Returns:
        dict: {merics, value}
    """
    assert isinstance(pred, (Tensor, pd.Series, np.ndarray, DataLoader))
    assert isinstance(target, (Tensor, pd.Series, np.ndarray, DataLoader))

    # Cast `pred` and `target` to `np.array`
    if isinstance(pred, pd.Series):
        pred = pred.to_numpy()
    elif isinstance(pred, Tensor):
        pred = pred.cpu().numpy()

    if isinstance(target, pd.Series):
        target = target.to_numpy()
    elif isinstance(target, Tensor):
        target = target.cpu().numpy()
    elif isinstance(target, DataLoader):
        target = np.concatenate([x["target"].cpu().numpy() for x in target], axis=0)

    pred = pred.astype(float)
    target = target.astype(float)

    if task == "binary":
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        assert pred.shape == target.shape
        scores = get_binary_scores(pred, target, sigmoid)

    elif task == "regression":
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        assert pred.shape == target.shape
        scores = get_regression_scores(pred, target)

    elif task == "multiclass":
        assert pred.shape[0] == target.shape[0]
        scores = get_multiclass_scores(pred, target, softmax)

    else:
        raise ValueError(f"task: {task} is not implemented")

    if prefix is not None:
        scores = {prefix + "_" + str(key): val for key, val in scores.items()}

    return scores


def get_binary_scores(
    pred: np.ndarray, target: np.ndarray, sigmoid: bool = True
) -> Dict[str, float]:
    """Scores for binary classification task."""
    if sigmoid:
        pred = _sigmoid(pred)
    accuracy = accuracy_score(
        y_pred=np.where(pred > 0.5, np.ones_like(pred), np.zeros_like(pred)),
        y_true=target,
    )
    auc = roc_auc_score(y_score=pred, y_true=target)
    f1 = f1_score(
        y_pred=np.where(pred > 0.5, np.ones_like(pred), np.zeros_like(pred)),
        y_true=target,
    )
    cross_entropy = log_loss(y_pred=pred, y_true=target)
    return {
        "accuracy": accuracy,
        "AUC": auc,
        "F1 score": f1,
        "cross_entropy": cross_entropy,
    }


def get_regression_scores(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Scores for regression task."""
    rmse = np.sqrt(mean_squared_error(y_pred=pred, y_true=target))
    mae = mean_absolute_error(y_pred=pred, y_true=target)
    mape = mean_absolute_percentage_error(y_pred=pred, y_true=target)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def get_multiclass_scores(
    pred: np.ndarray, target: np.ndarray, softmax: bool = True
) -> Dict[str, float]:
    """Scores for multiclass classification."""
    if softmax:
        pred = _softmax(pred)
    pred_class = np.argmax(pred, axis=1)
    cross_entropy = log_loss(y_pred=pred, y_true=target)
    accuracy = accuracy_score(y_pred=pred_class, y_true=target)
    return {"cross_entropy": cross_entropy, "accuracy": accuracy}


def _softmax(x: np.ndarray) -> np.ndarray:
    """Softmax function

    Apply softmax function to the 2D array.
    The sum of returned values (axis=1) is 1.

    Examples:
        >>> input_array = np.array([
        >>>       [5, 5, 5],
        >>>       [2, 2, 2],
        >>> ])
        >>> _softmax(input_array)
        np.array([
            [0.333, 0.333, 0.333],
            [0.333, 0.333, 0.333]
        ])
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function

    Apply sigmoid function to the all elements in the array.

    Examples:
        >>> input_array = np.array([[2, 0, -2], [1, 0, -1]])
        >>> _sigmoid(input_array)
        np.array([
            [0.8807970, 0.5, 0.1192029],
            [0.7310585, 0.5, 0.2689414],
        ])
    """
    return 1 / (1 + np.exp(-x))


def mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray, eps=1e-10
) -> float:
    """Mean absolute percentage error (MAPE)

    Args:
        y_true (`np.array`): Target values.
        y_pred (`np.array`): Prediction values.
        eps (float): If `abs(y_true)` is less than `eps`, `eps` will be used
            as denominator to avoid ZeroDivisionError. Defaults to 1e-10.
    """
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)
    return np.mean(mape)
