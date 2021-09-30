from pathlib import Path

import pandas as pd
import xgboost as xgb

from deep_table.data import datasets
from deep_table.utils import get_scores

DATASET_DIR = Path("data")

DATA = "ForestCoverType"

seed = 42
num_round = 500
early_stopping_rounds = 20


if __name__ == "__main__":
    # Prepare datasets
    dataset = getattr(datasets, DATA)(root=DATASET_DIR)
    dataframes = dataset.processed_dataframes(random_state=seed)
    df_train = dataframes["train"]
    df_val = dataframes["val"]
    df_test = dataframes["test"]
    target_cols = dataset.target_cols

    # Preprocessing
    X_train = pd.get_dummies(df_train.drop(columns=target_cols)).values
    y_train = df_train[target_cols].values
    X_val = pd.get_dummies(df_val.drop(columns=target_cols)).values
    y_val = df_val[target_cols].values
    X_test = pd.get_dummies(df_test.drop(columns=target_cols)).values
    y_test = df_test[target_cols].values

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    # Parameters for xgb.train
    params = {
        "objective": "binary:logistic"
        if dataset.task == "binary"
        else "multi:softprob",
        "random_state": seed,
    }
    if DATA == "ForestCoverType":
        params["num_class"] = 7

    watchlist = [(dtrain, "train"), (dval, "eval")]

    # Training
    model = xgb.train(
        params,
        dtrain,
        num_round,
        early_stopping_rounds=early_stopping_rounds,
        evals=watchlist,
    )

    # Prediction
    pred = model.predict(dtest)

    scores = get_scores(
        pred=pred, target=y_test, task=dataset.task, sigmoid=False, softmax=False
    )
    print(scores)
