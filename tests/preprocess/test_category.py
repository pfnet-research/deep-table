import numpy as np
import pandas as pd

from deep_table.preprocess.category import CategoryPreprocessor


def test_category_preprocess():
    df_input = pd.DataFrame(
        [
            ["a", 4.2],
            ["b", 2.3],
            ["a", 1.3],
            ["d", 1.3],
        ]
    )
    df_input.columns = ["dummy_1", "dummy_2"]
    category_preprocess = CategoryPreprocessor(categorical_columns=["dummy_1"])
    df_output = category_preprocess.fit_transform(df_input)
    assert len(category_preprocess) == 4  # "a", "b", "d", "unk"
    assert df_output["dummy_1"].nunique() == 3

    df_pred = pd.DataFrame(
        [
            [0, 4.2],
            [1, 2.3],
            [0, 1.3],
            [2, 1.3],
        ]
    )
    assert np.array_equal(df_input.to_numpy(), df_pred.to_numpy())

    inv = category_preprocess.inverse_transform(df_output)
    assert np.array_equal(df_input.to_numpy(), inv.to_numpy())

    df_input = pd.DataFrame(
        [
            ["a", 4.2],
            ["c", 2.3],
            ["e", 1.3],
            ["d", 1.3],
        ]
    )
    df_input.columns = ["dummy_1", "dummy_2"]
    df_output = category_preprocess.transform(df_input)
    df_pred = pd.DataFrame(
        [
            [0, 4.2],
            [3, 2.3],
            [3, 1.3],
            [2, 1.3],
        ]
    )
    assert np.array_equal(df_input.to_numpy(), df_pred.to_numpy())
