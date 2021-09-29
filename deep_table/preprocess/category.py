from typing import Any, List

from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from deep_table.preprocess.base import BasePreprocessor


class CategoryPreprocessor(BasePreprocessor):
    """Preprocessing categorical features.

    Each category is converted to int.
    If `use_unk` is True, unknown categories are considered.

    Examples:
    >>> import pandas as pd
    >>> df_input = pd.DataFrame(
    >>>    [["a", 4.2],
    >>>     ["b", 2.3],
    >>>     ["a", 1.3],
    >>>     ["d", 1.3]],
    >>>    columns = ["dummy_1", "dummy_2"],
    >>> )
    >>> category_preprocess = CategoryPreprocessor(categorical_columns=["dummy_1"])
    >>> category_preprocess.fit_transform(df_input)
    pd.DataFrame(
        [[0, 4.2],
         [1, 2.3],
         [0, 1.3],
         [2, 1.3]], columns = ["dummy_1", "dummy_2"])
    """

    def __init__(
        self,
        categorical_columns: List[str],
        use_unk: bool = True,
        unk_token: Any = "unk",
    ) -> None:
        """
        Args:
            categorical_columns (list[str]): Categorical features of the dataset.
            use_unk (bool): If True, unknown categories are considered.
                Defaults to True.
            unk_token (bool): The name of the token for the unknown category.
                Defaults to "unk".
        """
        self.categorical_columns = categorical_columns
        self.encoders = dict()
        self.uniques = dict()
        self.reset()

        self.start_labels = dict()
        self.num_categories = 0
        self.use_unk = use_unk
        self.unk_token = unk_token

    def __len__(self):
        return self.num_categories

    def fit(self, df: DataFrame) -> "CategoryPreprocessor":
        self.reset()
        start_label = 0
        for col in self.categorical_columns:
            self.start_labels[col] = start_label
            if self.use_unk:
                targets = [self.unk_token] + list(df[col])
            else:
                targets = list(df[col])
            self.encoders[col].fit(targets)
            self.uniques[col] = set(self.encoders[col].classes_)
            start_label += len(self.uniques[col])
        self.num_categories = start_label
        return self

    def transform(self, df: DataFrame, inplace: bool = True) -> DataFrame:
        if not inplace:
            df = df.copy()
        for col in self.categorical_columns:
            if self.use_unk:
                unique = self.uniques[col]
                dtype = type(next(iter(unique)))
                targets = [
                    dtype(x) if dtype(x) in unique else self.unk_token for x in df[col]
                ]
            else:
                targets = list(df[col])
            df[col] = self.encoders[col].transform(targets) + self.start_labels[col]
        return df

    def inverse_transform(self, df: DataFrame, inplace: bool = True) -> DataFrame:
        if not inplace:
            df = df.copy()
        for col in self.categorical_columns:
            df[col] = self.encoders[col].inverse_transform(df[col])
        return df

    def reset(self) -> None:
        for col in self.categorical_columns:
            self.encoders[col] = LabelEncoder()
            self.uniques[col] = set()
