import os
from typing import Dict
from urllib.error import URLError

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision.datasets.utils import download_url

from deep_table.data.datasets.tabular_dataframe import TabularDataFrame
from deep_table.preprocess import CategoryPreprocessor


class Adult(TabularDataFrame):
    mirrors = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/",
    ]

    resources = [
        ("adult.data", "5d7c39d7b8804f071cdd1f2a7c460872"),
        ("adult.test", "35238206dfdf7f1fe215bbb874adecdc"),
    ]

    dim_out = 1

    all_columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    continuous_columns = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    categorical_columns = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    target_columns = ["income"]

    task = "binary"

    def __init__(self, root: str, download: bool = True) -> None:
        super().__init__(root=root, download=download)

    def download(self) -> None:
        if self._check_exists(self.resources):
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    download_url(url, root=self.raw_folder, filename=filename, md5=md5)
                except URLError as error:
                    print("Failed to download (trying next):\n{}".format(error))
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))

    def raw_dataframe(self, train: bool = True) -> pd.DataFrame:
        if train:
            df = pd.read_csv(os.path.join(self.raw_folder, "adult.data"), header=None)
            df.columns = self.all_columns
        else:
            df = pd.read_csv(
                os.path.join(self.raw_folder, "adult.test"), header=None, skiprows=1
            )
            df.columns = self.all_columns
            df["income"] = df["income"].replace(" <=50K.", " <=50K")
            df["income"] = df["income"].replace(" >50K.", " >50K")
        return df

    def processed_dataframes(self, *args, **kwargs) -> Dict[str, pd.DataFrame]:
        df_train = self.raw_dataframe(train=True)
        df_test = self.raw_dataframe(train=False)

        df_train, df_val = train_test_split(
            df_train, stratify=df_train["income"], **kwargs
        )

        # preprocessing
        ce = CategoryPreprocessor(categorical_columns=self.categorical_columns)
        ce.fit_transform(df_train)
        ce.transform(df_val)
        ce.transform(df_test)

        sc = StandardScaler()
        df_train[self.continuous_columns] = sc.fit_transform(
            df_train[self.continuous_columns]
        )
        df_val[self.continuous_columns] = sc.transform(df_val[self.continuous_columns])
        df_test[self.continuous_columns] = sc.transform(
            df_test[self.continuous_columns]
        )

        df_train = self._label_encoder(df_train)
        df_val = self._label_encoder(df_val)
        df_test = self._label_encoder(df_test)

        return {
            "train": df_train,
            "val": df_val,
            "test": df_test,
        }

    def _label_encoder(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.target_columns] = df[self.target_columns].replace(" <=50K", 1)
        df[self.target_columns] = df[self.target_columns].replace(" >50K", 0)
        return df
