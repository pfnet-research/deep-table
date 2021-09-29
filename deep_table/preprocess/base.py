from abc import ABCMeta, abstractmethod

from pandas import DataFrame


class BasePreprocessor(metaclass=ABCMeta):
    def fit_transform(self, df: DataFrame, inplace: bool = True) -> DataFrame:
        if not inplace:
            df = df.copy()
        self.fit(df)
        return self.transform(df)

    @abstractmethod
    def fit(self, df: DataFrame) -> "Encoder":
        raise NotImplementedError

    @abstractmethod
    def transform(self, df: DataFrame) -> DataFrame:
        raise NotImplementedError
