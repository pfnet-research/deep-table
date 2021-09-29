import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from torchvision.datasets.utils import check_integrity


class TabularDataFrame(object):
    """Base class for datasets"""

    def __init__(self, root: str, download: bool = False) -> None:
        """
        Args:
            root (str): Path to the root of datasets for saving/loading.
            download (bool): If True, you must implement `self.download` method
                in the child class. Defaults to False.
        """
        self.root = root
        if download:
            self.download()

    @property
    def mirrors(self) -> None:
        pass

    @property
    def resources(self) -> None:
        pass

    @property
    def raw_folder(self) -> str:
        """The folder where raw data will be stored"""
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _check_exists(self, fpath: Sequence[Tuple[str, Union[str, None]]]) -> bool:
        """
        Args:
            fpath (sequence of tuple[str, (str or None)]): Each value has the format
                [file_path, (md5sum or None)]. Checking if files are correctly
                stored in `self.raw_folder`. If `md5sum` is provided, checking
                the file itself is valid.
        """
        return all(
            check_integrity(os.path.join(self.raw_folder, path[0]), md5=path[1])
            for path in fpath
        )

    def download(self) -> None:
        """
        Implement this function if the dataset is downloadable.
        See :func:`~deep_table.data.datasets.adult.Adult` for an example implementation.
        """
        raise NotImplementedError

    def cat_cardinalities(self, use_unk: bool = True) -> Optional[List[int]]:
        """List of the numbers of the categories of each column.

        Args:
            use_unk (bool): If True, each feature (column) has "unknown" categories.

        Returns:
            List[int], optional: List of cardinalities. i-th value denotes
                the number of categories which i-th column has.
        """
        cardinalities = []
        df_train = self.raw_dataframe(train=True)
        df_train_cat = df_train[self.categorical_columns]
        cardinalities = df_train_cat.nunique().values.astype(int)
        if use_unk:
            cardinalities += 1
        cardinalities_list = cardinalities.tolist()
        return cardinalities_list

    def raw_dataframe(self, train: bool = True) -> pd.DataFrame:
        """
        Args:
            train (bool): If True, the returned value is `pd.DataFrame` for train.
                If False, the returned value is `pd.DataFrame` for test.

        Returns:
            `pd.DataFrame`
        """
        raise NotImplementedError

    def processed_dataframes(self, *args, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Returns:
            dict[str, DataFrame]: The value has the keys "train", "val" and "test".
        """
        raise NotImplementedError

    def num_categories(self, use_unk: bool = True) -> int:
        """Total numbers of categories

        Args:
            use_unk (bool): If True, the returned value is calculated
                as there are unknown categories.
        """
        return sum(self.cat_cardinalities(use_unk=use_unk))
