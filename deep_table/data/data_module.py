from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pandas import DataFrame
from torch import LongTensor, Tensor
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    def __init__(
        self,
        data: DataFrame,
        task: str = "binary",
        continuous_columns: Optional[Sequence[str]] = None,
        categorical_columns: Optional[Sequence[str]] = None,
        target: Optional[Union[str, Sequence[str]]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            data (pandas.DataFrame): DataFrame.
            task (str): One of "binary", "multiclass", "regression".
                Defaults to "binary".
            continuous_cols (sequence of str, optional): Sequence of names of
                continuous features (columns). Defaults to None.
            categorical_cols (sequence of str, optional): Sequence of names of
                categorical features (columns). Defaults to None.
            target (str, optional): If None, `np.zeros` is set as target.
                Defaults to None.
            transform (callable): Method of converting Tensor data.
                Defaults to None.
        """
        super().__init__()
        self.task = task
        self.num = data.shape[0]
        self.categorical_columns = categorical_columns if categorical_columns else []
        self.continuous_columns = continuous_columns if continuous_columns else []

        if self.continuous_columns:
            self.continuous = data[self.continuous_columns].values

        if self.categorical_columns:
            self.categorical = data[categorical_columns].values

        if target:
            self.target = data[target].values
            if isinstance(target, str):
                self.target = self.target.reshape(-1, 1)
        else:
            self.target = np.zeros((self.num, 1))

        self.transform = transform

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Args:
            idx (int): The index of the sample in the dataset.

        Returns:
            dict[str, Tensor]:
                The returned dict has the keys {"target", "continuous", "categorical"}
                and its values. If no continuous/categorical features, the returned value is `[]`.
        """
        if self.task == "multiclass":
            x = {
                "target": torch.LongTensor(self.target[idx]),
                "continuous": Tensor(self.continuous[idx])
                if self.continuous_columns
                else [],
                "categorical": LongTensor(self.categorical[idx])
                if self.categorical_columns
                else [],
            }
        elif self.task in {"binary", "regression"}:
            x = {
                "target": torch.Tensor(self.target[idx]),
                "continuous": Tensor(self.continuous[idx])
                if self.continuous_columns
                else [],
                "categorical": LongTensor(self.categorical[idx])
                if self.categorical_columns
                else [],
            }
        else:
            raise ValueError(
                f"task: {self.task} must be 'multiclass' or 'binary' or 'regression'"
            )

        if self.transform is not None:
            x = self.transform(x)
        return x


class TabularDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        train: DataFrame,
        num_categories: int = 0,
        categorical_columns: Optional[Sequence[str]] = None,
        continuous_columns: Optional[Sequence[str]] = None,
        target: Optional[Sequence[str]] = None,
        val: Optional[DataFrame] = None,
        test: Optional[DataFrame] = None,
        transform: Optional[Callable] = None,
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        task: str = "binary",
        dim_out: int = 1,
        batch_size: int = 128,
        num_workers: int = 3,
    ) -> None:
        """
        Args:
            train (`DataFrame`): DataFrame of train data.
            num_categories (int): All categories the dataset has. Defaults to 0.
            continuous_cols (sequence of str, optional): Sequence of names of
                continuous features (columns). Defaults to None.
            categorical_cols (sequence of str, optional): Sequence of names of
                categorical features (columns). Defaults to None.
            target (sequence of str, optional): Target features (columns) in training.
                If None, `np.zeros` is set as target. Defaults to None.
            validation (`DataFrame`, optional): DataFrame of validation data.
                If None, The returned value of `self.dataloader(split="val")` is None.
                Defaults to None.
            test: (`DataFrame`, optional): DataFrame of test data.
                If None, The returned value of `self.dataloader(split="test")` is None.
                Defaults to None.
            transform (callable, optional): Transformation applied to the Tensor.
                Defaults to None.
            train_sampler (`torch.utils.data.Sampler`, optional): Strategy of drawing
                samples. Defaults to None.
            task: (str): One of "binary", "multiclass", "regression".
                Defaults to "binary".
            dim_out (int): Dimension of outputs of models. For "binary" or "regression",
                `dim_out` should be 1. For "multiclass", `dim_out` should be the number
                of the classfication categories. Defaults to 1.
            batch_size (int): The number of samples for each batch. Defaults to 128.
            num_workers (int): The number of subprocess for loading data. Defaults to 3.
        """
        super().__init__()
        self.train = train
        self._num_categories = num_categories
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns
        self.target = target
        self.dim_out = dim_out

        self.val = val
        self.test = test
        self.transform = transform
        self.train_sampler = train_sampler
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def num_categories(self) -> int:
        return self._num_categories

    @property
    def num_continuous_features(self) -> int:
        return len(self.continuous_columns)

    @property
    def num_categorical_features(self) -> int:
        return len(self.categorical_columns)

    def dataloader(
        self,
        split: str,
        batch_size: Optional[int] = None,
        transform: Optional[Callable] = None,
    ) -> Optional[DataLoader]:
        """
        Args:
            split (str): One of "train", "val", "test".
                The returned value is a dataloader of `split`.
            batch_size (int): The number of samples for each batch.
                If the argument is set, `self.batch_size` will be overrided.
                Defaults to None.
            transform (callable): Transformation applied to the Tensor.
                If `transform` is not None, `self.transform` will be overrided.
                Defaults to None.

        Return:
            DataLoader
        """
        assert split in {"train", "val", "test"}
        if not hasattr(self, split):
            return None
        data = getattr(self, split)

        if split == "test":
            transform = None

        if transform is None:
            transform = self.transform

        dataset = TabularDataset(
            data=data,
            task=self.task,
            categorical_columns=self.categorical_columns,
            continuous_columns=self.continuous_columns,
            target=self.target,
            transform=transform,
        )
        return DataLoader(
            dataset,
            batch_size if batch_size is not None else self.batch_size,
            shuffle=True if split == "train" else False,
            num_workers=self.num_workers,
            sampler=self.train_sampler if split == "train" else None,
        )
