import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from deep_table.data.datasets.tabular_dataframe import TabularDataFrame


class Higgs(TabularDataFrame):
    """
    Link: https://www.kaggle.com/c/higgs-boson/overview
    """

    mirrors = []

    resources = []

    file = [("training.csv", "d7ee5a8f368cb2e33a21a53b39f4b69e")]

    dim_out = 1

    all_columns = [
        "DER_mass_MMC",
        "DER_mass_transverse_met_lep",
        "DER_mass_vis",
        "DER_pt_h",
        "DER_deltaeta_jet_jet",
        "DER_mass_jet_jet",
        "DER_prodeta_jet_jet",
        "DER_deltar_tau_lep",
        "DER_pt_tot",
        "DER_sum_pt",
        "DER_pt_ratio_lep_tau",
        "DER_met_phi_centrality",
        "DER_lep_eta_centrality",
        "PRI_tau_pt",
        "PRI_tau_eta",
        "PRI_tau_phi",
        "PRI_lep_pt",
        "PRI_lep_eta",
        "PRI_lep_phi",
        "PRI_met",
        "PRI_met_phi",
        "PRI_met_sumet",
        "PRI_jet_num",
        "PRI_jet_leading_pt",
        "PRI_jet_leading_eta",
        "PRI_jet_leading_phi",
        "PRI_jet_subleading_pt",
        "PRI_jet_subleading_eta",
        "PRI_jet_subleading_phi",
        "PRI_jet_all_pt",
        "Weight",
        "Label",
    ]

    continuous_columns = [
        "DER_mass_MMC",
        "DER_mass_transverse_met_lep",
        "DER_mass_vis",
        "DER_pt_h",
        "DER_deltaeta_jet_jet",
        "DER_mass_jet_jet",
        "DER_prodeta_jet_jet",
        "DER_deltar_tau_lep",
        "DER_pt_tot",
        "DER_sum_pt",
        "DER_pt_ratio_lep_tau",
        "DER_met_phi_centrality",
        "DER_lep_eta_centrality",
        "PRI_tau_pt",
        "PRI_tau_eta",
        "PRI_tau_phi",
        "PRI_lep_pt",
        "PRI_lep_eta",
        "PRI_lep_phi",
        "PRI_met",
        "PRI_met_phi",
        "PRI_met_sumet",
        "PRI_jet_num",
        "PRI_jet_leading_pt",
        "PRI_jet_leading_eta",
        "PRI_jet_leading_phi",
        "PRI_jet_subleading_pt",
        "PRI_jet_subleading_eta",
        "PRI_jet_subleading_phi",
        "PRI_jet_all_pt",
    ]

    categorical_columns = []

    target_columns = ["Label"]

    task = "binary"

    def __init__(self, root: str, download: bool = False) -> None:
        """
        This dataset must be downloaded and moved to
        Higgs/raw/training.csv, Higgs/raw/test.csv
        """
        super().__init__(root=root, download=False)
        if self._check_exists(self.file):
            print("Using downloaded and verified file: ", self.file)
            return

    def download(self) -> None:
        pass

    def raw_dataframe(self, train: bool = True) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.raw_folder, "training.csv"), header=0, index_col=0
        )
        df_train, df_test = train_test_split(
            df, stratify=df[self.target_columns], test_size=0.25, random_state=0
        )
        df = df_train if train else df_test
        return df

    def processed_dataframes(self, *args, **kwargs) -> Dict[str, pd.DataFrame]:
        df_train = self.raw_dataframe(train=True)
        df_train = df_train.drop(columns="Weight")
        df_test = self.raw_dataframe(train=False)
        df_test = df_test.drop(columns="Weight")

        df_train, df_val = train_test_split(
            df_train, stratify=df_train[self.target_columns], **kwargs
        )

        # preprocessing
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

        # if value == -999.0, imputated by avg of the column
        df_train = df_train.replace("-999", np.nan)
        df_val = df_val.replace("-999", np.nan)
        df_test = df_test.replace("-999", np.nan)
        df_train = pd.DataFrame(
            data=SimpleImputer().fit_transform(df_train),
            index=df_train.index,
            columns=df_train.columns,
        )
        df_val = pd.DataFrame(
            data=SimpleImputer().fit_transform(df_val),
            index=df_val.index,
            columns=df_test.columns,
        )
        df_test = pd.DataFrame(
            data=SimpleImputer().fit_transform(df_test),
            index=df_test.index,
            columns=df_test.columns,
        )

        return {
            "train": df_train,
            "val": df_val,
            "test": df_test,
        }

    def _label_encoder(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.target_columns] = df[self.target_columns].replace("s", 1)
        df[self.target_columns] = df[self.target_columns].replace("b", 0)
        return df
