import gzip
import os
import shutil
from typing import Dict
from urllib.error import URLError

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision.datasets.utils import download_url

from deep_table.data.datasets.tabular_dataframe import TabularDataFrame
from deep_table.preprocess import CategoryPreprocessor


class ForestCoverType(TabularDataFrame):
    """Forest CoverType dataset
    Link: https://archive.ics.uci.edu/ml/datasets/covertype


    Name                                     Data Type    Measurement                       Description

    Elevation                               quantitative    meters                       Elevation in meters
    Aspect                                  quantitative    azimuth                      Aspect in degrees azimuth
    Slope                                   quantitative    degrees                      Slope in degrees
    Horizontal_Distance_To_Hydrology        quantitative    meters                       Horz Dist to nearest surface water features
    Vertical_Distance_To_Hydrology          quantitative    meters                       Vert Dist to nearest surface water features
    Horizontal_Distance_To_Roadways         quantitative    meters                       Horz Dist to nearest roadway
    Hillshade_9am                           quantitative    0 to 255 index               Hillshade index at 9am, summer solstice
    Hillshade_Noon                          quantitative    0 to 255 index               Hillshade index at noon, summer soltice
    Hillshade_3pm                           quantitative    0 to 255 index               Hillshade index at 3pm, summer solstice
    Horizontal_Distance_To_Fire_Points      quantitative    meters                       Horz Dist to nearest wildfire ignition points
    Wilderness_Area (4 binary columns)      qualitative     0 (absence) or 1 (presence)  Wilderness area designation
    Soil_Type (40 binary columns)           qualitative     0 (absence) or 1 (presence)  Soil Type designation
    Cover_Type (7 types)                    integer         1 to 7                       Forest Cover Type designation

    Forest Cover Type Classes:
    1 -- Spruce/Fir
    2 -- Lodgepole Pine
    3 -- Ponderosa Pine
    4 -- Cottonwood/Willow
    5 -- Aspen
    6 -- Douglas-fir
    7 -- Krummholz
    """

    mirrors = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/",
    ]

    resources = [
        ("covtype.data.gz", "99670d8d942f09d459c7d4486fca8af5"),
    ]

    file = [("covtype.data", "71df19898bd3e11db15ae0faf4159f2c")]

    dim_out = 7

    all_columns = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
        *[f"Wilderness_Area_{i}" for i in range(4)],
        *[f"Soil_Type_{i}" for i in range(40)],
        "Cover_Type",
    ]

    continuous_columns = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]

    categorical_columns = [
        *[f"Wilderness_Area_{i}" for i in range(4)],
        *[f"Soil_Type_{i}" for i in range(40)],
    ]

    target_columns = ["Cover_Type"]

    task = "multiclass"

    def __init__(self, root: str, download: bool = True) -> None:
        super().__init__(root=root, download=download)

    def download(self) -> None:
        if self._check_exists(self.file):
            print("Using downloaded and verified file: " + self.file[0][0])
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    download_url(url, root=self.raw_folder, filename=filename, md5=md5)
                    with gzip.open(
                        os.path.join(self.raw_folder, "covtype.data.gz"), "rb"
                    ) as f_in:
                        with open(
                            os.path.join(self.raw_folder, "covtype.data"), "wb"
                        ) as f_out:
                            shutil.copyfileobj(f_in, f_out)
                except URLError as error:
                    print("Failed to download (trying next):\n{}".format(error))
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))

    def raw_dataframe(self, train: bool = True) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.raw_folder, self.file[0][0]), header=None)
        df.columns = self.all_columns
        df = self._label_encoder(df)
        if train:
            df = df[:15120]
        else:
            df = df[15120:]
        return df

    def processed_dataframes(self, *args, **kwargs) -> Dict[str, pd.DataFrame]:
        df_train = self.raw_dataframe(train=True)
        df_test = self.raw_dataframe(train=False)

        df_train, df_val = train_test_split(
            df_train, stratify=df_train[self.target_columns], **kwargs
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

        # target labels to [0, 6]
        df_train[self.target_columns] -= 1
        df_val[self.target_columns] -= 1
        df_test[self.target_columns] -= 1

        return {
            "train": df_train,
            "val": df_val,
            "test": df_test,
        }

    def _label_encoder(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.target_columns] = df[self.target_columns].astype(int)
        return df
