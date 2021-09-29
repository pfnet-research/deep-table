import numpy as np
import pandas as pd
import pytest

from deep_table.data.data_module import TabularDatamodule


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return

    # no marker
    skip = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
    return


@pytest.fixture(scope="session")
def dummy_dataframe():
    """
        categorical_column  continuous_column  target
    0                    0           0.675088       0
    1                    0           0.211024       1
    2                    0          -1.764863       0
    ..                 ...                ...     ...
    97                   0           0.529406       1
    98                   1          -0.444050       0
    99                   1           0.295234       1
    """
    categorical_data = np.where(np.random.randn(100, 1) > 0, 0, 1)
    continuous_data = np.random.randn(100, 1)
    target_data = np.where(np.random.randn(100, 1) > 0, 1, 0).astype(int)
    data = np.hstack([categorical_data, continuous_data, target_data])
    df = pd.DataFrame(data)
    df.columns = ["categorical_column", "continuous_column", "target"]
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)
    df.iloc[:, 1] = df.iloc[:, 1].astype(float)
    df.iloc[:, 2] = df.iloc[:, 2].astype(int)
    return df


@pytest.fixture(scope="session")
def dummy_datamodule(dummy_dataframe):
    datamodule = TabularDatamodule(
        train=dummy_dataframe[:35],
        val=dummy_dataframe[35:70],
        test=dummy_dataframe[70:],
        num_categories=2,
        continuous_columns=["continuous_column"],
        categorical_columns=["categorical_column"],
        target=["target"],
    )
    return datamodule
