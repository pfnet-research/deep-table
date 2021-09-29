from deep_table.data.datasets import TabularDataFrame


def test_cat_cardinalities(dummy_dataframe, mocker):
    tabular_dataframe = mocker.Mock(TabularDataFrame)
    mocker.patch.object(
        tabular_dataframe,
        "raw_dataframe",
        side_effect=lambda train: dummy_dataframe.iloc[:70]
        if train
        else dummy_dataframe.iloc[70:],
    )
    mocker.patch.object(
        tabular_dataframe,
        "cat_cardinalities",
        side_effect=TabularDataFrame.cat_cardinalities,
    )
    mocker.patch.object(
        tabular_dataframe, "categorical_columns", ["categorical_column"], create=True
    )

    assert tabular_dataframe.cat_cardinalities(tabular_dataframe, use_unk=False) == [2]
    assert tabular_dataframe.cat_cardinalities(tabular_dataframe, use_unk=True) == [3]
