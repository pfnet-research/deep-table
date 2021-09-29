import numpy as np

from deep_table.utils import _sigmoid, _softmax, mean_absolute_percentage_error


def test_sigmoid():
    input_array = np.array([[2, 0, -2], [1, 0, -1]])

    output_array = _sigmoid(input_array)
    target_array = np.array(
        [
            [0.8807970, 0.5, 0.1192029],
            [0.7310585, 0.5, 0.2689414],
        ]
    )
    assert np.allclose(output_array, target_array)


def test_softmax():
    input_array = np.random.randint(1, 100) * np.ones((2, 3))
    output_array = _softmax(input_array)
    target_array = np.ones((2, 3)) / 3
    assert np.allclose(output_array, target_array)


def test_mean_absolute_percentage_error():
    pred = np.array([0.9])
    true = np.array([1.0])
    mape = mean_absolute_percentage_error(y_pred=pred, y_true=true)
    assert np.isclose(mape, 0.1)

    pred = np.array([1.0])
    true = np.array([0.0])
    mape = mean_absolute_percentage_error(y_pred=pred, y_true=true)
    assert np.isclose(mape, 1e10)
