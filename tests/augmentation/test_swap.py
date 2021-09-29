import pytest
import torch

from deep_table.augmentation.swap import ColumnSwap, RandomizedSwap, RowSwap


@pytest.mark.parametrize("class_swap", [ColumnSwap, RowSwap, RandomizedSwap])
def test_swap(class_swap):
    prob = 0.2

    # Test Case 1.
    input_tensor = torch.randn((1024, 1024))
    swap = class_swap(prob=prob)
    output_tensor, mask = swap(input_tensor.clone())

    assert input_tensor.shape == output_tensor.shape
    diff_size = torch.ne(input_tensor, output_tensor).sum() / input_tensor.numel()
    assert diff_size == pytest.approx(prob, abs=0.05)


@pytest.mark.parametrize("class_swap", [RandomizedSwap, RowSwap, ColumnSwap])
def test_swap_overlap(class_swap):
    prob = 0.2

    # Test Case 1. shape is the same as when overlap=False
    input_tensor = torch.randn((1024, 1024))
    swap = class_swap(prob=prob, overlap=False)
    swap_overlap = class_swap(prob=prob, overlap=True)
    output_tensor, _ = swap(input_tensor.clone())
    output_tensor_overlap, _ = swap_overlap(input_tensor.clone())
    assert output_tensor.shape == output_tensor_overlap.shape
