import pytest
import torch

from deep_table.augmentation import Cutmix


def test_cutmix():
    prob = 0.2

    # Test Case 1.
    input_tensor = torch.randn((1024, 1024))
    cutmix = Cutmix(mask_prob=prob)
    output_tensor = cutmix(input_tensor.clone())[0]

    assert input_tensor.shape == output_tensor.shape
    diff_size = torch.ne(input_tensor, output_tensor).sum() / input_tensor.numel()
    assert diff_size == pytest.approx(prob, abs=0.05)
