import torch

from deep_table.augmentation.mixup import mixup


def test_mixup_fn():
    x = torch.tensor([1, 2, 3])
    x_b = torch.tensor([0, 0, 0])
    alpha = 0.5
    assert torch.equal(mixup(x, x_b, alpha), torch.tensor([0.5, 1.0, 1.5]))

    x_b = torch.tensor([3, 2, 1])
    assert torch.equal(
        mixup(x, x_b, alpha),
        torch.tensor(
            [
                2.0,
                2.0,
                2,
            ]
        ),
    )
