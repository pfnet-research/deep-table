import torch

from deep_table.augmentation.utils import masking


def test_masking_2d():
    """
    test masking for row features whose dim = 2D.
    """
    x = torch.ones(3, 2)
    x_mask = -torch.ones(3, 2)
    mask = torch.LongTensor(
        [
            [1, 1],
            [1, 0],
            [0, 1],
        ]
    )
    correct_tensor = torch.Tensor(
        [
            [1, 1],
            [1, -1],
            [-1, 1],
        ]
    )

    assert torch.equal(masking(x, x_mask, mask), correct_tensor)


def test_masking_3d():
    """
    test masking for embedded features whose dim = 3D.
    """
    x = torch.ones(2, 2, 3)
    x_mask = -torch.ones(2, 2, 3)
    mask = torch.LongTensor([[1, 0], [0, 1]])
    correct_tensor = torch.Tensor(
        [[[1, 1, 1], [-1, -1, -1]], [[-1, -1, -1], [1, 1, 1]]]
    )

    assert torch.equal(masking(x, x_mask, mask), correct_tensor)
