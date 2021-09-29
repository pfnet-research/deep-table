import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


class InfoNCELoss(_Loss):
    """Info NCE Loss. A type of contrastive loss function used for self-supervised learning.

    References:
        A. Oord, Y. Li, and O. Vinyals,
        "Representation Learning with Contrastive Predictive Coding,"
        ArXiv:1807.03748 [cs.LG], 2018. <https://arxiv.org/abs/1807.03748v2>
    """

    def __init__(self, reduction: str = "sum") -> None:
        """
        Args:
            reduction (str)
        """
        super().__init__(reduction=reduction)
        self.reduction = reduction

    def forward(self, z_origin: Tensor, z_noisy: Tensor, t: float = 0.7) -> Tensor:
        sim = cos_sim_matrix(z_origin, z_noisy)
        exp_sim = torch.exp(sim / t)
        loss = -torch.log(torch.diagonal(exp_sim) / exp_sim.sum(1))
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss


def cos_sim_matrix(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    a_n, b_n = a.norm(dim=1), b.norm(dim=1)
    a_norm = a / torch.clamp(a_n.unsqueeze(1), min=eps)
    b_norm = b / torch.clamp(b_n.unsqueeze(1), min=eps)
    sim_matrix = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_matrix
