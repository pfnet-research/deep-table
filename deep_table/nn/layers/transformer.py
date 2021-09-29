from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .utils import get_activation_fn


class MultiheadAttention(nn.Module):
    def __init__(
        self, emb_dim: int, n_heads: int = 8, dim_head: int = 16, dropout: float = 0.0
    ) -> None:
        """
        Args:
            emb_dim (int): Size of embedding for each sample.
            n_heads (int): Parallel attention heads. Defaults to 8.
            dim_head (int): Dimension of each attention head. Defaults to 16.
            drop_out (float): Dropout value. Defaults to 0.0.
        """
        super(MultiheadAttention, self).__init__()
        inner_dim = dim_head * n_heads
        self.n_heads = n_heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(emb_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.n_heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_head: Optional[int] = None,
        dim_feedforward: int = 2048,
        dropout=0.1,
        activation="relu",
    ) -> None:
        """
        Args:
            d_model (int): Total dimension of the model.
            n_heads (int): Parallel attention heads. Defaults to 8.
            dim_head (int): Dimension of each attention head. Defaults to None.
            dim_feedforward (int): The dimension of feedforward network model.
                Defaults to 2048.
            drop_out (int): Dropout value. Defaults to 0.1
            activation (str): Activation in the intermediate layer. Defaults to "ReLU".
        """
        super(TransformerEncoderLayer, self).__init__()
        if dim_head is None:
            dim_head = d_model // n_heads
            assert (
                dim_head * n_heads == d_model
            ), "embed_dim must be divisible by num_heads"

        self.self_attn = MultiheadAttention(d_model, n_heads, dim_head, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def forward(self, src: Tensor) -> Tensor:
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class ColRowTransformer(nn.Module):
    """Transformer block to get contextual represenation.

    References:
        SAINT: Improved Neural Networks for Tabular Data via Row Attention and
        Contrastive Pre-Training (https://arxiv.org/abs/2106.01342)
    """

    def __init__(
        self,
        dim_embed: int,
        n_heads: int,
        dim_sa_head: int,
        dim_is_head: int,
        num_features: int,
        dim_feedforward: int,
        dropout: float,
        attn_type: str = "colrow",
    ) -> None:
        """
        dim_embed (int): Size of embedding for each sample.
        n_heads (int): Parallel attention heads.
        dim_sa_head (int): Dimension of self attention head.
        dim_is_head (int): Dimension of inter-sample attention head.
        num_features (int): The number of features transformers taken.
            Usually the number of categorical and continuous features.
        dim_feedforward (int): The dimension of feedforward network model.
        dropout (float): Dropout value for transformer blocks.
        attn_type (str): Attention type. If "colrow", inter-sample attention will be
            applied to the tensor after self-attention layer. Defaults to "colrow".
        """
        super().__init__()
        self.dim_embed = dim_embed
        self.num_features = num_features

        self.self_attention = TransformerEncoderLayer(
            dim_embed,
            n_heads,
            dim_head=dim_sa_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
        )
        self.intersample_attention = (
            TransformerEncoderLayer(
                dim_embed * num_features,
                n_heads,
                dim_head=dim_is_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
            )
            if attn_type == "colrow"
            else nn.Identify()
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, _ = x.size()
        x = self.self_attention(x)
        x = x.view(1, batch_size, self.num_features * self.dim_embed)
        x = self.intersample_attention(x)
        x = x.view(batch_size, self.num_features, self.dim_embed)
        return x
