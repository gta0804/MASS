from math import sqrt
from typing import Any

import numpy as np
import torch
import torch.nn as nn

class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class AttentionLSTM(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers=1, batch_first: bool = True
    ):
        super(AttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=self.num_layers, batch_first=batch_first
        )

    def forward(self, x: torch.Tensor):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM 层
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # 使用最后一个隐藏状态作为查询向量
        query = hn[-1]  # (batch_size, hidden_dim)

        # 计算注意力权重
        scores = torch.bmm(out, query.unsqueeze(2)).squeeze(2)
        attn_weights = torch.softmax(scores, dim=1)

        # 加权求和
        context_vector = torch.sum(attn_weights.unsqueeze(2) * out, dim=1)

        return context_vector

class FullAttention(nn.Module):

    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(
        self, attention: FullAttention, d_model, n_heads, d_keys=None, d_values=None
    ):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        scale: Any | None = None,
        attention_dropout: float = 0.1,
    ):
        super(TransformerLayer, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.transformer = AttentionLayer(
            FullAttention(
                mask_flag=False, scale=scale, attention_dropout=attention_dropout
            ),
            d_model=model_dim,
            n_heads=num_heads,
        )

    def forward(self, x: torch.Tensor):
        # print(x.shape)
        # assert len(x.shape) == 2
        # assert x.shape[1] == self.model_dim
        assert self.model_dim % self.num_heads == 0
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 0)
        """Now x.shape == (batch_size, time_stamps, dim_model)  or(1, num_stocks, dim_model)"""
        x = x + self.transformer(x, x, x, attn_mask=None)
        return x


class DTML(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        scale: Any | None = None,
        attention_dropout: float = 0.1,
    ):
        super(DTML, self).__init__()
        self.feature_transform = nn.Linear(feature_dim, input_dim)
        self.attentive_lstm = AttentionLSTM(
            input_dim=input_dim, hidden_dim=hidden_dim, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.transformer = TransformerLayer(
            model_dim=hidden_dim,
            num_heads=num_heads,
            scale=scale,
            attention_dropout=attention_dropout,
        )
        self.output_transformation = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim), nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())

    def forward(self, x: torch.Tensor):
        """Forward process of DTML."""

        """
        Get Attentive lstm output(Time-axis attention).
        Num_stocks * Look_back_window * input_dim -> Num_stocks * hidden_dim
        """
        x = self.feature_transform(x)
        x = self.attentive_lstm(x)

        """Layer Normalization."""
        x = self.layer_norm(x)

        """Multi-Head Attention."""
        assert len(x.shape) == 2
        x = torch.squeeze(self.transformer(x))

        """Residual layer for output."""
        # x = self.output_transformation(x) + x
        out = self.output_layer(x)
        return out