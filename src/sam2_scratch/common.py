"""Small building blocks shared by several SAM 2 modules.

Parameter names deliberately match the released checkpoint keys so that every module can
be loaded with a strict ``load_state_dict`` (see ``weights.py``).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    """Plain multi-layer perceptron: Linear layers with an activation between them.

    Weights live in ``layers.{i}`` to match the checkpoint. ``hidden_dim`` is used for every
    hidden layer; the last layer has no activation unless ``sigmoid_output`` is set.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: type[nn.Module] = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n_in, n_out)
            for n_in, n_out in zip([input_dim, *hidden], [*hidden, output_dim], strict=True)
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.act(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an NCHW tensor (ConvNeXt style)."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def sine_pe_1d(positions: Tensor, dim: int, temperature: float = 10000.0) -> Tensor:
    """1-D sinusoidal embedding of scalar positions: ``[N] -> [N, dim]`` (sin, cos halves)."""
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=positions.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos = positions.float().unsqueeze(-1) / dim_t
    return torch.cat([pos.sin(), pos.cos()], dim=-1)
