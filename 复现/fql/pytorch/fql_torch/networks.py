"""State MLPs matching the pinned author's Flax FQL network definitions."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MLP(nn.Module):
    """Dense → approximate GELU → optional LayerNorm, except the output layer."""

    def __init__(self, input_dim: int, sizes: Sequence[int], layer_norm: bool = False):
        super().__init__()
        dims = (input_dim, *sizes)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(
            nn.LayerNorm(width, eps=1e-6) for width in sizes[:-1]
        ) if layer_norm else nn.ModuleList()
        for layer in self.layers:
            # Flax variance_scaling(1, 'fan_avg', 'uniform') = Xavier uniform.
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, inputs: Tensor, detach_parameters: bool = False) -> Tensor:
        x = inputs
        for i, layer in enumerate(self.layers):
            weight = layer.weight.detach() if detach_parameters else layer.weight
            bias = layer.bias.detach() if detach_parameters else layer.bias
            x = F.linear(x, weight, bias)
            if i < len(self.layers) - 1:
                # flax.linen.gelu defaults to the tanh approximation.
                x = F.gelu(x, approximate="tanh")
                if self.norms:
                    norm = self.norms[i]
                    scale = norm.weight.detach() if detach_parameters else norm.weight
                    shift = norm.bias.detach() if detach_parameters else norm.bias
                    x = F.layer_norm(x, norm.normalized_shape, scale, shift, norm.eps)
        return x


class EnsembleDense(nn.Module):
    """Independent Dense parameters, with the ensemble axis batched on the GPU."""

    def __init__(self, ensembles: int, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ensembles, output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(ensembles, output_dim))
        # Calling Xavier on this 3-D tensor would count the ensemble axis as fan.
        nn.init.uniform_(self.weight, -math.sqrt(6 / (input_dim + output_dim)),
                         math.sqrt(6 / (input_dim + output_dim)))


class EnsembleNorm(nn.Module):
    def __init__(self, ensembles: int, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ensembles, width))
        self.bias = nn.Parameter(torch.zeros(ensembles, width))
        self.eps = 1e-6

    def forward(self, inputs: Tensor, detach_parameters: bool = False) -> Tensor:
        scale = self.weight.detach() if detach_parameters else self.weight
        shift = self.bias.detach() if detach_parameters else self.bias
        normalized = F.layer_norm(inputs, (inputs.shape[-1],), eps=self.eps)
        return normalized * scale[:, None, :] + shift[:, None, :]


class EnsembleMLP(nn.Module):
    def __init__(self, input_dim: int, sizes: Sequence[int], ensembles: int = 2,
                 layer_norm: bool = True):
        super().__init__()
        dims = (input_dim, *sizes)
        self.layers = nn.ModuleList(
            EnsembleDense(ensembles, a, b) for a, b in zip(dims[:-1], dims[1:])
        )
        self.norms = nn.ModuleList(
            EnsembleNorm(ensembles, width) for width in sizes[:-1]
        ) if layer_norm else nn.ModuleList()

    def forward(self, inputs: Tensor, detach_parameters: bool = False) -> Tensor:
        x = inputs
        for i, layer in enumerate(self.layers):
            weight = layer.weight.detach() if detach_parameters else layer.weight
            bias = layer.bias.detach() if detach_parameters else layer.bias
            transposed = weight.transpose(-1, -2)
            x = torch.matmul(x, transposed) if i == 0 else torch.bmm(x, transposed)
            x = x + bias[:, None, :]
            if i < len(self.layers) - 1:
                x = F.gelu(x, approximate="tanh")
                if self.norms:
                    x = self.norms[i](x, detach_parameters=detach_parameters)
        return x


class Value(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int,
                 hidden_dims: Sequence[int], layer_norm: bool = True):
        super().__init__()
        self.value_net = EnsembleMLP(observation_dim + action_dim,
                                     (*hidden_dims, 1), layer_norm=layer_norm)

    def forward(self, observations: Tensor, actions: Tensor,
                detach_parameters: bool = False) -> Tensor:
        inputs = torch.cat((observations, actions), dim=-1)
        batch_shape = inputs.shape[:-1]
        values = self.value_net(inputs.reshape(-1, inputs.shape[-1]),
                                detach_parameters=detach_parameters)
        return values.reshape(2, *batch_shape)


class ActorVectorField(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int,
                 hidden_dims: Sequence[int], layer_norm: bool = False,
                 with_time: bool = False):
        super().__init__()
        self.with_time = with_time
        self.mlp = MLP(observation_dim + action_dim + int(with_time),
                       (*hidden_dims, action_dim), layer_norm=layer_norm)

    def forward(self, observations: Tensor, actions: Tensor,
                times: Tensor | None = None) -> Tensor:
        if self.with_time != (times is not None):
            raise ValueError("BC flow requires times; one-step actor must omit times.")
        inputs = (observations, actions) if times is None else (observations, actions, times)
        return self.mlp(torch.cat(inputs, dim=-1))
