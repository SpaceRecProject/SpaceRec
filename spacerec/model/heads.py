from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from torch import Tensor, nn


class REGISTRY_KEYS:
    OUTPUT_EMBEDDING = "output_embedding"
    OUTPUT_PREDICTION = "output_prediction"


class TemperatureSoftmax(nn.Module):
    """Temperature-scaled softmax used by the formal SpaceRec type head."""

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = max(float(temperature), 1e-6)

    def forward(self, logits: Tensor) -> Tensor:
        return torch.softmax(logits / self.temperature, dim=-1)


class BaseGenePredictor(nn.Module, ABC):
    """Formal SpaceRec base gene predictor."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        final_activation: str = "softplus",
        hidden_dim: Optional[List[int]] = None,
        dropout_rate: Optional[float] = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else []
        self.dropout_rate = dropout_rate

        if final_activation == "relu":
            self.final_activation_layer = nn.ReLU()
        elif final_activation == "softplus":
            self.final_activation_layer = nn.Softplus(beta=20)
        elif final_activation == "sigmoid":
            self.final_activation_layer = nn.Sigmoid()
        elif final_activation == "softmax":
            self.final_activation_layer = nn.Softmax(dim=-1)
        elif final_activation == "identity":
            self.final_activation_layer = nn.Identity()
        else:
            raise ValueError(
                f"final activation layer must be one of [relu, softplus, sigmoid, softmax, identity], got {final_activation}"
            )

        self.input_dims = [input_dim] + self.hidden_dim
        self.output_dims = self.hidden_dim + [output_dim]

        self.model = self._create_model()

    def _create_model(self) -> nn.Module:
        return nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_channel, out_channel),
                    nn.LeakyReLU(),
                    nn.Dropout(self.dropout_rate),
                )
                for in_channel, out_channel in zip(
                    self.input_dims[:-1], self.output_dims[:-1]
                )
            ]
            + [
                nn.Sequential(
                    nn.Linear(self.input_dims[-1], self.output_dims[-1]),
                    self.final_activation_layer,
                )
            ]
        )

    @abstractmethod
    def forward(
        self, batch_data: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        pass


class GeneHead(BaseGenePredictor):
    """Formal SpaceRec per-cell gene-expression predictor."""

    def forward(
        self, batch_data: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        out = self.model(batch_data[REGISTRY_KEYS.OUTPUT_EMBEDDING])
        return {
            REGISTRY_KEYS.OUTPUT_PREDICTION: out,
        }


class ResidualTypeHead(nn.Module):
    """Formal SpaceRec residual MLP cell-type head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_cell_types: int,
        dropout: float,
        temperature: float,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(input_dim, num_cell_types),
            TemperatureSoftmax(temperature),
        )

    def forward(self, features: Tensor) -> Tensor:
        residual_features = features + self.block(self.norm(features))
        return self.classifier(residual_features)
