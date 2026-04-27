"""
Tabular MLP model for clinical + molecular features.

Architecture:
  Input (tabular_dim)
    → [Linear → BatchNorm → ReLU → Dropout] × n_layers
    → tabular_embedding (TABULAR_FEATURE_DIM)
    → (optional) classification head → num_classes

When used in multimodal fusion the classification head is omitted.
"""

import torch
import torch.nn as nn
from typing import List

from src.config import (
    TABULAR_HIDDEN_DIMS, TABULAR_FEATURE_DIM,
    NUM_CLASSES, DROPOUT,
)


def _build_mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
    )


class TabularModel(nn.Module):
    """
    Multi-layer perceptron for tabular data.

    Args:
        input_dim: number of input features (set after preprocessing)
        num_classes: if > 0, adds classification head; 0 → return embedding only
    """

    def __init__(self, input_dim: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        # ── MLP hidden layers ─────────────────────────────────────────────────
        dims = [input_dim] + TABULAR_HIDDEN_DIMS + [TABULAR_FEATURE_DIM]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(_build_mlp_block(dims[i], dims[i + 1], DROPOUT))
        self.mlp = nn.Sequential(*layers)

        # ── Optional classification head ──────────────────────────────────────
        if num_classes > 0:
            self.classifier = nn.Linear(TABULAR_FEATURE_DIM, num_classes)
        else:
            self.classifier = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim)
        Returns logits (B, num_classes) or embedding (B, TABULAR_FEATURE_DIM)
        """
        embed = self.mlp(x)
        if self.classifier is not None:
            return self.classifier(embed)
        return embed

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
