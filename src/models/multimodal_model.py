"""
Multimodal fusion model combining image and tabular streams.

Fusion strategy: Late fusion (feature concatenation)
  ImageModel  → image_embedding  (IMAGE_FEATURE_DIM)
  TabularModel → tabular_embedding (TABULAR_FEATURE_DIM)
         ↓ concat
  FusionHead [Linear → BN → ReLU → Dropout → Linear] → num_classes

This is the primary model. The standalone ImageModel and TabularModel
are also provided as baselines.
"""

import torch
import torch.nn as nn

from src.config import (
    IMAGE_FEATURE_DIM, TABULAR_FEATURE_DIM,
    FUSION_HIDDEN_DIM, NUM_CLASSES, DROPOUT,
)
from src.models.image_model import ImageModel
from src.models.tabular_model import TabularModel


class MultimodalModel(nn.Module):
    """
    Late-fusion multimodal classifier.

    Args:
        tabular_input_dim: number of tabular features (after preprocessing)
        pretrained_image: use ImageNet pretrained backbone
    """

    def __init__(self, tabular_input_dim: int, pretrained_image: bool = True):
        super().__init__()

        # ── Sub-models (no classification heads — embeddings only) ─────────────
        self.image_branch = ImageModel(num_classes=0, pretrained=pretrained_image)
        self.tabular_branch = TabularModel(input_dim=tabular_input_dim, num_classes=0)

        fusion_input_dim = IMAGE_FEATURE_DIM + TABULAR_FEATURE_DIM

        # ── Fusion head ───────────────────────────────────────────────────────
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, FUSION_HIDDEN_DIM),
            nn.BatchNorm1d(FUSION_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(FUSION_HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        """
        image:   (B, 3, H, W)
        tabular: (B, tabular_input_dim)
        Returns: logits (B, NUM_CLASSES)
        """
        img_embed = self.image_branch.get_embedding(image)       # (B, IMAGE_FEATURE_DIM)
        tab_embed = self.tabular_branch.get_embedding(tabular)   # (B, TABULAR_FEATURE_DIM)
        fused = torch.cat([img_embed, tab_embed], dim=1)         # (B, combined)
        return self.fusion_head(fused)

    def freeze_image_backbone(self):
        self.image_branch.freeze_backbone()

    def unfreeze_image_backbone(self):
        self.image_branch.unfreeze_backbone()


def build_model(mode: str, tabular_input_dim: int = 0, pretrained: bool = True) -> nn.Module:
    """
    Factory function.

    mode:
      "multimodal" → MultimodalModel
      "image"      → ImageModel (standalone, with classifier head)
      "tabular"    → TabularModel (standalone, with classifier head)
    """
    if mode == "multimodal":
        assert tabular_input_dim > 0, "tabular_input_dim required for multimodal"
        return MultimodalModel(tabular_input_dim=tabular_input_dim, pretrained_image=pretrained)
    elif mode == "image":
        return ImageModel(num_classes=NUM_CLASSES, pretrained=pretrained)
    elif mode == "tabular":
        assert tabular_input_dim > 0, "tabular_input_dim required for tabular model"
        return TabularModel(input_dim=tabular_input_dim, num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown mode: {mode}")
