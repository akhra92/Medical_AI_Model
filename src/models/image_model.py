"""
Image classification model using a pretrained EfficientNet-B3 backbone.

Architecture:
  EfficientNet-B3 (pretrained on ImageNet)
    → Global Average Pool
    → Dropout
    → Linear projection → image_embedding (IMAGE_FEATURE_DIM)
    → (optional) classification head → num_classes

When used in multimodal fusion the classification head is omitted and only
the embedding is returned.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from src.config import IMAGE_FEATURE_DIM, NUM_CLASSES, DROPOUT, IMAGE_BACKBONE


class ImageModel(nn.Module):
    """
    Pretrained CNN feature extractor + optional classifier.

    Args:
        num_classes: if > 0, adds a final FC layer for standalone classification.
                     Set to 0 when used as part of multimodal fusion.
        pretrained: use ImageNet pretrained weights
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone ──────────────────────────────────────────────────────────
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)

        # Remove the built-in classifier; keep features only
        in_features = backbone.classifier[1].in_features   # 1536 for EfficientNet-B3
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # ── Projection head ───────────────────────────────────────────────────
        self.projection = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features, IMAGE_FEATURE_DIM),
            nn.BatchNorm1d(IMAGE_FEATURE_DIM),
            nn.ReLU(inplace=True),
        )

        # ── Optional classification head ──────────────────────────────────────
        if num_classes > 0:
            self.classifier = nn.Linear(IMAGE_FEATURE_DIM, num_classes)
        else:
            self.classifier = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        Returns:
          logits (B, num_classes) if num_classes > 0
          embedding (B, IMAGE_FEATURE_DIM) otherwise
        """
        feat = self.backbone(x)          # (B, 1536)
        embed = self.projection(feat)    # (B, IMAGE_FEATURE_DIM)
        if self.classifier is not None:
            return self.classifier(embed)
        return embed

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Always returns the embedding regardless of num_classes setting."""
        feat = self.backbone(x)
        return self.projection(feat)

    def freeze_backbone(self):
        """Freeze backbone weights (fine-tune only the head)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all weights."""
        for param in self.backbone.parameters():
            param.requires_grad = True
