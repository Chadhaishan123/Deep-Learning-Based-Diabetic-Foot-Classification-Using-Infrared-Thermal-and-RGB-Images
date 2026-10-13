"""ResNet Models for Diabetic Foot Classification
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Section 2.5.1
Supports ResNet-18 (11.7M params) and ResNet-50 (23.9M params).
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights


def get_resnet_model(
    model_name: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Build ResNet18 or ResNet50 classifier with customized classification head and dropout."""
    name = model_name.lower()
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
    elif name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
    else:
        raise ValueError(f"Unsupported ResNet model name: {model_name}")

    if dropout and dropout > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model
