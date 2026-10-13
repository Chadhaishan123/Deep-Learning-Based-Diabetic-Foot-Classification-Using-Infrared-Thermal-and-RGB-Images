"""Multimodal Fusion Architectures: Double-Encoder ResNet (DE-ResNet) and Late-Fusion ResNet (LF-ResNet)
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Section 2.6.1 & Figure 3
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights


class LF_ResNet(nn.Module):
    """Late-Fusion ResNet (LF-ResNet).
    Processes RGB and IR modalities through independent backbones, concatenates
    the final global average pooled feature vectors, and feeds them to the classifier.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            bb_rgb = resnet18(weights=weights)
            bb_ir = resnet18(weights=weights)
            feat_dim = 512
        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            bb_rgb = resnet50(weights=weights)
            bb_ir = resnet50(weights=weights)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove original fc heads
        self.encoder_rgb = nn.Sequential(
            bb_rgb.conv1,
            bb_rgb.bn1,
            bb_rgb.relu,
            bb_rgb.maxpool,
            bb_rgb.layer1,
            bb_rgb.layer2,
            bb_rgb.layer3,
            bb_rgb.layer4,
            bb_rgb.avgpool,
        )
        self.encoder_ir = nn.Sequential(
            bb_ir.conv1,
            bb_ir.bn1,
            bb_ir.relu,
            bb_ir.maxpool,
            bb_ir.layer1,
            bb_ir.layer2,
            bb_ir.layer3,
            bb_ir.layer4,
            bb_ir.avgpool,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(feat_dim * 2, num_classes),
        )

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        feat_rgb = torch.flatten(self.encoder_rgb(rgb), 1)
        feat_ir = torch.flatten(self.encoder_ir(ir), 1)
        fused = torch.cat([feat_rgb, feat_ir], dim=1)
        return self.classifier(fused)


class DE_ResNet(nn.Module):
    """Double-Encoder ResNet (DE-ResNet).
    Processes RGB and IR through two independent encoder branches, fusing intermediate
    feature maps at each residual stage (layers 1-4) via lateral residual connections
    as depicted in Fig. 3 (inspired by Bouallal et al. 2022).
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if (pretrained and backbone == "resnet18") else None
        
        # Branch 1: RGB
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Branch 2: IR
        self.ir_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        base_m = resnet18(weights=weights)
        
        self.rgb_l1 = base_m.layer1
        self.ir_l1 = resnet18(weights=weights).layer1

        self.rgb_l2 = base_m.layer2
        self.ir_l2 = resnet18(weights=weights).layer2

        self.rgb_l3 = base_m.layer3
        self.ir_l3 = resnet18(weights=weights).layer3

        self.rgb_l4 = base_m.layer4
        self.ir_l4 = resnet18(weights=weights).layer4

        # Lateral fusion projection layers at each stage
        self.fuse_l1 = nn.Conv2d(64 * 2, 64, kernel_size=1)
        self.fuse_l2 = nn.Conv2d(128 * 2, 128, kernel_size=1)
        self.fuse_l3 = nn.Conv2d(256 * 2, 256, kernel_size=1)
        self.fuse_l4 = nn.Conv2d(512 * 2, 512, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(512, num_classes),
        )

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        # Stem
        x_rgb = self.rgb_stem(rgb)
        x_ir = self.ir_stem(ir)

        # Stage 1 + Lateral Fusion
        x_rgb = self.rgb_l1(x_rgb)
        x_ir = self.ir_l1(x_ir)
        f1 = self.fuse_l1(torch.cat([x_rgb, x_ir], dim=1))
        x_rgb = x_rgb + f1
        x_ir = x_ir + f1

        # Stage 2 + Lateral Fusion
        x_rgb = self.rgb_l2(x_rgb)
        x_ir = self.ir_l2(x_ir)
        f2 = self.fuse_l2(torch.cat([x_rgb, x_ir], dim=1))
        x_rgb = x_rgb + f2
        x_ir = x_ir + f2

        # Stage 3 + Lateral Fusion
        x_rgb = self.rgb_l3(x_rgb)
        x_ir = self.ir_l3(x_ir)
        f3 = self.fuse_l3(torch.cat([x_rgb, x_ir], dim=1))
        x_rgb = x_rgb + f3
        x_ir = x_ir + f3

        # Stage 4 + Lateral Fusion
        x_rgb = self.rgb_l4(x_rgb)
        x_ir = self.ir_l4(x_ir)
        f4 = self.fuse_l4(torch.cat([x_rgb, x_ir], dim=1))

        # Final pooling & classification
        pooled = torch.flatten(self.avgpool(f4), 1)
        return self.classifier(pooled)
