"""Multimodal Fusion Architectures:
1. LF-ResNet (Late-Fusion ResNet) - Baseline Paper (Muscillo et al., 2026)
2. DE-ResNet (Double-Encoder ResNet) - Baseline Paper (Muscillo et al., 2026)
3. CMA-ResNet (Cross-Modal Attention & Gated Multimodal ResNet) - NOVELTY ARCHITECTURE
   Addresses the 4-class risk staging bottleneck by dynamically weighting thermal hotspot cues
   and RGB morphological textures via Bidirectional Spatial-Channel Cross-Attention and Gated Fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        pretrained: bool = False,
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

        self.encoder_rgb = nn.Sequential(
            bb_rgb.conv1, bb_rgb.bn1, bb_rgb.relu, bb_rgb.maxpool,
            bb_rgb.layer1, bb_rgb.layer2, bb_rgb.layer3, bb_rgb.layer4,
            bb_rgb.avgpool,
        )
        self.encoder_ir = nn.Sequential(
            bb_ir.conv1, bb_ir.bn1, bb_ir.relu, bb_ir.maxpool,
            bb_ir.layer1, bb_ir.layer2, bb_ir.layer3, bb_ir.layer4,
            bb_ir.avgpool,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim * 2, num_classes),
        )

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        f_rgb = torch.flatten(self.encoder_rgb(rgb), 1)
        f_ir = torch.flatten(self.encoder_ir(ir), 1)
        fused = torch.cat([f_rgb, f_ir], dim=1)
        return self.classifier(fused)


class DE_ResNet(nn.Module):
    """Double-Encoder ResNet (DE-ResNet).
    Fuses intermediate feature maps across RGB and IR streams via lateral residual summation at each stage.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.rgb_net = resnet18(weights=weights)
        self.ir_net = resnet18(weights=weights)
        feat_dim = 512

        self.rgb_net.fc = nn.Identity()
        self.ir_net.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        x_rgb = self.rgb_net.maxpool(self.rgb_net.relu(self.rgb_net.bn1(self.rgb_net.conv1(rgb))))
        x_ir = self.ir_net.maxpool(self.ir_net.relu(self.ir_net.bn1(self.ir_net.conv1(ir))))

        x_rgb = self.rgb_net.layer1(x_rgb)
        x_ir = self.ir_net.layer1(x_ir) + x_rgb

        x_rgb = self.rgb_net.layer2(x_rgb)
        x_ir = self.ir_net.layer2(x_ir) + x_rgb

        x_rgb = self.rgb_net.layer3(x_rgb)
        x_ir = self.ir_net.layer3(x_ir) + x_rgb

        x_rgb = self.rgb_net.layer4(x_rgb)
        x_ir = self.ir_net.layer4(x_ir) + x_rgb

        out = torch.flatten(self.ir_net.avgpool(x_ir), 1)
        return self.classifier(out)


class CrossModalAttentionBlock(nn.Module):
    """Bidirectional Spatial-Channel Cross-Modal Attention & Squeeze-and-Excitation Gating Block.
    
    1. Spatial Cross-Attention:
       Thermal IR hotspot maps guide spatial focus on RGB morphological features (callus, skin integrity, ulcers),
       while RGB anatomical boundaries suppress background thermal noise.
    2. Adaptive Channel Gate:
       Dynamically balances modality contributions per channel via a learnable gate g in [0, 1]^C.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid_c = max(8, channels // reduction)

        # Spatial cross-attention convs (avg + max pool -> 7x7 conv -> sigmoid)
        self.spatial_ir_to_rgb = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.spatial_rgb_to_ir = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        # Squeeze-and-Excitation Adaptive Modality Channel Gate
        self.channel_gate = nn.Sequential(
            nn.Linear(channels * 2, mid_c, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_c, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, f_rgb: torch.Tensor, f_ir: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Bidirectional Spatial Cross-Modulation
        ir_spatial_stats = torch.cat([torch.mean(f_ir, dim=1, keepdim=True), torch.max(f_ir, dim=1, keepdim=True)[0]], dim=1)
        rgb_spatial_stats = torch.cat([torch.mean(f_rgb, dim=1, keepdim=True), torch.max(f_rgb, dim=1, keepdim=True)[0]], dim=1)

        attn_ir_on_rgb = self.spatial_ir_to_rgb(ir_spatial_stats)
        attn_rgb_on_ir = self.spatial_rgb_to_ir(rgb_spatial_stats)

        rgb_refined = f_rgb * (1.0 + attn_ir_on_rgb)
        ir_refined = f_ir * (1.0 + attn_rgb_on_ir)

        # 2. Adaptive Channel Gating
        b, c, _, _ = f_rgb.shape
        gap_rgb = F.adaptive_avg_pool2d(rgb_refined, 1).view(b, c)
        gap_ir = F.adaptive_avg_pool2d(ir_refined, 1).view(b, c)

        gate = self.channel_gate(torch.cat([gap_rgb, gap_ir], dim=1)).view(b, c, 1, 1)
        f_fused = gate * rgb_refined + (1.0 - gate) * ir_refined

        return rgb_refined, ir_refined, f_fused


class CMA_ResNet(nn.Module):
    """NOVELTY ARCHITECTURE: Cross-Modal Attention & Gated Multimodal ResNet (CMA-ResNet).
    
    Key Innovations over LF-ResNet and DE-ResNet:
    1. Multi-stage Bidirectional Spatial-Channel Cross-Modal Attention (at Stages 2, 3, and 4).
    2. Learnable Squeeze-and-Excitation Modality Gating that prevents thermal noise from washing out
       visual ulcer/callus cues in 4-class diabetic risk staging (Healthy vs R0 vs R1 vs R2).
    3. Multi-Scale Feature Aggregation + Optional Angiosome Thermal Asymmetry (FiLM) conditioning.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 4,
        pretrained: bool = False,
        dropout: float = 0.3,
        angiosome_dim: int = 0,
    ):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.rgb_net = resnet18(weights=weights)
        self.ir_net = resnet18(weights=weights)

        self.cma2 = CrossModalAttentionBlock(channels=128)
        self.cma3 = CrossModalAttentionBlock(channels=256)
        self.cma4 = CrossModalAttentionBlock(channels=512)

        self.angiosome_dim = angiosome_dim
        if angiosome_dim > 0:
            self.film_generator = nn.Sequential(
                nn.Linear(angiosome_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 512 * 2),
            )

        # Combined representation: [f_rgb_pool (512) || f_ir_pool (512) || f_cma4_pool (512)] -> 1536
        self.classifier = nn.Sequential(
            nn.LayerNorm(512 * 3),
            nn.Dropout(p=dropout),
            nn.Linear(512 * 3, 256),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
        angiosome_feats: torch.Tensor = None,
    ) -> torch.Tensor:
        # Stem + Stage 1
        x_rgb = self.rgb_net.maxpool(self.rgb_net.relu(self.rgb_net.bn1(self.rgb_net.conv1(rgb))))
        x_ir = self.ir_net.maxpool(self.ir_net.relu(self.ir_net.bn1(self.ir_net.conv1(ir))))

        x_rgb = self.rgb_net.layer1(x_rgb)
        x_ir = self.ir_net.layer1(x_ir)

        # Stage 2 + Cross-Modal Attention
        x_rgb = self.rgb_net.layer2(x_rgb)
        x_ir = self.ir_net.layer2(x_ir)
        x_rgb, x_ir, _ = self.cma2(x_rgb, x_ir)

        # Stage 3 + Cross-Modal Attention
        x_rgb = self.rgb_net.layer3(x_rgb)
        x_ir = self.ir_net.layer3(x_ir)
        x_rgb, x_ir, _ = self.cma3(x_rgb, x_ir)

        # Stage 4 + Cross-Modal Attention
        x_rgb = self.rgb_net.layer4(x_rgb)
        x_ir = self.ir_net.layer4(x_ir)
        x_rgb, x_ir, f_fused4 = self.cma4(x_rgb, x_ir)

        p_rgb = torch.flatten(F.adaptive_avg_pool2d(x_rgb, 1), 1)
        p_ir = torch.flatten(F.adaptive_avg_pool2d(x_ir, 1), 1)
        p_fused = torch.flatten(F.adaptive_avg_pool2d(f_fused4, 1), 1)

        if self.angiosome_dim > 0 and angiosome_feats is not None:
            film_params = self.film_generator(angiosome_feats)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            p_fused = p_fused * (1.0 + torch.tanh(gamma)) + beta

        joint = torch.cat([p_rgb, p_ir, p_fused], dim=1)
        return self.classifier(joint)

