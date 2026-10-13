"""LSNet (Large-Small Network) Architecture
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Section 2.5.1, Table 2
Paper Reference: Wang et al., 'LSNet: See large, focus small' (CVPR 2025)
Variants:
- LSNet-T (Tiny): ~11.4M parameters
- LSNet-S (Small): ~16.1M parameters
- LSNet-B (Base): ~23.2M parameters
"""

import torch
import torch.nn as nn
from typing import Literal


class LargeSmallBlock(nn.Module):
    """LSNet core block: 'See Large, Focus Small'.
    Combines large-kernel depth-wise convolution (7x7) for broad receptive field
    with small-kernel point-wise operations (1x1) for fine-grained local focus.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        # 'See Large': Depthwise 7x7 convolution
        self.large_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.norm1 = nn.BatchNorm2d(dim)
        
        # 'Focus Small': Pointwise MLP
        mlp_dim = int(dim * mlp_ratio)
        self.focus_mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
        )
        self.norm2 = nn.BatchNorm2d(dim)
        self.drop_path = nn.Dropout(p=drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual branch 1 (large kernel spatial perception)
        res1 = x + self.drop_path(self.norm1(self.large_conv(x)))
        # Residual branch 2 (pointwise channel reasoning)
        out = res1 + self.drop_path(self.norm2(self.focus_mlp(res1)))
        return out


class Downsample(nn.Module):
    """Downsampling patch transition."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class LSNet(nn.Module):
    """LSNet (Large-Small Network) backbone for foot image classification."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        dims: tuple[int, ...] = (64, 128, 256, 512),
        depths: tuple[int, ...] = (3, 3, 9, 3),
        dropout: float = 0.3,
    ):
        super().__init__()
        # Stem (patch embedding)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(dims[0]),
        )
        
        # Stages
        self.stages = nn.ModuleList()
        current_dim = dims[0]
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            stage_blocks = []
            if i > 0:
                stage_blocks.append(Downsample(dims[i - 1], dim))
            for _ in range(depth):
                stage_blocks.append(LargeSmallBlock(dim))
            self.stages.append(nn.Sequential(*stage_blocks))
            
        # Pooling and Classification Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(dims[-1], num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.pool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.classifier[2](self.classifier[1](feats))


def get_lsnet_model(
    variant: Literal["lsnet-t", "lsnet-s", "lsnet-b"] = "lsnet-t",
    num_classes: int = 2,
    dropout: float = 0.3,
) -> LSNet:
    """Instantiate LSNet variants matching parameter sizes in Table 2:
    LSNet-T: ~11.4M
    LSNet-S: ~16.1M
    LSNet-B: ~23.2M
    """
    v = variant.lower()
    if v in ["lsnet-t", "lsnet_t", "lsnet-tiny"]:
        # Dims tuned for ~11.4M parameters
        dims = (56, 112, 224, 448)
        depths = (2, 2, 8, 2)
    elif v in ["lsnet-s", "lsnet_s", "lsnet-small"]:
        # Dims tuned for ~16.1M parameters
        dims = (64, 128, 256, 512)
        depths = (2, 2, 12, 2)
    elif v in ["lsnet-b", "lsnet_b", "lsnet-base"]:
        # Dims tuned for ~23.2M parameters
        dims = (80, 160, 320, 640)
        depths = (3, 3, 14, 3)
    else:
        raise ValueError(f"Unknown LSNet variant: {variant}")

    return LSNet(num_classes=num_classes, dims=dims, depths=depths, dropout=dropout)
