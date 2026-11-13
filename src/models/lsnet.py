"""Large-Small Network (LSNet) Implementation
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Section 2.5.1
Reference: Wang et al., 'See Large, Focus Small: LSNet' (CVPR 2025)
Supports LSNet-T (Tiny), LSNet-S (Small), and LSNet-B (Base).
"""

import torch
import torch.nn as nn


class LSBlock(nn.Module):
    """Large-Small Convolution-MLP Block:
    7x7 depthwise conv ('See Large') + 1x1 pointwise MLP ('Focus Small') with residual connection.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pwconv2(x)
        return shortcut + x


class LSNet(nn.Module):
    """Hierarchical 4-stage Large-Small Network."""

    def __init__(
        self,
        depths: list[int] = [2, 2, 6, 2],
        dims: list[int] = [64, 128, 256, 512],
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(dims[0]),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample = nn.Sequential(
                nn.BatchNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[LSBlock(dim=dims[i], drop=dropout * 0.2) for _ in range(depths[i])])
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(dims[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])  # Global Average Pooling
        x = self.norm(x)
        return self.head(x)


def get_lsnet_model(
    variant: str = "lsnet-t",
    num_classes: int = 2,
    pretrained: bool = False,
    dropout: float = 0.3,
) -> nn.Module:
    v = variant.lower()
    if "s" in v and "lsnet-t" not in v and "lsnet_t" not in v:
        return LSNet(depths=[3, 3, 9, 3], dims=[64, 128, 320, 512], num_classes=num_classes, dropout=dropout)
    elif "b" in v:
        return LSNet(depths=[3, 3, 18, 3], dims=[96, 192, 384, 768], num_classes=num_classes, dropout=dropout)
    else:
        return LSNet(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], num_classes=num_classes, dropout=dropout)

