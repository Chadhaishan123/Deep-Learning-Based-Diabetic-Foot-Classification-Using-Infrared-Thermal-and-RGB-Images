"""Clinically-Aware Ordinal & Focal Loss Functions (NOVELTY MODULE)
Addresses the 4-class diabetic risk staging progression:
    Healthy (0) < R0: Uncomplicated (1) < R1: Neuropathy/Ischemia (2) < R2: Ulceration Risk (3)

Standard Cross-Entropy treats misclassifying R2 as Healthy (distance 3) identically to
misclassifying R2 as R1 (distance 1). OrdinalFocalLoss combines:
  1. Focal Modulation (1 - p_t)^gamma to focus learning on hard intermediate stages (R0, R1, R2).
  2. Ordinal Clinical Distance Penalty W(y, y_hat) = 1 + alpha * E[|y - y_hat| / (K - 1)]
  3. Inverse-frequency class balancing weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalFocalLoss(nn.Module):
    """Clinically-Aware Ordinal Distance + Focal Loss for Multi-Stage Risk Stratification."""

    def __init__(
        self,
        num_classes: int = 4,
        gamma: float = 2.0,
        ordinal_alpha: float = 1.5,
        class_weights: torch.Tensor = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.ordinal_alpha = ordinal_alpha
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        # Precompute normalized distance matrix D[i, j] = |i - j| / (K - 1)
        classes = torch.arange(num_classes, dtype=torch.float32)
        dist_matrix = torch.abs(classes.unsqueeze(1) - classes.unsqueeze(0)) / max(1, num_classes - 1)
        self.register_buffer("dist_matrix", dist_matrix)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # True class probability p_t
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt).pow(self.gamma)

        # Standard cross-entropy with optional label smoothing and class weights
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing if self.num_classes > 2 else 0.0,
        )

        # Expected ordinal distance penalty: E_{y_hat ~ probs}[|y - y_hat| / (K - 1)]
        if self.num_classes > 2 and self.ordinal_alpha > 0:
            target_dists = self.dist_matrix[targets]  # (B, K)
            expected_ordinal_dist = torch.sum(probs * target_dists, dim=-1)  # (B,)
            ordinal_multiplier = 1.0 + self.ordinal_alpha * expected_ordinal_dist
        else:
            ordinal_multiplier = 1.0

        loss = ordinal_multiplier * focal_weight * ce_loss
        return loss.mean()

