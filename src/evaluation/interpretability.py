"""Interpretability and Explainability: Feature Importance Analysis and Grad-CAM
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Section 3.4 & Figures 8, 9
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
import torch
import torch.nn.functional as F


def analyze_feature_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    top_k: int = 15,
    output_path: Path = None,
) -> dict:
    """Perform both Univariate ANOVA F-scoring and Permutation Feature Importance (Figure 8).
    
    Returns ranked feature dictionaries for both methods.
    """
    # 1. Univariate ANOVA F-score
    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(X, y)
    f_scores = selector.scores_
    # Replace NaN with 0
    f_scores = np.nan_to_num(f_scores, nan=0.0)

    # 2. Permutation Importance (mean decrease in balanced accuracy)
    perm_result = permutation_importance(
        model,
        X,
        y,
        scoring="balanced_accuracy",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    perm_scores = perm_result.importances_mean

    # Sort
    f_ranked_idx = np.argsort(f_scores)[::-1][:top_k]
    p_ranked_idx = np.argsort(perm_scores)[::-1][:top_k]

    f_top = [(feature_names[i], float(f_scores[i])) for i in f_ranked_idx]
    p_top = [(feature_names[i], float(perm_scores[i])) for i in p_ranked_idx]

    # Plot if output_path is provided
    if output_path is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: ANOVA F-test
        f_feats, f_vals = zip(*reversed(f_top))
        axes[0].barh(f_feats, f_vals, color="#3470a3")
        axes[0].set_title("Univariate ANOVA F-test Score")
        axes[0].set_xlabel("ANOVA F-score")

        # Right: Permutation Importance
        p_feats, p_vals = zip(*reversed(p_top))
        axes[1].barh(p_feats, p_vals, color="#e07148")
        axes[1].set_title("Permutation Importance")
        axes[1].set_xlabel("Mean decrease in balanced accuracy")

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
        plt.close()

    return {"anova_f_top": f_top, "permutation_top": p_top}


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN visualization (Figure 9)."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Compute 2D Grad-CAM heatmap."""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize heatmap to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def cleanup(self):
        for h in self._hook_handles:
            h.remove()
