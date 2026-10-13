"""Hybrid Late-Fusion Ensemble (CNN + Statistical Thermal Descriptors)
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Section 2.6.2 & Tables 7, 8
Combines deep visual representations with the 120 handcrafted angiosome features.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.config import RANDOM_SEED


class HybridLateFusionEnsemble:
    """Late-fusion ensemble combining CNN predictions with statistical features."""

    def __init__(
        self,
        strategy: str = "soft_voting",  # "soft_voting" or "stacking"
        weight_cnn: float = 0.5,
        random_state: int = RANDOM_SEED,
    ):
        self.strategy = strategy
        self.weight_cnn = weight_cnn
        self.random_state = random_state
        self.meta_classifier = None

    def fit(
        self,
        cnn_probs: np.ndarray,
        tabular_probs: np.ndarray,
        labels: np.ndarray,
    ):
        """Fit ensemble meta-learner if stacking strategy is chosen."""
        if self.strategy == "stacking":
            X_meta = np.hstack([cnn_probs, tabular_probs])
            self.meta_classifier = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight="balanced",
            )
            self.meta_classifier.fit(X_meta, labels)
        return self

    def predict_proba(
        self,
        cnn_probs: np.ndarray,
        tabular_probs: np.ndarray,
    ) -> np.ndarray:
        """Combine predictions."""
        if self.strategy == "stacking" and self.meta_classifier is not None:
            X_meta = np.hstack([cnn_probs, tabular_probs])
            return self.meta_classifier.predict_proba(X_meta)
        else:
            # Soft voting / weighted average
            w_cnn = self.weight_cnn
            w_tab = 1.0 - w_cnn
            fused_probs = w_cnn * cnn_probs + w_tab * tabular_probs
            # Renormalize to ensure valid simplex
            row_sums = fused_probs.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            return fused_probs / row_sums

    def predict(
        self,
        cnn_probs: np.ndarray,
        tabular_probs: np.ndarray,
    ) -> np.ndarray:
        """Predict class labels with highest probability."""
        probs = self.predict_proba(cnn_probs, tabular_probs)
        return np.argmax(probs, axis=1)
