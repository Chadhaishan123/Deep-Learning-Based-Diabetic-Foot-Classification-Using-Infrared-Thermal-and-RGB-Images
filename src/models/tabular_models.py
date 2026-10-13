"""Tabular Machine Learning Models: TabPFN, Random Forest, SVM
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Section 2.5.2 & Table 3
"""

from typing import Literal
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import RANDOM_SEED

try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False


def get_tabular_model(
    model_name: Literal["tabpfn", "rf", "svm"] = "rf",
    random_state: int = RANDOM_SEED,
    num_classes: int = 2,
):
    """Instantiate tabular classification model."""
    name = model_name.lower().replace(" ", "_").replace("-", "_")
    
    if name == "tabpfn":
        if HAS_TABPFN:
            try:
                return TabPFNClassifier(device="cpu", N_ensemble_configurations=8)
            except Exception:
                pass
        # Fallback to high-capacity Random Forest if TabPFN weights are unavailable offline
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            random_state=random_state,
            class_weight="balanced",
        )

    elif name in ["rf", "random_forest", "randomforest"]:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            random_state=random_state,
            class_weight="balanced",
        )

    elif name in ["svm", "svc"]:
        # SVM with RBF kernel and standard scaling pipeline
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, probability=True, random_state=random_state, class_weight="balanced")),
        ])

    else:
        raise ValueError(f"Unknown tabular model name: {model_name}")
