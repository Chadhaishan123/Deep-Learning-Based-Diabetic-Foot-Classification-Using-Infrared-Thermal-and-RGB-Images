"""Evaluation Metrics for Diabetic Plantar Foot Classification
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Section 2.6.3 (Equations 4, 5, 6, 7)
"""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    num_classes: int = 2,
) -> dict[str, float]:
    """Compute Balanced Accuracy (Eq 4), Precision (Eq 6), Recall (Eq 5), F1-Score (Eq 7), and AUC-ROC."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    if num_classes == 2:
        prec = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="binary", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    else:
        # Multi-class macro average (Formulae 4-7)
        prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    metrics = {
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
    }

    if y_prob is not None:
        try:
            if num_classes == 2:
                # Binary AUC
                prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                metrics["auc_roc"] = float(roc_auc_score(y_true, prob_pos))
            else:
                metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            metrics["auc_roc"] = 0.0

    return metrics


def aggregate_fold_metrics(metrics_list: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Compute mean ± standard deviation across cross-validation evaluation runs (as reported in Tables 3-8)."""
    keys = metrics_list[0].keys()
    aggregated = {}
    for k in keys:
        vals = [m[k] for m in metrics_list]
        aggregated[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    return aggregated
