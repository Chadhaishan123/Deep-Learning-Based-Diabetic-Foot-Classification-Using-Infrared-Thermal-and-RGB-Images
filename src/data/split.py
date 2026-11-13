"""Patient-Level Disjoint Data Splitting Utility
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Section 2.6.2
Ensures strict patient-level separation across held-out test set (15%) and 5-fold CV.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import SPLITS_DIR, RANDOM_SEED


def create_patient_level_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: float = 0.15,
    seed: int = RANDOM_SEED,
    output_path: Path = SPLITS_DIR / "patient_splits.json",
) -> dict:
    """Create stratified patient-level held-out test split and k-fold cross-validation splits."""
    rng = np.random.default_rng(seed)
    patient_df = df.groupby("patient_id").first().reset_index()

    strat_col = "multiclass_label" if "multiclass_label" in patient_df.columns else "label"
    test_patients = []
    train_val_patients = []

    for cls_val, group in patient_df.groupby(strat_col):
        pids = group["patient_id"].values.copy()
        rng.shuffle(pids)
        n_test = max(1, int(round(len(pids) * test_size)))
        test_patients.extend(pids[:n_test].tolist())
        train_val_patients.extend(pids[n_test:].tolist())

    tv_df = patient_df[patient_df["patient_id"].isin(train_val_patients)].reset_index(drop=True)
    fold_buckets = [[] for _ in range(n_splits)]
    for cls_val, group in tv_df.groupby(strat_col):
        pids = group["patient_id"].values.copy()
        rng.shuffle(pids)
        for idx, pid in enumerate(pids):
            fold_buckets[idx % n_splits].append(pid)

    folds = []
    for f_idx in range(n_splits):
        val_p = fold_buckets[f_idx]
        train_p = [p for i, b in enumerate(fold_buckets) if i != f_idx for p in b]
        folds.append({
            "fold": f_idx,
            "train_patients": train_p,
            "val_patients": val_p,
        })

    splits_data = {
        "seed": seed,
        "test_patients": test_patients,
        "folds": folds,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)
    return splits_data


def load_patient_splits(splits_path: Path = SPLITS_DIR / "patient_splits.json") -> dict:
    """Load saved patient-level splits JSON."""
    with open(splits_path, "r", encoding="utf-8") as f:
        return json.load(f)

