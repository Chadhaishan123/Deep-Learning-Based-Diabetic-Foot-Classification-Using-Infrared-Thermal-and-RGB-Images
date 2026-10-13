"""Patient-Level Disjoint Dataset Partitioning
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Section 2.6.3
Ensures zero data leakage across training, validation, and held-out test sets.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from src.config import RANDOM_SEED, HELD_OUT_TEST_RATIO, N_SPLITS_CV, SPLITS_DIR


def create_patient_level_splits(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    label_col: str = "label",
    stratify_col: str = "clinical_stage",
    test_size: float = HELD_OUT_TEST_RATIO,
    n_splits_cv: int = N_SPLITS_CV,
    random_state: int = RANDOM_SEED,
    output_dir: Path = SPLITS_DIR,
) -> dict:
    """Perform strict patient-level partition:
    1. 15% fixed held-out test set based on unique patients, stratified by class.
    2. Remaining 85% partitioned into stratified 5-fold cross-validation (70% train / 15% val).
    3. Programmatically verifies 0 intersection between patient ID sets.
    """
    # Group by patient to get patient-level label for stratification
    strat_key = stratify_col if stratify_col in df.columns else label_col
    patient_df = (
        df.groupby(patient_col)
        .agg({strat_key: "first", label_col: "first"})
        .reset_index()
    )
    
    unique_patients = patient_df[patient_col].values
    patient_strat_labels = patient_df[strat_key].values

    # Stratified split for test set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    dev_idx, test_idx = next(sss.split(unique_patients, patient_strat_labels))
    
    test_patients = set(unique_patients[test_idx])
    dev_patients = unique_patients[dev_idx]
    dev_labels = patient_strat_labels[dev_idx]
    
    # 5-fold stratified cross validation on dev patients
    sgkf = StratifiedGroupKFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state)
    folds = []
    
    for fold_idx, (train_sub_idx, val_sub_idx) in enumerate(
        sgkf.split(dev_patients, dev_labels, groups=dev_patients)
    ):
        fold_train_patients = set(dev_patients[train_sub_idx])
        fold_val_patients = set(dev_patients[val_sub_idx])
        
        # Verify disjointness
        assert len(fold_train_patients.intersection(fold_val_patients)) == 0, "Train-val overlap detected!"
        assert len(fold_train_patients.intersection(test_patients)) == 0, "Train-test overlap detected!"
        assert len(fold_val_patients.intersection(test_patients)) == 0, "Val-test overlap detected!"
        
        folds.append({
            "fold": fold_idx,
            "train_patients": sorted(list(fold_train_patients)),
            "val_patients": sorted(list(fold_val_patients)),
        })
        
    split_info = {
        "random_seed": random_state,
        "total_patients": len(unique_patients),
        "test_patients": sorted(list(test_patients)),
        "dev_patients": sorted(list(dev_patients)),
        "folds": folds,
    }
    
    # Save splits to JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "patient_splits.json"
    with open(out_file, "w") as f:
        json.dump(split_info, f, indent=2)
        
    return split_info


def load_patient_splits(split_file: Path = SPLITS_DIR / "patient_splits.json") -> dict:
    """Load precomputed patient-level splits."""
    with open(split_file, "r") as f:
        return json.load(f)
