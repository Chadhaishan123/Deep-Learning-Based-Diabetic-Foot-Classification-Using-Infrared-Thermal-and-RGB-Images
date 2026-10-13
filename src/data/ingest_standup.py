"""Ingestion and Mask Preprocessing for the Real STANDUP Database
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) 6:1415
"""

import os
import sys
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import (
    DATA_DIR,
    RAW_DATA_DIR,
    MANIFESTS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)
from src.data.segmentation import segment_foot_mask
from src.data.feature_extraction import extract_120_thermal_features
from src.data.split import create_patient_level_splits


DEFAULT_STANDUP_SRC = Path(r"C:\Users\ishan\Downloads\STANDUP_Database\STANDUP_Database")


def ingest_real_standup(
    source_dir: Path = DEFAULT_STANDUP_SRC,
    output_raw_dir: Path = RAW_DATA_DIR / "standup",
    manifest_path: Path = MANIFESTS_DIR / "standup_real.csv",
) -> pd.DataFrame:
    """Ingest the 415 pairs (830 images) from the official STANDUP dataset:
    - 125 Healthy pairs (RGB and thermal)
    - 138 Diabetic R0 pairs (RGB and thermal)
    - 68 Diabetic R1 pairs (RGB and thermal)
    - 84 Diabetic R2 pairs (RGB and thermal)
    Generates segmentation foot masks and constructs unified manifest.
    """
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"STANDUP source directory not found: {source_dir}")

    output_raw_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_raw_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    records = []
    
    categories = [
        ("healthy", "Healthy", 0, 0, source_dir / "healthy"),
        ("diabetic/R0", "R0", 1, 1, source_dir / "diabetic" / "R0"),
        ("diabetic/R1", "R1", 1, 2, source_dir / "diabetic" / "R1"),
        ("diabetic/R2", "R2", 1, 3, source_dir / "diabetic" / "R2"),
    ]

    print(f"Scanning and ingesting STANDUP database from {source_dir}...")

    for cat_rel, stage_name, bin_label, multi_label, cat_path in categories:
        rgb_folder = cat_path / "RGB"
        th_folder = cat_path / "thermal"
        
        target_cat_dir = output_raw_dir / cat_rel
        (target_cat_dir / "RGB").mkdir(parents=True, exist_ok=True)
        (target_cat_dir / "thermal").mkdir(parents=True, exist_ok=True)

        rgb_files = sorted(list(rgb_folder.glob("*.png")))
        print(f"Found {len(rgb_files)} pairs for category {stage_name}...")

        for rgb_file in tqdm(rgb_files, desc=f"Processing {stage_name}"):
            th_file = th_folder / rgb_file.name
            if not th_file.exists():
                print(f"Warning: Missing thermal match for {rgb_file.name}")
                continue

            stem = rgb_file.stem
            patient_id = f"{stage_name}_{stem}"

            # Copy to workspace raw directory
            dest_rgb = target_cat_dir / "RGB" / rgb_file.name
            dest_th = target_cat_dir / "thermal" / th_file.name
            
            if not dest_rgb.exists():
                shutil.copy2(rgb_file, dest_rgb)
            if not dest_th.exists():
                shutil.copy2(th_file, dest_th)

            # Generate segmentation mask
            dest_mask = mask_dir / f"{patient_id}_mask.png"
            if not dest_mask.exists():
                # Load RGB and thermal to segment
                th_img = Image.open(dest_th).convert("L")
                mask_np = segment_foot_mask(th_img)
                # If mask is too small, try segmenting RGB
                if mask_np.sum() < (mask_np.size * 0.02 * 255):
                    rgb_img = Image.open(dest_rgb).convert("RGB")
                    mask_np = segment_foot_mask(rgb_img)
                mask_img = Image.fromarray(mask_np)
                mask_img.save(dest_mask)

            # Relative paths from data/
            rel_rgb = dest_rgb.relative_to(DATA_DIR).as_posix()
            rel_th = dest_th.relative_to(DATA_DIR).as_posix()
            rel_mask = dest_mask.relative_to(DATA_DIR).as_posix()

            records.append({
                "patient_id": patient_id,
                "label": bin_label,
                "clinical_stage": stage_name,
                "multiclass_label": multi_label,
                "timepoint": "T0",
                "rgb_path": rel_rgb,
                "ir_path": rel_th,
                "thermal_path": rel_th,
                "mask_path": rel_mask,
            })

    df = pd.DataFrame(records)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    print(f"Ingested {len(df)} patient records saved to {manifest_path}!")
    return df


def extract_real_standup_features(
    df: pd.DataFrame,
    output_features_csv: Path = PROCESSED_DATA_DIR / "standup_real_features_120.csv",
) -> pd.DataFrame:
    """Extract 120 statistical features across the 4 angiosomes on the real STANDUP dataset."""
    if output_features_csv.exists():
        print(f"Loading existing features from {output_features_csv}...")
        return pd.read_csv(output_features_csv)

    print("Extracting 120 angiosome features on real STANDUP dataset...")
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        th_path = DATA_DIR / row["ir_path"]
        mask_path = DATA_DIR / row["mask_path"]

        th_img = Image.open(th_path).convert("L")
        mask_img = Image.open(mask_path).convert("L")

        temp_np = np.array(th_img, dtype=np.float32)
        mask_np = np.array(mask_img, dtype=np.uint8)

        # For single timepoint acquisitions, compute T0 and pseudo-recovery descriptors
        # or evaluate T0 thermal distributions
        feat_dict = extract_120_thermal_features(temp_np, mask_np, temp_np, mask_np)
        feat_dict["patient_id"] = row["patient_id"]
        feat_dict["label"] = int(row["label"])
        feat_dict["multiclass_label"] = int(row["multiclass_label"])
        feat_dict["clinical_stage"] = row["clinical_stage"]
        records.append(feat_dict)

    feat_df = pd.DataFrame(records)
    output_features_csv.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(output_features_csv, index=False)
    print(f"Saved real 120-feature dataset to {output_features_csv}!")
    return feat_df


if __name__ == "__main__":
    df = ingest_real_standup()
    feat_df = extract_real_standup_features(df)
    splits = create_patient_level_splits(df, output_dir=DATA_DIR / "splits_real")
    print("Ingestion, segmentation, feature extraction, and patient-level splitting complete!")
