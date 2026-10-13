"""PyTorch Dataset Implementations for Diabetic Foot Modalities
Supports RGB, IR, Shape, Multimodal (RGB+IR), and Tabular Feature inputs.
"""

from pathlib import Path
from typing import Optional, Literal
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.config import DATA_DIR, INPUT_IMAGE_SIZE
from src.data.preprocessing import preprocess_image_to_tensor
from src.data.segmentation import apply_mask_background_removal


class PlantarFootDataset(Dataset):
    """Multimodal dataset for Diabetic Foot classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: Path = DATA_DIR,
        modality: Literal["RGB", "IR", "SHAPE", "RGB-IR"] = "RGB",
        task: Literal["binary", "multiclass"] = "binary",
        features_df: Optional[pd.DataFrame] = None,
        apply_background_removal: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.modality = modality.upper()
        self.task = task
        self.features_df = features_df
        self.apply_background_removal = apply_background_removal

        # Precompute features array if available
        if self.features_df is not None:
            # Merge on patient_id and timepoint
            feature_cols = [c for c in self.features_df.columns if c not in ["patient_id", "label", "multiclass_label", "clinical_stage"]]
            self.feature_matrix = self.features_df[feature_cols].values.astype(np.float32)
        else:
            self.feature_matrix = None

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, rel_path: str) -> Image.Image:
        full_path = self.data_root / rel_path
        if not full_path.exists():
            # Try direct relative path
            full_path = Path(rel_path)
        img = Image.open(full_path)
        return img.convert("RGB")

    def _load_mask(self, rel_path: str) -> Image.Image:
        full_path = self.data_root / rel_path
        if not full_path.exists():
            full_path = Path(rel_path)
        return Image.open(full_path).convert("L")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        # Determine target label
        if self.task == "binary":
            label = int(row["label"])
        else:
            label = int(row["multiclass_label"])

        mask = None
        if "mask_path" in row and pd.notna(row["mask_path"]):
            mask = self._load_mask(row["mask_path"])

        # Load modalities
        if self.modality == "RGB":
            img = self._load_image(row["rgb_path"])
            if self.apply_background_removal and mask is not None:
                img = apply_mask_background_removal(img, np.array(mask))
            tensor = preprocess_image_to_tensor(img)
            sample = tensor

        elif self.modality == "IR":
            img = self._load_image(row["ir_path"])
            if self.apply_background_removal and mask is not None:
                img = apply_mask_background_removal(img, np.array(mask))
            tensor = preprocess_image_to_tensor(img)
            sample = tensor

        elif self.modality == "SHAPE":
            if mask is None:
                # Fallback: create mask from RGB
                rgb_img = self._load_image(row["rgb_path"])
                mask_np = np.array(rgb_img.convert("L"))
                mask = Image.fromarray(mask_np)
            tensor = preprocess_image_to_tensor(mask.convert("RGB"))
            sample = tensor

        elif self.modality in ["RGB-IR", "MULTIMODAL"]:
            rgb_img = self._load_image(row["rgb_path"])
            ir_img = self._load_image(row["ir_path"])
            if self.apply_background_removal and mask is not None:
                mask_np = np.array(mask)
                rgb_img = apply_mask_background_removal(rgb_img, mask_np)
                ir_img = apply_mask_background_removal(ir_img, mask_np)
            rgb_tensor = preprocess_image_to_tensor(rgb_img)
            ir_tensor = preprocess_image_to_tensor(ir_img)
            sample = (rgb_tensor, ir_tensor)
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

        output = {
            "sample": sample,
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": row["patient_id"],
            "index": idx,
        }

        if self.feature_matrix is not None:
            output["features"] = torch.tensor(self.feature_matrix[idx], dtype=torch.float32)

        return output
