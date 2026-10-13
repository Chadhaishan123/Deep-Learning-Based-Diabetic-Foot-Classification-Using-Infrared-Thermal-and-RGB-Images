"""STANDUP Dataset Generator & Ingestion Utility
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Section 2.1
Replicates the cohort: 415 paired RGB and IR images across 125 healthy and 290 type II diabetic subjects (R0, R1, R2), with T0 and T10 acquisitions.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from src.config import (
    RAW_DATA_DIR,
    MANIFESTS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)
from src.data.preprocessing import calibrate_thermal_to_temperature, pad_to_square
from src.data.feature_extraction import extract_120_thermal_features


def draw_foot_contour(draw: ImageDraw.ImageDraw, offset_x: int, offset_y: int, scale: float = 1.0, is_left: bool = True):
    """Draw a realistic anatomical plantar foot outline (heel, arch, metatarsal, toes)."""
    # Key anatomical points normalized
    points = [
        (40, 20),   # big toe
        (60, 25),   # second toe
        (75, 35),   # third toe
        (85, 50),   # fourth toe
        (90, 70),   # pinky toe
        (85, 120),  # lateral forefoot
        (80, 180),  # lateral midfoot
        (75, 230),  # lateral heel
        (55, 260),  # posterior heel
        (35, 250),  # medial heel
        (25, 200),  # medial arch indentation
        (20, 140),  # medial ball of foot
        (25, 60),   # medial toe base
    ]
    if not is_left:
        # Mirror along x
        points = [(100 - x, y) for (x, y) in points]
        
    scaled_points = [
        (int(offset_x + x * scale), int(offset_y + y * scale))
        for (x, y) in points
    ]
    draw.polygon(scaled_points, fill=255)


def generate_mock_standup_dataset(
    num_healthy: int = 125,
    num_diabetic: int = 290,
    output_raw_dir: Path = RAW_DATA_DIR,
    manifest_path: Path = MANIFESTS_DIR / "dataset.csv",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate the full synthetic STANDUP dataset conforming to the cohort described in the paper:
    - 125 Healthy controls
    - 290 Type II diabetic patients: R0 (no neuropathy), R1 (neuropathic), R2 (neuroischemic)
    - Acquisitions at baseline (T0) and post-cold stress (T10)
    - Paired RGB and IR thermal images with ground-truth temperature bounds
    """
    rng = np.random.default_rng(seed)
    output_raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    rgb_dir = output_raw_dir / "rgb"
    ir_dir = output_raw_dir / "ir"
    mask_dir = output_raw_dir / "masks"
    for d in [rgb_dir, ir_dir, mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    records = []
    
    # Class breakdown
    # Diabetics are split into R0, R1, R2
    r0_count = int(round(num_diabetic * 0.40))
    r1_count = int(round(num_diabetic * 0.35))
    r2_count = num_diabetic - r0_count - r1_count

    patient_specs = []
    for i in range(num_healthy):
        patient_specs.append((f"H_{i:03d}", 0, "Healthy"))
    for i in range(r0_count):
        patient_specs.append((f"D_R0_{i:03d}", 1, "R0"))
    for i in range(r1_count):
        patient_specs.append((f"D_R1_{i:03d}", 1, "R1"))
    for i in range(r2_count):
        patient_specs.append((f"D_R2_{i:03d}", 1, "R2"))

    stage_to_multiclass_idx = {"Healthy": 0, "R0": 1, "R1": 2, "R2": 3}

    img_w, img_h = 320, 240
    
    print(f"Generating STANDUP cohort: {len(patient_specs)} patients...")

    for pat_id, binary_label, clinical_stage in patient_specs:
        multiclass_label = stage_to_multiclass_idx[clinical_stage]
        
        # Base temperatures
        # Healthy feet: ~28-31°C, symmetric between feet
        # Diabetic feet: temperature elevation/asymmetry or peripheral cool spots, impaired response at T10
        for timepoint in ["T0", "T10"]:
            # Create foot mask
            mask_img = Image.new("L", (img_w, img_h), 0)
            draw_mask = ImageDraw.Draw(mask_img)
            draw_foot_contour(draw_mask, offset_x=40, offset_y=10, scale=0.8, is_left=True)
            draw_foot_contour(draw_mask, offset_x=180, offset_y=10, scale=0.8, is_left=False)
            mask_np = np.array(mask_img)

            # Thermal base values
            if binary_label == 0:
                # Healthy
                base_temp = 29.5 + rng.normal(0, 0.5)
                asymmetry = rng.normal(0, 0.2)
                t_min = 24.0
                t_max = 34.0
                # After cold stress (T10), healthy feet recover or drop uniformly
                cold_drop = 2.0 if timepoint == "T10" else 0.0
            else:
                # Diabetic: Higher mean temperature due to inflammation, or ischemia, higher asymmetry
                asym_mult = 1.0 if clinical_stage == "R0" else (1.8 if clinical_stage == "R1" else 2.5)
                base_temp = 31.0 + rng.normal(0, 1.0)
                asymmetry = rng.normal(1.2 * asym_mult, 0.5)
                t_min = 23.0
                t_max = 36.0
                cold_drop = 3.5 if timepoint == "T10" else 0.0

            # Generate synthetic calibrated temperatures
            left_t = base_temp - cold_drop + rng.normal(0, 0.4, size=(img_h, img_w // 2))
            right_t = base_temp - cold_drop + asymmetry + rng.normal(0, 0.4, size=(img_h, img_w - img_w // 2))
            temp_map = np.hstack([left_t, right_t])
            # Outside foot is ambient temp (~22°C)
            temp_map = np.where(mask_np > 0, temp_map, 22.0)
            
            # Convert temperature to IR grayscale intensity (0-255)
            ir_gray = np.clip((temp_map - t_min) / (t_max - t_min) * 255.0, 0, 255).astype(np.uint8)
            # Create FLIR Ironbow / colormap-like RGB from IR
            ir_rgb_np = np.stack([ir_gray, (ir_gray * 0.8).astype(np.uint8), (255 - ir_gray)], axis=-1)
            ir_img = Image.fromarray(ir_rgb_np)

            # Generate synthetic RGB visible image (skin tone for foot, dark background)
            rgb_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            # Skin tone base
            skin_base = np.array([210, 160, 130], dtype=np.float32)
            noise = rng.normal(0, 10, size=(img_h, img_w, 3))
            skin_pixels = np.clip(skin_base + noise, 0, 255).astype(np.uint8)
            rgb_np[mask_np > 0] = skin_pixels[mask_np > 0]
            rgb_img = Image.fromarray(rgb_np)

            # Filenames
            filename_stem = f"{pat_id}_{timepoint}"
            rgb_filename = f"{filename_stem}_rgb.png"
            ir_filename = f"{filename_stem}_ir.png"
            mask_filename = f"{filename_stem}_mask.png"

            rgb_img.save(rgb_dir / rgb_filename)
            ir_img.save(ir_dir / ir_filename)
            mask_img.save(mask_dir / mask_filename)

            records.append({
                "patient_id": pat_id,
                "timepoint": timepoint,
                "label": binary_label,
                "clinical_stage": clinical_stage,
                "multiclass_label": multiclass_label,
                "rgb_path": f"raw/rgb/{rgb_filename}",
                "ir_path": f"raw/ir/{ir_filename}",
                "mask_path": f"raw/masks/{mask_filename}",
                "t_min": t_min,
                "t_max": t_max,
            })

    df = pd.DataFrame(records)
    df.to_csv(manifest_path, index=False)
    print(f"Generated {len(df)} image records saved to manifest: {manifest_path}")
    return df
