"""Configuration and Hyperparameters for Diabetic Plantar Foot Classification
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) 6:1415
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MANIFESTS_DIR = DATA_DIR / "manifests"
SPLITS_DIR = DATA_DIR / "splits"
RUNS_DIR = BASE_DIR / "runs"
REPORTS_DIR = BASE_DIR / "reports"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MANIFESTS_DIR, SPLITS_DIR, RUNS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Random Seed
RANDOM_SEED = 42

# Image Processing Specifications (Paper Section 2.3)
SQUARE_PADDED_SIZE = 640  # Resized square with black padding
INPUT_IMAGE_SIZE = (224, 224)  # Final CNN input resolution

# ImageNet standard normalization (Paper Section 2.3)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Plantar Angiosome anatomical regions (Paper Section 2.4)
ANGIOSOME_REGIONS = ["MPA", "LPA", "MCA", "LCA"]
TIME_POINTS = ["T0", "T10"]

# 15 Statistical Descriptors per Angiosome (Table 1)
FEATURE_METRICS = [
    "lr_mean",
    "std",
    "pooled_std",
    "p5",
    "p95",
    "skewness",
    "kurtosis",
    "idr",  # Inter-decile range (90th - 10th percentile)
    "z_score",
    "glcm_asm",
    "glcm_contrast",
    "glcm_correlation",
    "glcm_dissimilarity",
    "glcm_energy",
    "glcm_homogeneity",
]
TOTAL_STATISTICAL_FEATURES = len(ANGIOSOME_REGIONS) * len(FEATURE_METRICS) * len(TIME_POINTS) # 4 * 15 * 2 = 120

# Class Definitions (Paper Section 2.1 & 2.6)
BINARY_CLASSES = {0: "Healthy", 1: "Diabetic"}
MULTICLASS_LABELS = {
    0: "Healthy",
    1: "R0",  # Diabetic without neuropathy
    2: "R1",  # Neuropathic diabetic
    3: "R2",  # Neuroischemic diabetic
}

# Evaluation Protocol (Paper Section 2.6.3)
HELD_OUT_TEST_RATIO = 0.15
N_SPLITS_CV = 5
VALIDATION_RATIO_PER_FOLD = 0.15  # 70% train, 15% val, 15% test overall
