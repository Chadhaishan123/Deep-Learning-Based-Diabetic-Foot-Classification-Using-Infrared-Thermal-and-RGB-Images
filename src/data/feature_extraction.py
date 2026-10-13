"""Angiosome Subdivision and Statistical Thermal Feature Extraction (120 Features)
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Section 2.4 & Table 1
"""

import numpy as np
from scipy import stats
from skimage.feature import graycomatrix, graycoprops
from src.config import ANGIOSOME_REGIONS, FEATURE_METRICS


def partition_foot_into_angiosomes(foot_mask: np.ndarray, is_left_foot: bool = True) -> dict[str, np.ndarray]:
    """Subdivide a single foot mask into 4 plantar angiosomes:
    - MPA: Medial Plantar Angiosome
    - LPA: Lateral Plantar Angiosome
    - MCA: Medial Calcaneal Angiosome
    - LCA: Lateral Calcaneal Angiosome
    
    Returns dictionary mapping region name to boolean mask.
    """
    binary = (foot_mask > 0).astype(bool)
    h, w = binary.shape
    
    # Find bounding box of the foot
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not np.any(rows) or not np.any(cols):
        # Empty mask fallback
        empty = np.zeros_like(binary)
        return {r: empty for r in ANGIOSOME_REGIONS}
        
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    foot_height = ymax - ymin
    
    # Anatomical split:
    # Calcaneal region (heel) occupies roughly the lower 35% of the foot (from ymax upwards)
    calcaneal_boundary_y = int(ymax - 0.35 * foot_height)
    
    # Lateral vs Medial split:
    # If left foot: medial (inner) is to the right side of the foot, lateral (outer) is to the left side
    # If right foot: medial (inner) is to the left side, lateral is to the right side
    mid_x = (xmin + xmax) // 2
    
    y_coords, x_coords = np.ogrid[:h, :w]
    
    is_calcaneal = (y_coords >= calcaneal_boundary_y) & binary
    is_plantar = (y_coords < calcaneal_boundary_y) & binary
    
    is_right_side = x_coords >= mid_x
    is_left_side = x_coords < mid_x
    
    if is_left_foot:
        # Left foot: inner (medial) is right side
        medial_mask = is_right_side
        lateral_mask = is_left_side
    else:
        # Right foot: inner (medial) is left side
        medial_mask = is_left_side
        lateral_mask = is_right_side
        
    regions = {
        "MPA": is_plantar & medial_mask,
        "LPA": is_plantar & lateral_mask,
        "MCA": is_calcaneal & medial_mask,
        "LCA": is_calcaneal & lateral_mask,
    }
    return regions


def compute_glcm_features(diff_patch: np.ndarray) -> dict[str, float]:
    """Compute 6 GLCM texture features: ASM, Contrast, Correlation, Dissimilarity, Energy, Homogeneity."""
    if diff_patch.size == 0 or np.all(diff_patch == 0):
        return {
            "glcm_asm": 0.0,
            "glcm_contrast": 0.0,
            "glcm_correlation": 0.0,
            "glcm_dissimilarity": 0.0,
            "glcm_energy": 0.0,
            "glcm_homogeneity": 0.0,
        }
        
    # Quantize difference patch to 16 gray levels for fast and stable GLCM calculation
    patch_min, patch_max = np.min(diff_patch), np.max(diff_patch)
    if patch_max > patch_min:
        quantized = np.clip(
            ((diff_patch - patch_min) / (patch_max - patch_min) * 15).astype(np.uint8),
            0,
            15,
        )
    else:
        quantized = np.zeros_like(diff_patch, dtype=np.uint8)
        
    # GLCM with distance=1, angle=0
    glcm = graycomatrix(quantized, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
    
    asm = float(graycoprops(glcm, "ASM")[0, 0])
    contrast = float(graycoprops(glcm, "contrast")[0, 0])
    correlation = float(graycoprops(glcm, "correlation")[0, 0])
    dissimilarity = float(graycoprops(glcm, "dissimilarity")[0, 0])
    energy = float(graycoprops(glcm, "energy")[0, 0])
    homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
    
    # Handle NaN in correlation if uniform
    if np.isnan(correlation):
        correlation = 0.0
        
    return {
        "glcm_asm": asm,
        "glcm_contrast": contrast,
        "glcm_correlation": correlation,
        "glcm_dissimilarity": dissimilarity,
        "glcm_energy": energy,
        "glcm_homogeneity": homogeneity,
    }


def compute_angiosome_features(
    left_temp_vals: np.ndarray,
    right_temp_vals: np.ndarray,
    left_patch: np.ndarray,
    right_patch: np.ndarray,
) -> dict[str, float]:
    """Compute the 15 statistical features for a single angiosome region.
    
    Thermal difference |left - right| metrics:
    - Mean (lr_mean)
    - Standard Deviation
    - Pooled Std
    - 5th Percentile
    - 95th Percentile
    - Skewness
    - Kurtosis
    - Inter-decile range (p90 - p10)
    - Z-Score
    - 6 GLCM texture metrics
    """
    if len(left_temp_vals) == 0 or len(right_temp_vals) == 0:
        return {metric: 0.0 for metric in FEATURE_METRICS}
        
    # Sample or align length for pairwise absolute difference
    n = min(len(left_temp_vals), len(right_temp_vals))
    l_sorted = np.sort(left_temp_vals)[:n]
    r_sorted = np.sort(right_temp_vals)[:n]
    abs_diff = np.abs(l_sorted - r_sorted)
    
    lr_mean = float(np.mean(abs_diff))
    std = float(np.std(abs_diff))
    
    # Pooled standard deviation between left and right foot distributions
    s_l = np.var(left_temp_vals, ddof=1) if len(left_temp_vals) > 1 else 0.0
    s_r = np.var(right_temp_vals, ddof=1) if len(right_temp_vals) > 1 else 0.0
    pooled_std = float(np.sqrt(max((s_l + s_r) / 2.0, 0.0)))
    
    p5 = float(np.percentile(abs_diff, 5))
    p95 = float(np.percentile(abs_diff, 95))
    p10 = float(np.percentile(abs_diff, 10))
    p90 = float(np.percentile(abs_diff, 90))
    idr = float(p90 - p10)
    
    # Distribution shape
    skewness = float(stats.skew(abs_diff)) if std > 1e-6 else 0.0
    kurt = float(stats.kurtosis(abs_diff)) if std > 1e-6 else 0.0
    
    # Z-Score
    z_score = float(lr_mean / (std + 1e-6))
    
    # GLCM difference map on 2D patch
    min_h = min(left_patch.shape[0], right_patch.shape[0])
    min_w = min(left_patch.shape[1], right_patch.shape[1])
    diff_patch_2d = np.abs(
        left_patch[:min_h, :min_w].astype(np.float32) - right_patch[:min_h, :min_w].astype(np.float32)
    )
    glcm_dict = compute_glcm_features(diff_patch_2d)
    
    metrics = {
        "lr_mean": lr_mean,
        "std": std,
        "pooled_std": pooled_std,
        "p5": p5,
        "p95": p95,
        "skewness": skewness,
        "kurtosis": kurt,
        "idr": idr,
        "z_score": z_score,
        **glcm_dict,
    }
    return metrics


def extract_120_thermal_features(
    temp_map_t0: np.ndarray,
    mask_t0: np.ndarray,
    temp_map_t10: np.ndarray,
    mask_t10: np.ndarray,
) -> dict[str, float]:
    """Extract complete 120-feature statistical vector across 4 angiosomes and 2 time points (T0, T10).
    
    Features are named: `{time}_{metric}_{region}` or `{time}_{region}_{metric}`.
    For instance: `T0_lr_mean_MPA`, `T0_pooled_std_MPA`, `T10_pooled_std_MCA`, etc.
    """
    feature_dict = {}
    
    for time_point, (temp_map, mask) in [("T0", (temp_map_t0, mask_t0)), ("T10", (temp_map_t10, mask_t10))]:
        # Split image into left foot and right foot halves
        w = temp_map.shape[1]
        mid_x = w // 2
        
        l_temp = temp_map[:, :mid_x]
        l_mask = mask[:, :mid_x]
        
        r_temp = temp_map[:, mid_x:]
        r_mask = mask[:, mid_x:]
        
        l_angiosomes = partition_foot_into_angiosomes(l_mask, is_left_foot=True)
        r_angiosomes = partition_foot_into_angiosomes(r_mask, is_left_foot=False)
        
        for region in ANGIOSOME_REGIONS:
            l_reg_mask = l_angiosomes[region]
            r_reg_mask = r_angiosomes[region]
            
            l_vals = l_temp[l_reg_mask]
            r_vals = r_temp[r_reg_mask]
            
            # Extract 2D patches
            l_patch = np.where(l_reg_mask, l_temp, 0.0)
            r_patch = np.where(r_reg_mask, r_temp, 0.0)
            
            reg_features = compute_angiosome_features(l_vals, r_vals, l_patch, r_patch)
            for metric_name, val in reg_features.items():
                col_name = f"{time_point}_{metric_name}_{region}"
                feature_dict[col_name] = val
                
    return feature_dict
