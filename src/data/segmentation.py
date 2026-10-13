"""Foot Segmentation and Background Removal Pipeline
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Section 2.3 & 3.1
"""

import numpy as np
from PIL import Image
import cv2


def compute_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) between predicted and ground truth masks (Formula 2)."""
    pred = (mask_pred > 0).astype(bool)
    gt = (mask_gt > 0).astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_dice(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """Compute DICE score / F1 overlap between predicted and ground truth masks (Formula 3)."""
    pred = (mask_pred > 0).astype(bool)
    gt = (mask_gt > 0).astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / denom)


def segment_foot_mask(image: Image.Image) -> np.ndarray:
    """Segment plantar foot region from background.
    
    Produces a binary mask (255 for foot, 0 for background) matching the STANDUP
    foot outline specification. Supports RGB or grayscale IR inputs.
    """
    img_np = np.array(image)
    if img_np.ndim == 3:
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()

    # Mild gaussian blur to suppress high-frequency noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu thresholding as robust initialization
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological closing to fill small internal holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Keep only the two largest connected components (left and right feet)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    
    if num_labels > 1:
        # Sort components by area excluding background (label 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        sorted_indices = np.argsort(areas)[::-1] + 1
        
        # Keep top 1 or 2 components (feet)
        mask = np.zeros_like(gray, dtype=np.uint8)
        top_k = min(2, len(sorted_indices))
        for k in range(top_k):
            # Only include if component area is non-trivial (>1% of image)
            if stats[sorted_indices[k], cv2.CC_STAT_AREA] > (gray.size * 0.01):
                mask[labels == sorted_indices[k]] = 255
    else:
        mask = closed

    return mask


def apply_mask_background_removal(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Apply binary mask to keep only foot pixels against completely black background."""
    img_np = np.array(image)
    binary_mask = (mask > 0).astype(np.uint8)
    
    if img_np.ndim == 3:
        masked_np = img_np * binary_mask[:, :, None]
    else:
        masked_np = img_np * binary_mask
        
    return Image.fromarray(masked_np)
