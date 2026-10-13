"""Preprocessing Pipeline for RGB and Infrared Thermal Foot Images
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Section 2.3
"""

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from src.config import SQUARE_PADDED_SIZE, INPUT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def calibrate_thermal_to_temperature(
    intensity_map: np.ndarray,
    t_min: float,
    t_max: float,
    i_min: float = 0.0,
    i_max: float = 255.0,
) -> np.ndarray:
    """Calibrate grayscale pixel intensity to absolute temperature in degrees Celsius.
    
    Formula (1):
        T(x, y) = T_min + (I(x, y) - I_min) / (I_max - I_min) * (T_max - T_min)
    """
    intensity = np.asarray(intensity_map, dtype=np.float32)
    denom = max(i_max - i_min, 1e-6)
    normalized_intensity = np.clip((intensity - i_min) / denom, 0.0, 1.0)
    temperature_map = t_min + normalized_intensity * (t_max - t_min)
    return temperature_map


def pad_to_square(image: Image.Image, target_size: int = SQUARE_PADDED_SIZE, bg_color=(0, 0, 0)) -> Image.Image:
    """Aspect-ratio preserving resizing with black padding to square (Figure 2 in paper).
    
    1. Scales the larger dimension to `target_size` preserving aspect ratio.
    2. Centers the resized image onto a `target_size x target_size` black canvas.
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    
    resample_filter = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    resized_img = image.resize((new_w, new_h), resample=resample_filter)
    
    # Determine mode and background color
    mode = image.mode
    if mode == "L":
        canvas_bg = 0 if isinstance(bg_color, tuple) else bg_color
    else:
        canvas_bg = bg_color
        if mode != "RGB":
            resized_img = resized_img.convert("RGB")
            mode = "RGB"

    padded_img = Image.new(mode, (target_size, target_size), canvas_bg)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    padded_img.paste(resized_img, (paste_x, paste_y))
    return padded_img


def preprocess_image_to_tensor(
    image: Image.Image,
    target_size: tuple = INPUT_IMAGE_SIZE,
    apply_imagenet_norm: bool = True,
) -> torch.Tensor:
    """Full preprocessing:
    1. Square padding (640x640)
    2. Resizing to 224x224
    3. Normalization with ImageNet mean and std
    """
    # Pad to square
    squared = pad_to_square(image, target_size=SQUARE_PADDED_SIZE)
    # Resize to 224x224
    resample_filter = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    final_img = squared.resize(target_size, resample=resample_filter)
    
    if final_img.mode != "RGB":
        final_img = final_img.convert("RGB")

    tensor = T.functional.to_tensor(final_img)  # [0, 1] range, (3, H, W)
    if apply_imagenet_norm:
        tensor = T.functional.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor


def separate_left_right_feet(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Separate dual-foot plantar image into left and right foot components."""
    w, h = image.size
    mid_x = w // 2
    left_box = (0, 0, mid_x, h)
    right_box = (mid_x, 0, w, h)
    left_foot = image.crop(left_box)
    right_foot = image.crop(right_box)
    return left_foot, right_foot
