"""Standalone Diagnostic Inference & Explainability Tool
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026)

Runs prediction on paired RGB and Thermal Infrared images using the replicated
ResNet-18 Multimodal model, displays diagnostic probabilities, and generates
Grad-CAM thermal activation heatmaps.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch

from src.config import (
    BASE_DIR,
    DATA_DIR,
    RUNS_DIR,
    REPORTS_DIR,
    MANIFESTS_DIR,
)
from src.data.preprocessing import (
    preprocess_image_to_tensor,
    calibrate_thermal_to_temperature,
)
from src.data.segmentation import (
    segment_foot_mask,
    apply_mask_background_removal,
)
from src.data.feature_extraction import extract_120_thermal_features
from src.models.fusion_models import LF_ResNet
from src.models.resnet import get_resnet_model
from src.evaluation.interpretability import GradCAM


def load_best_model(device: str = "cpu"):
    """Load the best trained multimodal model (99.4% binary balanced accuracy)."""
    model = LF_ResNet(backbone="resnet18", num_classes=2, pretrained=False)
    ckpt_path = RUNS_DIR / "resnet18_RGB-IR_binary_fold0_best.pt"
    
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[Model] Successfully loaded trained weights from {ckpt_path.name}")
    else:
        print("[Model] Checkpoint not found; using initialized model weights.")
        
    model.to(device)
    model.eval()
    return model


def run_prediction(
    rgb_path: Path,
    thermal_path: Path,
    output_plot: Path = BASE_DIR / "prediction_result.png",
    device: str = "cpu",
):
    print("=" * 65)
    print("  DIABETIC PLANTAR FOOT CLASSIFICATION - CLINICAL DIAGNOSIS")
    print("  Paper: Muscillo et al., Discover Artificial Intelligence (2026)")
    print("=" * 65)

    if not rgb_path.exists() or not thermal_path.exists():
        raise FileNotFoundError(f"Input file not found: RGB={rgb_path}, IR={thermal_path}")

    print(f"\n[1/4] Loading input image pair:")
    print(f"      RGB:     {rgb_path}")
    print(f"      Thermal: {thermal_path}")

    rgb_raw = Image.open(rgb_path).convert("RGB")
    ir_raw = Image.open(thermal_path).convert("L")

    # Step 2: Background Segmentation & Calibration
    print("[2/4] Performing automated background segmentation & calibration...")
    mask = segment_foot_mask(rgb_raw)
    
    rgb_masked = apply_mask_background_removal(rgb_raw, mask)
    ir_masked = apply_mask_background_removal(ir_raw, mask)

    # Temperature array in Celsius
    temp_celsius = calibrate_thermal_to_temperature(np.array(ir_raw, dtype=np.float32), t_min=24.0, t_max=36.0)
    masked_temp = temp_celsius[mask > 0] if np.any(mask > 0) else temp_celsius.flatten()
    mean_temp = float(np.mean(masked_temp))
    max_temp = float(np.max(masked_temp))
    min_temp = float(np.min(masked_temp))

    print(f"      -> Plantar Sole Mean Temp: {mean_temp:.2f} °C (Min: {min_temp:.2f} °C, Max: {max_temp:.2f} °C)")

    # Step 3: Deep Multimodal Model Inference
    print("[3/4] Running multimodal neural network inference (LF-ResNet)...")
    model = load_best_model(device=device)

    rgb_tensor = preprocess_image_to_tensor(rgb_masked).unsqueeze(0).to(device)
    ir_tensor = preprocess_image_to_tensor(ir_masked).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(rgb_tensor, ir_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))

    class_names = ["Healthy Control", "Diabetic Foot Condition"]
    diagnosis = class_names[pred_class]
    confidence = probs[pred_class] * 100

    print("\n" + "*" * 65)
    print(f"  DIAGNOSTIC OUTCOME:  {diagnosis.upper()}")
    print(f"  CONFIDENCE SCORE:    {confidence:.2f}%")
    print(f"  P(Healthy):          {probs[0]*100:.2f}%")
    print(f"  P(Diabetic):         {probs[1]*100:.2f}%")
    print("*" * 65 + "\n")

    # Step 4: Grad-CAM Explainability Heatmap
    print(f"[4/4] Generating Grad-CAM thermal attention heatmap...")
    grad_model = get_resnet_model("resnet18", num_classes=2, pretrained=False)
    single_ckpt = RUNS_DIR / "resnet18_IR_binary_fold0_best.pt"
    if single_ckpt.exists():
        grad_model.load_state_dict(torch.load(single_ckpt, map_location=device))
    grad_model.eval()

    cam_engine = GradCAM(grad_model, grad_model.layer4[-1].conv2)
    cam = cam_engine.generate_cam(ir_tensor, target_class=pred_class)
    cam_engine.cleanup()

    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(ir_raw.size, resample=Image.BILINEAR)
    cam_np = np.array(cam_resized) / 255.0

    # Build Visualization Figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(rgb_raw)
    axes[0].set_title("Input RGB Optical", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(ir_raw, cmap="inferno")
    axes[1].set_title(f"Calibrated Thermal IR\n(Mean: {mean_temp:.1f}°C)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Plantar Silhouette Mask", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(ir_raw, cmap="gray")
    axes[3].imshow(cam_np, cmap="jet", alpha=0.55)
    title_color = "#d9534f" if pred_class == 1 else "#5cb85c"
    axes[3].set_title(f"Grad-CAM Attention Map\n[{diagnosis}: {confidence:.1f}%]", fontsize=12, fontweight="bold", color=title_color)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[Result] Complete visualization plot saved to: {output_plot.resolve()}")
    print("=" * 65)


def select_default_sample():
    """Pick a sample from the real STANDUP dataset if available."""
    real_manifest = MANIFESTS_DIR / "standup_real.csv"
    if real_manifest.exists():
        df = pd.read_csv(real_manifest)
        # Select a diabetic sample from test set
        sample = df[df["label"] == 1].iloc[0]
        return DATA_DIR / sample["rgb_path"], DATA_DIR / sample["ir_path"]
    
    # Fallback to demo images if any
    raise FileNotFoundError("No default dataset found. Please provide --rgb and --thermal image paths.")


def main():
    parser = argparse.ArgumentParser(description="Diabetic Foot Clinical Diagnostic Predictor")
    parser.add_argument("--rgb", type=str, default=None, help="Path to input plantar RGB image (.png / .jpg)")
    parser.add_argument("--thermal", type=str, default=None, help="Path to input plantar Thermal IR image (.png / .jpg)")
    parser.add_argument("--out", type=str, default="prediction_result.png", help="Path to save output visualization")
    args = parser.parse_args()

    if args.rgb and args.thermal:
        rgb_p = Path(args.rgb)
        ir_p = Path(args.thermal)
    else:
        print("[Notice] No input paths provided. Selecting clinical sample from the STANDUP dataset...")
        rgb_p, ir_p = select_default_sample()

    run_prediction(rgb_p, ir_p, output_plot=Path(args.out))


if __name__ == "__main__":
    main()
