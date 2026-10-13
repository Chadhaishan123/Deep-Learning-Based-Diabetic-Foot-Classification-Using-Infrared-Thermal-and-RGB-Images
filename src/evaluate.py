"""Evaluation and Diagnostic Inference Script
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Section 2.6.3 & Section 3
"""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from src.config import MANIFESTS_DIR, SPLITS_DIR, RUNS_DIR, REPORTS_DIR
from src.data.dataset import PlantarFootDataset
from src.data.split import load_patient_splits
from src.train import instantiate_model, evaluate_model
from src.evaluation.interpretability import GradCAM


def run_evaluation(
    manifest_path: Path = MANIFESTS_DIR / "dataset.csv",
    checkpoint_path: Path = None,
    model_name: str = "resnet18",
    modality: str = "IR",
    task: str = "binary",
    device: str = "cpu",
    generate_gradcam: bool = True,
):
    df = pd.read_csv(manifest_path)
    splits = load_patient_splits(SPLITS_DIR / "patient_splits.json")
    test_patients = set(splits["test_patients"])
    test_df = df[df["patient_id"].isin(test_patients)].reset_index(drop=True)

    num_classes = 2 if task == "binary" else 4
    is_multimodal = modality.upper() in ["RGB-IR", "MULTIMODAL"]

    test_ds = PlantarFootDataset(test_df, modality=modality, task=task)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    model = instantiate_model(model_name, modality, num_classes=num_classes).to(device)
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from: {checkpoint_path}")

    metrics, test_probs = evaluate_model(model, test_loader, device, num_classes, is_multimodal)
    print("\nHeld-out Test Set Evaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Generate sample Grad-CAM if requested and model has residual layers
    if generate_gradcam and hasattr(model, "layer4") and not is_multimodal:
        try:
            sample_item = test_ds[0]
            inp = sample_item["sample"].unsqueeze(0).to(device)
            cam_engine = GradCAM(model, model.layer4)
            cam = cam_engine.generate_cam(inp, target_class=sample_item["label"].item())
            cam_engine.cleanup()
            gradcam_out = REPORTS_DIR / f"gradcam_{model_name}_{modality}_{task}.npy"
            np.save(gradcam_out, cam)
            print(f"Saved Grad-CAM attention map to: {gradcam_out}")
        except Exception as e:
            print(f"Grad-CAM generation notice: {e}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate foot classification model")
    parser.add_argument("--manifest", type=str, default=str(MANIFESTS_DIR / "dataset.csv"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--modality", type=str, default="IR", choices=["RGB", "IR", "SHAPE", "RGB-IR"])
    parser.add_argument("--task", type=str, default="binary", choices=["binary", "multiclass"])
    args = parser.parse_args()

    run_evaluation(
        manifest_path=Path(args.manifest),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        model_name=args.model,
        modality=args.modality,
        task=args.task,
    )
