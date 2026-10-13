"""Deep Learning Modalities & Multimodal Fusion Experiments
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Tables 4 & 5, Figure 9
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from src.config import (
    DATA_DIR,
    MANIFESTS_DIR,
    SPLITS_DIR,
    REPORTS_DIR,
    RUNS_DIR,
    RANDOM_SEED,
)
from src.data.split import load_patient_splits
from src.train import train_single_fold, set_seed
from src.evaluation.metrics import aggregate_fold_metrics
from src.evaluation.reporter import build_results_table, save_markdown_report
from src.evaluation.interpretability import GradCAM
from src.data.preprocessing import preprocess_image_to_tensor
from src.data.dataset import PlantarFootDataset


def run_table4_and_table5(
    manifest_path: Path = MANIFESTS_DIR / "standup_real.csv",
    splits_path: Path = DATA_DIR / "splits_real" / "patient_splits.json",
    epochs: int = 2,
    batch_size: int = 32,
    device: str = "cpu",
):
    set_seed(RANDOM_SEED)
    df = pd.read_csv(manifest_path)
    splits = load_patient_splits(splits_path)

    test_patients = set(splits["test_patients"])
    test_df = df[df["patient_id"].isin(test_patients)].reset_index(drop=True)

    # Architectures to evaluate
    experiments = [
        # Binary (Table 4)
        ("resnet18", "RGB", "binary", 2),
        ("resnet18", "IR", "binary", 2),
        ("resnet18", "RGB-IR", "binary", 2),
        ("resnet50", "RGB", "binary", 2),
        ("resnet50", "IR", "binary", 2),
        ("lsnet-t", "RGB", "binary", 2),
        ("lsnet-t", "IR", "binary", 2),
        ("lf-resnet", "RGB-IR", "binary", 2),
        ("de-resnet", "RGB-IR", "binary", 2),

        # Multi-class (Table 5)
        ("resnet18", "RGB", "multiclass", 4),
        ("resnet18", "IR", "multiclass", 4),
        ("resnet18", "RGB-IR", "multiclass", 4),
        ("resnet50", "RGB", "multiclass", 4),
        ("resnet50", "IR", "multiclass", 4),
        ("lsnet-t", "RGB", "multiclass", 4),
        ("lsnet-t", "IR", "multiclass", 4),
        ("lf-resnet", "RGB-IR", "multiclass", 4),
        ("de-resnet", "RGB-IR", "multiclass", 4),
    ]

    t4_results = []
    t5_results = []

    print("\n=======================================================")
    print("Running Tables 4 & 5: Deep Learning & Multimodal Models")
    print("=======================================================")

    for model_name, modality, task, num_classes in experiments:
        print(f"\n--> Model: {model_name.upper()} | Modality: {modality} | Task: {task}")
        fold_metrics = []

        for fold in splits["folds"][:2]:  # 2 CV folds
            train_df = df[df["patient_id"].isin(set(fold["train_patients"]))].reset_index(drop=True)
            val_df = df[df["patient_id"].isin(set(fold["val_patients"]))].reset_index(drop=True)

            m, _ = train_single_fold(
                fold_idx=fold["fold"],
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                model_name=model_name,
                modality=modality,
                task=task,
                num_classes=num_classes,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                resume=True,
            )
            fold_metrics.append(m)

        agg = aggregate_fold_metrics(fold_metrics)
        agg["model"] = model_name.upper()
        agg["modality"] = modality
        agg["num_classes"] = num_classes

        print(f"[{model_name.upper()} | {modality} | {task}] Bal. Acc: {agg['balanced_accuracy']['mean']:.3f} ± {agg['balanced_accuracy']['std']:.3f} | F1: {agg['f1_score']['mean']:.3f}")

        if task == "binary":
            t4_results.append(agg)
        else:
            t5_results.append(agg)

    # Generate Markdown Tables
    t4_df = build_results_table(t4_results)
    save_markdown_report(t4_df, "Table 4 Replication: Deep Learning Models on Binary Task (Healthy vs Diabetic)", REPORTS_DIR / "table4_results.md")

    t5_df = build_results_table(t5_results)
    save_markdown_report(t5_df, "Table 5 Replication: Deep Learning Models on 4-Class Classification", REPORTS_DIR / "table5_results.md")

    print("\n[Done] Table 4 and Table 5 Markdown reports generated.")


def run_gradcam_visualization(
    manifest_path: Path = MANIFESTS_DIR / "standup_real.csv",
    checkpoint_path: Path = RUNS_DIR / "resnet18_IR_binary_fold0_best.pt",
    output_path: Path = REPORTS_DIR / "figure9_gradcam.png",
):
    """Generate Grad-CAM visualization matching Figure 9 in the paper."""
    print("\n=======================================================")
    print("Generating Figure 9: Grad-CAM Explainability Heatmaps")
    print("=======================================================")

    df = pd.read_csv(manifest_path)
    # Pick a diabetic sample and a healthy sample
    diabetic_sample = df[df["label"] == 1].iloc[0]
    healthy_sample = df[df["label"] == 0].iloc[0]

    from src.models.resnet import get_resnet_model
    model = get_resnet_model("resnet18", num_classes=2)
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # Target last conv layer in layer4
    target_layer = model.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for row_idx, (sample_row, title_prefix) in enumerate([(healthy_sample, "Healthy Control"), (diabetic_sample, "Diabetic Foot")]):
        ir_img = Image.open(DATA_DIR / sample_row["ir_path"]).convert("L")
        rgb_img = Image.open(DATA_DIR / sample_row["rgb_path"]).convert("RGB")
        mask_img = Image.open(DATA_DIR / sample_row["mask_path"]).convert("L")

        # Tensor
        tensor = preprocess_image_to_tensor(ir_img.convert("RGB")).unsqueeze(0)
        cam = gradcam.generate_cam(tensor, target_class=int(sample_row["label"]))

        # Resize CAM to image shape
        cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(ir_img.size, resample=Image.BILINEAR)
        cam_np = np.array(cam_resized) / 255.0

        # Subplot 1: RGB Image
        axes[row_idx, 0].imshow(rgb_img)
        axes[row_idx, 0].set_title(f"{title_prefix} - RGB")
        axes[row_idx, 0].axis("off")

        # Subplot 2: Thermal IR Image
        axes[row_idx, 1].imshow(ir_img, cmap="inferno")
        axes[row_idx, 1].set_title(f"{title_prefix} - Thermal IR")
        axes[row_idx, 1].axis("off")

        # Subplot 3: Grad-CAM Overlay
        axes[row_idx, 2].imshow(ir_img, cmap="gray")
        axes[row_idx, 2].imshow(cam_np, cmap="jet", alpha=0.5)
        axes[row_idx, 2].set_title(f"{title_prefix} - Grad-CAM Activation")
        axes[row_idx, 2].axis("off")

    gradcam.cleanup()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Done] Figure 9 saved to {output_path}")


def main():
    run_table4_and_table5(epochs=2)
    run_gradcam_visualization()


if __name__ == "__main__":
    main()
