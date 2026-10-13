"""Training Engine for Deep Learning, Fusion, and Tabular Foot Classification Models
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Section 2.5.1 & 2.6.3
"""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DATA_DIR,
    MANIFESTS_DIR,
    SPLITS_DIR,
    RUNS_DIR,
    RANDOM_SEED,
)
from src.data.dataset import PlantarFootDataset
from src.data.split import create_patient_level_splits, load_patient_splits
from src.models.resnet import get_resnet_model
from src.models.lsnet import get_lsnet_model
from src.models.fusion_models import DE_ResNet, LF_ResNet
from src.models.tabular_models import get_tabular_model
from src.evaluation.metrics import compute_classification_metrics, aggregate_fold_metrics


def set_seed(seed: int = RANDOM_SEED):
    """Seed Python, NumPy, and PyTorch random number generators for strict reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_inverse_freq_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class balancing weights (Section 2.5.1)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = len(labels)
    weights = total / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def instantiate_model(
    model_name: str,
    modality: str,
    num_classes: int = 2,
    dropout: float = 0.3,
) -> nn.Module:
    """Instantiate appropriate neural network based on model and modality."""
    m_name = model_name.lower()
    mod = modality.upper()

    if mod in ["RGB-IR", "MULTIMODAL"]:
        if "de" in m_name or "double" in m_name:
            return DE_ResNet(backbone="resnet18", num_classes=num_classes, dropout=dropout)
        else:
            return LF_ResNet(backbone="resnet18", num_classes=num_classes, dropout=dropout)
            
    if "resnet18" in m_name:
        return get_resnet_model("resnet18", num_classes=num_classes, dropout=dropout)
    elif "resnet50" in m_name:
        return get_resnet_model("resnet50", num_classes=num_classes, dropout=dropout)
    elif "lsnet-t" in m_name or "lsnet_t" in m_name:
        return get_lsnet_model("lsnet-t", num_classes=num_classes, dropout=dropout)
    elif "lsnet-s" in m_name or "lsnet_s" in m_name:
        return get_lsnet_model("lsnet-s", num_classes=num_classes, dropout=dropout)
    elif "lsnet-b" in m_name or "lsnet_b" in m_name:
        return get_lsnet_model("lsnet-b", num_classes=num_classes, dropout=dropout)
    else:
        # Default fallback
        return get_resnet_model("resnet18", num_classes=num_classes, dropout=dropout)


def train_epoch(model, loader, criterion, optimizer, device, is_multimodal=False):
    model.train()
    running_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        if is_multimodal:
            rgb, ir = batch["sample"]
            rgb, ir = rgb.to(device), ir.to(device)
            labels = batch["label"].to(device)
            outputs = model(rgb, ir)
        else:
            inputs = batch["sample"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(labels)
    return running_loss / len(loader.dataset)


def evaluate_model(model, loader, device, num_classes=2, is_multimodal=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            if is_multimodal:
                rgb, ir = batch["sample"]
                rgb, ir = rgb.to(device), ir.to(device)
                outputs = model(rgb, ir)
            else:
                inputs = batch["sample"].to(device)
                outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = compute_classification_metrics(all_labels, all_preds, all_probs, num_classes=num_classes)
    return metrics, all_probs


def train_single_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    modality: str,
    task: str = "binary",
    num_classes: int = 2,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "cpu",
    save_dir: Path = RUNS_DIR,
    resume: bool = True,
):
    is_multimodal = modality.upper() in ["RGB-IR", "MULTIMODAL"]

    train_ds = PlantarFootDataset(train_df, modality=modality, task=task)
    val_ds = PlantarFootDataset(val_df, modality=modality, task=task)
    test_ds = PlantarFootDataset(test_df, modality=modality, task=task)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = instantiate_model(model_name, modality, num_classes=num_classes).to(device)

    best_weights_path = save_dir / f"{model_name}_{modality}_{task}_fold{fold_idx}_best.pt"
    if resume and best_weights_path.exists():
        print(f"Loading existing checkpoint: {best_weights_path.name}")
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        test_metrics, test_probs = evaluate_model(model, test_loader, device, num_classes, is_multimodal)
        return test_metrics, test_probs

    # Inverse-frequency class weights
    target_col = "label" if task == "binary" else "multiclass_label"
    train_labels = train_df[target_col].values.astype(int) if target_col in train_df.columns else np.array([train_ds[i]["label"].item() for i in range(len(train_ds))])
    class_weights = compute_inverse_freq_weights(np.array(train_labels), num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_bal_acc = -1.0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, is_multimodal)
        val_metrics, _ = evaluate_model(model, val_loader, device, num_classes, is_multimodal)
        scheduler.step()

        if val_metrics["balanced_accuracy"] > best_val_bal_acc:
            best_val_bal_acc = val_metrics["balanced_accuracy"]
            torch.save(model.state_dict(), best_weights_path)

    # Evaluate best model on the fixed patient-level held-out test set
    if best_weights_path.exists():
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
    test_metrics, test_probs = evaluate_model(model, test_loader, device, num_classes, is_multimodal)

    return test_metrics, test_probs


def train_cross_validation(
    manifest_path: Path = MANIFESTS_DIR / "dataset.csv",
    model_name: str = "resnet18",
    modality: str = "IR",
    task: str = "binary",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: Path = RUNS_DIR,
):
    set_seed(RANDOM_SEED)
    df = pd.read_csv(manifest_path)
    num_classes = 2 if task == "binary" else 4

    splits_file = SPLITS_DIR / "patient_splits.json"
    if not splits_file.exists():
        splits = create_patient_level_splits(df)
    else:
        splits = load_patient_splits(splits_file)

    test_patients = set(splits["test_patients"])
    test_df = df[df["patient_id"].isin(test_patients)].reset_index(drop=True)

    fold_test_metrics = []
    print(f"\n=======================================================")
    print(f"Training: {model_name} | Modality: {modality} | Task: {task} ({num_classes} classes)")
    print(f"Device: {device} | Folds: {len(splits['folds'])}")
    print(f"=======================================================")

    for fold_info in splits["folds"]:
        fold_idx = fold_info["fold"]
        train_p = set(fold_info["train_patients"])
        val_p = set(fold_info["val_patients"])

        train_df = df[df["patient_id"].isin(train_p)].reset_index(drop=True)
        val_df = df[df["patient_id"].isin(val_p)].reset_index(drop=True)

        metrics, _ = train_single_fold(
            fold_idx=fold_idx,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            model_name=model_name,
            modality=modality,
            task=task,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            save_dir=output_dir,
        )
        fold_test_metrics.append(metrics)
        print(f"Fold {fold_idx} Test Balanced Acc: {metrics['balanced_accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

    aggregated = aggregate_fold_metrics(fold_test_metrics)
    print("\nFinal 5-Fold Test Evaluation (Mean ± Std):")
    print(f"Balanced Accuracy: {aggregated['balanced_accuracy']['mean']:.4f} ± {aggregated['balanced_accuracy']['std']:.4f}")
    print(f"Precision:         {aggregated['precision']['mean']:.4f} ± {aggregated['precision']['std']:.4f}")
    print(f"Recall:            {aggregated['recall']['mean']:.4f} ± {aggregated['recall']['std']:.4f}")
    print(f"F1-Score:          {aggregated['f1_score']['mean']:.4f} ± {aggregated['f1_score']['std']:.4f}")

    # Save summary
    summary_path = output_dir / f"summary_{model_name}_{modality}_{task}.json"
    with open(summary_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    return aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train foot classification model")
    parser.add_argument("--manifest", type=str, default=str(MANIFESTS_DIR / "dataset.csv"))
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--modality", type=str, default="IR", choices=["RGB", "IR", "SHAPE", "RGB-IR"])
    parser.add_argument("--task", type=str, default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_cross_validation(
        manifest_path=Path(args.manifest),
        model_name=args.model,
        modality=args.modality,
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
