"""Results Reporter and Benchmark Tables Formatter
Paper: Muscillo et al. (Discover Artificial Intelligence, 2026) Tables 3, 4, 5, 6, 7, 8
"""

import pandas as pd
from pathlib import Path
from src.config import REPORTS_DIR

# Reference metrics published in Muscillo et al. (2026)
PAPER_REFERENCE_METRICS = {
    # Table 3: Tabular ML Models
    ("TabPFN", "Features", 2): {"balanced_accuracy": "0.984 ± 0.051"},
    ("Random Forest", "Features", 2): {"balanced_accuracy": "0.919 ± 0.025"},
    ("SVM", "Features", 2): {"balanced_accuracy": "0.920 ± 0.066"},
    ("Random Forest", "RGB", 2): {"balanced_accuracy": "0.823 ± 0.064"},
    ("SVM", "RGB", 2): {"balanced_accuracy": "0.822 ± 0.053"},
    ("Random Forest", "IR", 2): {"balanced_accuracy": "0.877 ± 0.047"},
    ("SVM", "IR", 2): {"balanced_accuracy": "0.842 ± 0.041"},
    ("TabPFN", "Features", 4): {"balanced_accuracy": "0.536 ± 0.050"},
    ("Random Forest", "Features", 4): {"balanced_accuracy": "0.442 ± 0.034"},
    ("SVM", "Features", 4): {"balanced_accuracy": "0.465 ± 0.067"},
    ("Random Forest", "RGB", 4): {"balanced_accuracy": "0.409 ± 0.031"},
    ("SVM", "RGB", 4): {"balanced_accuracy": "0.446 ± 0.042"},
    ("Random Forest", "IR", 4): {"balanced_accuracy": "0.446 ± 0.056"},
    ("SVM", "IR", 4): {"balanced_accuracy": "0.467 ± 0.018"},

    # Table 4: Deep learning binary
    ("ResNet18", "RGB", 2): {"balanced_accuracy": "0.970 ± 0.025", "precision": "0.982 ± 0.014", "recall": "0.981 ± 0.015", "f1_score": "0.981 ± 0.016"},
    ("ResNet50", "RGB", 2): {"balanced_accuracy": "0.955 ± 0.010", "precision": "0.973 ± 0.006", "recall": "0.972 ± 0.006", "f1_score": "0.972 ± 0.006"},
    ("ResNet18", "IR", 2): {"balanced_accuracy": "0.990 ± 0.012", "precision": "0.994 ± 0.008", "recall": "0.994 ± 0.008", "f1_score": "0.994 ± 0.008"},
    ("ResNet50", "IR", 2): {"balanced_accuracy": "0.995 ± 0.010", "precision": "0.997 ± 0.006", "recall": "0.997 ± 0.006", "f1_score": "0.997 ± 0.006"},
    ("LF-ResNet", "RGB-IR", 2): {"balanced_accuracy": "0.934 ± 0.015", "precision": "0.941 ± 0.008", "recall": "0.935 ± 0.013", "f1_score": "0.935 ± 0.012"},
    ("DE-ResNet", "RGB-IR", 2): {"balanced_accuracy": "0.954 ± 0.028", "precision": "0.955 ± 0.025", "recall": "0.952 ± 0.027", "f1_score": "0.953 ± 0.027"},

    # Table 7: Hybrid binary
    ("ResNet18", "RGB + Features", 2): {"balanced_accuracy": "1.000 ± 0.000", "precision": "1.000 ± 0.000", "recall": "1.000 ± 0.000", "f1_score": "1.000 ± 0.000"},
    ("ResNet50", "RGB + Features", 2): {"balanced_accuracy": "1.000 ± 0.000", "precision": "1.000 ± 0.000", "recall": "1.000 ± 0.000", "f1_score": "1.000 ± 0.000"},
    ("ResNet18", "IR + Features", 2): {"balanced_accuracy": "1.000 ± 0.000", "precision": "1.000 ± 0.000", "recall": "1.000 ± 0.000", "f1_score": "1.000 ± 0.000"},
    ("ResNet50", "IR + Features", 2): {"balanced_accuracy": "0.993 ± 0.006", "precision": "0.980 ± 0.016", "recall": "0.993 ± 0.006", "f1_score": "0.986 ± 0.011"},
}


def format_metric(mean: float, std: float) -> str:
    """Format floating point values as mean ± std."""
    return f"{mean:.3f} ± {std:.3f}"


def build_results_table(results_list: list[dict]) -> pd.DataFrame:
    """Convert raw experiment dictionaries to presentation DataFrame."""
    formatted_rows = []
    for r in results_list:
        row = {
            "Model": r.get("model", ""),
            "Data / Image": r.get("modality", ""),
            "Classes": r.get("num_classes", 2),
            "Bal. Accuracy": format_metric(r["balanced_accuracy"]["mean"], r["balanced_accuracy"]["std"]),
        }
        if "precision" in r:
            row["Precision"] = format_metric(r["precision"]["mean"], r["precision"]["std"])
            row["Recall"] = format_metric(r["recall"]["mean"], r["recall"]["std"])
            row["F1-Score"] = format_metric(r["f1_score"]["mean"], r["f1_score"]["std"])
        formatted_rows.append(row)
    return pd.DataFrame(formatted_rows)


def save_markdown_report(df: pd.DataFrame, title: str, output_path: Path):
    """Save results as formatted GitHub markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        try:
            f.write(df.to_markdown(index=False))
        except Exception:
            cols = list(df.columns)
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
            for _, row in df.iterrows():
                f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")
        f.write("\n\n")
