"""Training Log Viewer CLI
Usage:
    python show_logs.py                   # Shows list of available training logs and latest summary
    python show_logs.py --model resnet18_RGB-IR # Shows full training log for champion multimodal model
    python show_logs.py --model ir        # Shows training log for Thermal IR model
    python show_logs.py --model rgb       # Shows training log for RGB model
    python show_logs.py --model multiclass# Shows training log for 4-class model
    python show_logs.py --model shape     # Shows training log for Shape model
    python show_logs.py --model tabular   # Shows training log for Tabular models (TabPFN/RF/SVM)
    python show_logs.py --summary         # Shows benchmark summary across all models
    python show_logs.py --tail 15         # Shows last 15 lines of the active log
"""

import sys
import argparse
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent / "logs"


def list_logs():
    print("\n" + "="*70)
    print("  AVAILABLE MODEL TRAINING LOGS (logs/)")
    print("="*70)
    if not LOGS_DIR.exists():
        print("  [Error] logs/ directory does not exist.")
        return []

    files = sorted(list(LOGS_DIR.glob("*.log")) + sorted(list(LOGS_DIR.glob("*.csv"))))
    for idx, f in enumerate(files, 1):
        size_kb = f.stat().st_size / 1024
        print(f"  [{idx}] {f.name:<38} ({size_kb:.1f} KB)")
    print("="*70 + "\n")
    return files


def view_log(filename: str, tail: int = None):
    target = LOGS_DIR / filename
    if not target.exists():
        # Match case-insensitively or partial match
        matches = [f for f in LOGS_DIR.glob("*.log") if filename.lower() in f.name.lower()]
        if matches:
            target = matches[0]
        else:
            print(f"[Error] Log file matching '{filename}' not found in {LOGS_DIR}.")
            return

    with open(target, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if tail and tail > 0 and len(lines) > tail:
        print(f"--- Showing last {tail} lines of {target.name} ---")
        lines = lines[-tail:]
    else:
        print(f"--- Full log: {target.name} ---")

    print("".join(lines))


def main():
    parser = argparse.ArgumentParser(description="View model training logs in terminal")
    parser.add_argument("--list", action="store_true", help="List all available log files")
    parser.add_argument("--summary", action="store_true", help="Display master benchmark summary log")
    parser.add_argument("--model", type=str, default=None, help="Name or keyword of model log to view (e.g. resnet18_RGB-IR, ir, rgb, shape, tabular)")
    parser.add_argument("--tail", type=int, default=None, help="Display only the last N lines of the log")
    args = parser.parse_args()

    if args.summary:
        view_log("training_summary.log", args.tail)
        return

    if args.model:
        # Resolve common shortcuts
        m = args.model.lower()
        if "cma" in m or "novel" in m:
            target = "train_cma-resnet_RGB-IR_multiclass.log"
        elif "rgb-ir" in m or "multi" in m:
            if "multiclass" in m or "4" in m:
                target = "train_resnet18_RGB-IR_multiclass.log"
            else:
                target = "train_resnet18_RGB-IR_binary.log"
        elif "ir" in m or "thermal" in m:
            target = "train_resnet18_IR_binary.log"
        elif "rgb" in m:
            target = "train_resnet18_RGB_binary.log"
        elif "shape" in m:
            target = "train_resnet18_SHAPE_binary.log"
        elif "tab" in m or "rf" in m or "svm" in m:
            target = "train_tabular_models.log"
        else:
            target = args.model

        view_log(target, args.tail)
        return

    # Default action: list logs and print summary
    list_logs()
    print("Tip: Run 'python show_logs.py --summary' or 'python show_logs.py --model resnet18_RGB-IR' to inspect details.\n")
    view_log("training_summary.log", tail=args.tail or 35)


if __name__ == "__main__":
    main()

