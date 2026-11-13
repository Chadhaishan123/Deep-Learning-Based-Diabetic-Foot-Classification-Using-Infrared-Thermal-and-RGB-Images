"""Display All Replicated Paper Tables in the Terminal
Parses the replicated markdown reports and formats them into clean ASCII tables.
"""

import sys
import argparse
from pathlib import Path
from tabulate import tabulate

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"


def parse_markdown_table(file_path: Path):
    """Parse a GitHub-flavored Markdown table into (headers, rows)."""
    if not file_path.exists():
        return None, None

    lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines()]
    table_lines = [l for l in lines if l.startswith("|") and l.endswith("|")]

    if len(table_lines) < 3:
        return None, None

    headers = [col.strip() for col in table_lines[0].split("|")[1:-1]]
    rows = []
    for line in table_lines[2:]:
        sanitized_line = line.replace("±", "+/-").replace("**", "")
        cols = [col.strip() for col in sanitized_line.split("|")[1:-1]]
        if len(cols) == len(headers):
            rows.append(cols)

    return headers, rows


def print_table_section(title: str, file_name: str):
    file_path = REPORTS_DIR / file_name
    headers, rows = parse_markdown_table(file_path)

    print("\n" + "=" * 88)
    print(f"  {title.upper()}")
    print("=" * 88)

    if headers and rows:
        print(tabulate(rows, headers=headers, tablefmt="grid", showindex=False))
    else:
        print(f"Report file {file_name} not found or empty.")
    print("=" * 88 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Display Replicated Paper Tables in Terminal")
    parser.add_argument(
        "--table",
        type=str,
        default="all",
        choices=["3", "4", "5", "6", "hybrid", "all"],
        help="Specific table to display (3, 4, 5, 6, hybrid, all)",
    )
    args = parser.parse_args()

    tables = {
        "3": ("Table 3: Tabular ML Baselines (120 Features, RGB, IR)", "table3_results.md"),
        "4": ("Table 4: Deep Learning Models on Binary Task (Healthy vs Diabetic)", "table4_results.md"),
        "5": ("Table 5: Deep Learning Models on 4-Class Task (Healthy, R0, R1, R2)", "table5_results.md"),
        "6": ("Table 6: Shape-Based Morphological Classification", "table6_results.md"),
        "hybrid": ("Tables 7 & 8: Hybrid Late-Fusion (CNN + 120 Angiosome Features)", "hybrid_results.md"),
    }

    if args.table == "all":
        for key, (title, fname) in tables.items():
            print_table_section(title, fname)
    else:
        title, fname = tables[args.table]
        print_table_section(title, fname)


if __name__ == "__main__":
    main()
