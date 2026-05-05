# ruff: noqa: E402, I001
"""Generate benchmark CSV and Markdown validation report."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from derivatives_engine.utils.validation import generate_validation_report


def main() -> None:
    results, csv_path, report_path = generate_validation_report(ROOT / "reports")
    passed = int(results["passed"].sum())
    total = len(results)
    print(f"Validation complete: {passed}/{total} checks passed.")
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")
    if passed != total:
        failed = results.loc[
            ~results["passed"], ["category", "benchmark", "value", "target"]
        ]
        print("Failed checks:")
        print(failed.to_string(index=False))


if __name__ == "__main__":
    main()
