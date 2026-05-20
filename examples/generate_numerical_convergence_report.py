# ruff: noqa: E402, I001
"""Generate numerical convergence CSV and Markdown report."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from derivatives_engine.utils.convergence import generate_numerical_convergence_report


def main() -> None:
    results, csv_path, report_path = generate_numerical_convergence_report(
        ROOT / "reports"
    )
    print(f"Numerical convergence report generated with {len(results)} rows.")
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
