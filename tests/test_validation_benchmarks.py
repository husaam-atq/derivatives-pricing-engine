"""Tests for benchmark report generation."""

from __future__ import annotations

from derivatives_engine.utils.validation import generate_validation_report


def test_validation_report_generation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    results, csv_path, report_path = generate_validation_report(tmp_path)
    assert csv_path.exists()
    assert report_path.exists()
    assert not results.empty
    assert int(results["passed"].sum()) == len(results)
