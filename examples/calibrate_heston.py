# ruff: noqa: E402, I001
"""Example: calibrate Heston parameters to a synthetic option chain."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from derivatives_engine.calibration.heston_calibration import (
    calibrate_heston,
    generate_synthetic_heston_chain,
    parameter_comparison_table,
)
from derivatives_engine.models.heston import HestonParams


def main() -> None:
    true_params = HestonParams(v0=0.04, kappa=1.4, theta=0.04, sigma_v=0.35, rho=-0.55)
    chain = generate_synthetic_heston_chain(
        100.0,
        0.03,
        0.0,
        true_params,
        strikes=np.array([85.0, 100.0, 115.0]),
        maturities=np.array([0.5, 1.0, 1.5]),
    )
    result = calibrate_heston(
        chain,
        initial_params=HestonParams(
            v0=0.045, kappa=1.2, theta=0.045, sigma_v=0.40, rho=-0.45
        ),
        max_nfev=50,
    )
    print("Synthetic Heston calibration example")
    print(f"Success: {result.success}")
    print(f"RMSE: {result.rmse:.8f}")
    print(f"MAE:  {result.mae:.8f}")
    print(parameter_comparison_table(true_params, result.params).to_string(index=False))
    print("Note: Heston calibration can be non-unique even when pricing fit is strong.")


if __name__ == "__main__":
    main()
