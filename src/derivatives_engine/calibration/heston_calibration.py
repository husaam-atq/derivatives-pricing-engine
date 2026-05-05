"""Synthetic Heston calibration workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from derivatives_engine.models.heston import HestonParams, heston_price_cf
from derivatives_engine.risk.implied_volatility import implied_volatility


@dataclass(frozen=True)
class HestonCalibrationResult:
    """Heston calibration output."""

    params: HestonParams
    rmse: float
    mae: float
    success: bool
    message: str
    n_evaluations: int
    fitted_prices: pd.DataFrame


def generate_synthetic_heston_chain(
    S0: float,
    r: float,
    q: float,
    params: HestonParams,
    strikes: list[float] | np.ndarray,
    maturities: list[float] | np.ndarray,
    option_type: str = "call",
    noise_std: float = 0.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a synthetic option chain from known Heston parameters."""

    rng = np.random.default_rng(seed)
    rows = []
    for maturity in maturities:
        for strike in strikes:
            clean_price = heston_price_cf(
                S0, float(strike), float(maturity), r, q, params, option_type
            )
            market_price = max(clean_price + rng.normal(0.0, noise_std), 1e-8)
            iv = implied_volatility(
                market_price,
                S0,
                float(strike),
                float(maturity),
                r,
                q,
                option_type,
                "brent",
            ).implied_volatility
            rows.append(
                {
                    "spot": S0,
                    "strike": float(strike),
                    "maturity": float(maturity),
                    "rate": r,
                    "dividend_yield": q,
                    "option_type": option_type,
                    "market_price": market_price,
                    "clean_heston_price": clean_price,
                    "implied_vol": iv,
                }
            )
    return pd.DataFrame(rows)


def _array_to_params(values: np.ndarray) -> HestonParams:
    return HestonParams(
        v0=float(values[0]),
        kappa=float(values[1]),
        theta=float(values[2]),
        sigma_v=float(values[3]),
        rho=float(values[4]),
    )


def heston_residuals(
    values: np.ndarray,
    option_chain: pd.DataFrame,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return Heston pricing residuals for scipy optimisation."""

    params = _array_to_params(values)
    residuals = []
    for row in option_chain.itertuples(index=False):
        model_price = heston_price_cf(
            float(row.spot),
            float(row.strike),
            float(row.maturity),
            float(row.rate),
            float(row.dividend_yield),
            params,
            str(row.option_type),
            integration_limit=60.0,
        )
        residuals.append(model_price - float(row.market_price))
    residual_array = np.asarray(residuals, dtype=float)
    if weights is not None:
        residual_array = residual_array * weights
    return residual_array


def calibrate_heston(
    option_chain: pd.DataFrame,
    initial_params: HestonParams | None = None,
    bounds: tuple[list[float], list[float]] | None = None,
    max_nfev: int = 80,
) -> HestonCalibrationResult:
    """Calibrate Heston parameters to option prices using least squares."""

    required = {
        "spot",
        "strike",
        "maturity",
        "rate",
        "dividend_yield",
        "option_type",
        "market_price",
    }
    missing = required.difference(option_chain.columns)
    if missing:
        raise ValueError(f"option_chain is missing required columns: {sorted(missing)}")

    if initial_params is None:
        initial_params = HestonParams(
            v0=0.05, kappa=1.2, theta=0.05, sigma_v=0.45, rho=-0.4
        )
    initial_params.validate()
    x0 = np.array(
        [
            initial_params.v0,
            initial_params.kappa,
            initial_params.theta,
            initial_params.sigma_v,
            initial_params.rho,
        ],
        dtype=float,
    )
    if bounds is None:
        bounds = ([1e-4, 0.05, 1e-4, 0.02, -0.95], [1.0, 8.0, 1.0, 2.0, 0.95])

    result = least_squares(
        heston_residuals,
        x0,
        args=(option_chain,),
        bounds=bounds,
        xtol=1e-6,
        ftol=1e-6,
        gtol=1e-6,
        max_nfev=max_nfev,
    )
    fitted_params = _array_to_params(result.x)

    fitted = option_chain.copy()
    fitted_prices = []
    for row in fitted.itertuples(index=False):
        fitted_prices.append(
            heston_price_cf(
                float(row.spot),
                float(row.strike),
                float(row.maturity),
                float(row.rate),
                float(row.dividend_yield),
                fitted_params,
                str(row.option_type),
                integration_limit=75.0,
            )
        )
    fitted["model_price"] = fitted_prices
    fitted["pricing_error"] = fitted["model_price"] - fitted["market_price"]
    rmse = float(np.sqrt(np.mean(np.square(fitted["pricing_error"]))))
    mae = float(np.mean(np.abs(fitted["pricing_error"])))
    return HestonCalibrationResult(
        params=fitted_params,
        rmse=rmse,
        mae=mae,
        success=bool(result.success),
        message=str(result.message),
        n_evaluations=int(result.nfev),
        fitted_prices=fitted,
    )


def parameter_comparison_table(
    true_params: HestonParams, recovered_params: HestonParams
) -> pd.DataFrame:
    """Return true vs recovered Heston parameter comparison."""

    rows = []
    for name in ["v0", "kappa", "theta", "sigma_v", "rho"]:
        true_value = getattr(true_params, name)
        recovered_value = getattr(recovered_params, name)
        rows.append(
            {
                "parameter": name,
                "true": true_value,
                "recovered": recovered_value,
                "absolute_error": abs(recovered_value - true_value),
            }
        )
    return pd.DataFrame(rows)
