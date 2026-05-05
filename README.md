# Derivatives Pricing Engine

A production-style Python derivatives pricing and risk analytics engine for equity options, built as a reusable package with tests, examples, notebooks, validation reports and a Streamlit dashboard.

Expected GitHub URL: <https://github.com/husaam-atq/derivatives-pricing-engine>

## Project Overview

This repository fills a derivatives-pricing gap in a quant finance portfolio. It implements the core pricing, risk, calibration and hedging workflows that appear in quant analyst, quant developer, equity derivatives and model validation interviews:

- Black-Scholes / Black-Scholes-Merton analytical pricing
- Analytical and finite-difference Greeks
- Robust implied volatility solving
- Cox-Ross-Rubinstein binomial trees for European and American options
- Monte Carlo pricing for vanilla, Asian and barrier options
- Variance reduction using antithetic and control variates
- Heston stochastic volatility simulation and characteristic-function pricing
- Heston calibration to synthetic option chains
- Volatility smile and surface interpolation
- Delta hedging simulation and hedging error analytics
- Scenario and stress testing
- Streamlit dashboard, executable notebooks and automated validation reports

The project avoids external market data dependencies. Sample data is synthetic and stored locally under `data/`.

## Why This Matters for Quant Finance

Derivative pricing is a core bridge between mathematical finance, numerical methods and production engineering. This project demonstrates:

- analytical model implementation and benchmark validation;
- numerical convergence testing for tree and Monte Carlo methods;
- risk sensitivity calculation with clearly documented conventions;
- calibration workflow design with realistic warnings about parameter non-uniqueness;
- hedge simulation that connects model Greeks to realised P&L;
- clean package boundaries suitable for reuse in notebooks, dashboards and scripts.

## Architecture

```text
derivatives-pricing-engine/
├── app/                         # Streamlit dashboard
├── data/                        # Synthetic sample option chain
├── examples/                    # Executable scripts
├── notebooks/                   # Executable analysis notebooks
├── reports/                     # Generated benchmark CSV and Markdown report
├── src/derivatives_engine/      # Reusable Python package
│   ├── calibration/             # Vol surface and Heston calibration
│   ├── models/                  # BSM, binomial, Monte Carlo, Heston
│   ├── risk/                    # Greeks, IV, hedging, scenarios
│   └── utils/                   # Market data, plotting, validation
└── tests/                       # Pytest coverage
```

Core logic lives under `src/derivatives_engine`. Notebooks, examples and the dashboard import package functions rather than duplicating formulas.

## Models Implemented

### Black-Scholes-Merton

European call and put prices are implemented with continuous dividend yield:

```text
C = S exp(-qT) N(d1) - K exp(-rT) N(d2)
P = K exp(-rT) N(-d2) - S exp(-qT) N(-d1)
```

with:

```text
d1 = [ln(S/K) + (r - q + 0.5 sigma^2)T] / [sigma sqrt(T)]
d2 = d1 - sigma sqrt(T)
```

### Greeks

Analytical Delta, Gamma, Vega, Theta and Rho are implemented for calls and puts. Finite-difference versions are provided for validation.

Conventions:

- Vega is per 1.00 volatility change. `vega_per_vol_point` reports per one volatility point.
- Theta is annualised calendar theta using `-dV/dT`; `daily_theta` divides by calendar days.
- Rho is per 1.00 rate change. `rho_per_bp` reports per basis point.

### Implied Volatility

The IV module supports:

- Brent root solving with no-arbitrage bounds;
- Newton-Raphson when stable;
- automatic Newton-to-Brent fallback;
- call and put support;
- vectorised helper over option chains.

### Binomial Tree

The CRR tree supports European and American calls/puts, continuous dividend yield and configurable step count. The validation report checks convergence to Black-Scholes at increasing step counts.

### Monte Carlo

The Monte Carlo module simulates risk-neutral GBM paths and prices:

- European vanilla calls and puts;
- arithmetic-average Asian options;
- up-and-out / down-and-out barrier options.

It reports discounted payoff estimates, standard errors and confidence intervals. Variance reduction includes antithetic variates and a terminal-stock control variate. A CuPy GPU path is available for GBM path generation if CuPy and CUDA are installed; otherwise it falls back cleanly to NumPy.

### Heston Stochastic Volatility

Heston paths are simulated with full truncation Euler:

```text
dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S
dv_t = kappa(theta - v_t)dt + sigma_v sqrt(v_t) dW_t^v
corr(dW_t^S, dW_t^v) = rho
```

Parameters:

- `v0`: initial variance
- `kappa`: variance mean reversion speed
- `theta`: long-run variance
- `sigma_v`: volatility of variance
- `rho`: spot/variance shock correlation

The package includes Monte Carlo Heston pricing and characteristic-function pricing for deterministic synthetic calibration.

### Heston Calibration

Calibration uses `scipy.optimize.least_squares` with sensible bounds. The example workflow:

1. Generate a synthetic option chain from known Heston parameters.
2. Calibrate parameters back to synthetic prices.
3. Report true vs recovered parameters, RMSE and MAE.

Heston calibration is non-unique and sensitive to data quality, objective scaling and strike/maturity coverage. The project reports fit quality without claiming market superiority.

### Volatility Surface

The surface module:

- enriches option chains with implied volatility;
- handles missing IV values carefully;
- interpolates across strikes and maturities;
- plots smiles and 3D surfaces with Plotly.

### Delta Hedging

The hedging simulator tracks:

- option value;
- option delta;
- stock hedge;
- cash account;
- transaction costs;
- final hedging P&L.

It compares rebalancing frequencies and demonstrates that, under Black-Scholes assumptions before transaction costs, more frequent rebalancing generally reduces hedging error dispersion.

## Validation Benchmark Summary

Generated by:

```bash
python examples/generate_validation_report.py
```

Current benchmark summary from `reports/benchmark_results.csv`:

| Area | Result |
| --- | --- |
| Black-Scholes call | 10.45058357 vs expected 10.4506 |
| Black-Scholes put | 5.57352602 vs expected 5.5735 |
| Put-call parity | absolute error 0.0 |
| Implied volatility | Brent and Newton recovered test vols within tolerance |
| Greeks | analytical vs finite-difference checks passed |
| Binomial tree | 1000-step call/put errors about 0.002 |
| Monte Carlo | 100,000-path vanilla errors passed statistical target |
| Variance reduction | antithetic and control variate reduced standard error |
| Heston simulation | valid shapes, non-negative variances, finite prices |
| Heston calibration | synthetic RMSE about 3.1e-05 |
| Delta hedging | daily hedging error std below monthly hedging error std |

Full details are in:

- `reports/validation_report.md`
- `reports/benchmark_results.csv`

## Screenshots

Add dashboard screenshots after pushing/running locally:

- `docs/images/dashboard_black_scholes.png`
- `docs/images/dashboard_heston.png`
- `docs/images/dashboard_hedging.png`
- `docs/images/vol_surface.png`

## Installation

```bash
git clone https://github.com/husaam-atq/derivatives-pricing-engine.git
cd derivatives-pricing-engine
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Run Tests

```bash
python -m compileall src app examples
python -m pytest -v
python -m ruff check .
python -m black --check .
```

## Run Examples

```bash
python examples/price_option.py
python examples/calibrate_heston.py
python examples/run_delta_hedging.py
python examples/generate_validation_report.py
```

## Run Notebooks

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/01_black_scholes_and_greeks.ipynb --output executed_01_black_scholes_and_greeks.ipynb
python -m jupyter nbconvert --to notebook --execute notebooks/02_monte_carlo_pricing.ipynb --output executed_02_monte_carlo_pricing.ipynb
python -m jupyter nbconvert --to notebook --execute notebooks/03_heston_model_and_calibration.ipynb --output executed_03_heston_model_and_calibration.ipynb
python -m jupyter nbconvert --to notebook --execute notebooks/04_delta_hedging_simulation.ipynb --output executed_04_delta_hedging_simulation.ipynb
```

Executed notebooks are ignored by git to keep the repository light.

## Run Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard includes:

- project overview and benchmark summary;
- Black-Scholes pricer;
- Greeks visualiser;
- implied volatility solver;
- binomial convergence;
- Monte Carlo pricer;
- Heston simulation;
- volatility surface;
- delta hedging simulator;
- scenario analysis.

## Example Usage

```python
from derivatives_engine.models.black_scholes import call_price
from derivatives_engine.risk.greeks import greek_table
from derivatives_engine.risk.implied_volatility import implied_volatility

S, K, T, r, q, sigma = 100, 100, 1, 0.05, 0.0, 0.20

call = call_price(S, K, T, r, q, sigma)
greeks = greek_table(S, K, T, r, q, sigma, "call")
iv = implied_volatility(call, S, K, T, r, q, "call", method="auto")

print(call)
print(greeks)
print(iv.implied_volatility)
```

## Results and Interpretation

The project validates analytical model outputs against known benchmarks and numerical estimates against statistical/convergence targets. Monte Carlo results are reported with confidence intervals and standard errors. Heston calibration is shown on synthetic data so that true parameters are known; this is useful for workflow validation, but it is not evidence that Heston is universally better than Black-Scholes.

## Limitations

- Models assume idealised market conventions and continuous rates/dividend yields.
- Black-Scholes assumes lognormal dynamics and constant volatility.
- Binomial trees become slower at very high step counts.
- Monte Carlo estimates have sampling error even with deterministic seeds.
- Heston Monte Carlo uses full truncation Euler and can have discretisation bias.
- Heston calibration can be non-unique and unstable with sparse/noisy option chains.
- Sample data is synthetic and not a live market feed.
- Transaction cost and liquidity modelling in hedging is intentionally simplified.

## Future Improvements

- Add local volatility and Dupire surface workflows.
- Add stochastic rates or rates curve bootstrapping.
- Add quasi-Monte Carlo Sobol paths.
- Add Longstaff-Schwartz American Monte Carlo.
- Add model-risk comparison reports across BSM, local vol and Heston.
- Add real option-chain ingestion behind an optional adapter.
- Add GPU kernels for larger path-dependent books.

## CV Bullet Examples

General:

> Built a Python derivatives pricing engine implementing Black-Scholes, Greeks, implied volatility solving, binomial trees, Monte Carlo pricing, Heston stochastic volatility, delta hedging simulation and volatility surface analytics, validated against analytical pricing benchmarks and numerical convergence tests.

Quant Analyst:

> Developed a validated equity derivatives analytics library covering BSM pricing, implied volatility, volatility surfaces, Heston calibration, scenario analysis and hedging P&L attribution, with benchmark reports documenting model accuracy and limitations.

Quant Developer:

> Engineered a modular Python pricing package under `src/` with typed APIs, deterministic pytest coverage, executable examples, Streamlit dashboard integration and optional CuPy acceleration fallback for Monte Carlo path generation.

Risk / Model Validation:

> Implemented benchmark validation for option pricing models, comparing analytical Greeks against finite differences, binomial convergence against Black-Scholes, Monte Carlo estimates against confidence intervals and synthetic Heston calibration against known parameters.

Equity Derivatives:

> Built an equity options pricing and risk toolkit with dividend-adjusted BSM, American option trees, implied volatility surfaces, Heston stochastic volatility, stress testing and delta hedging simulations across rebalancing frequencies.

## Interview Talking Points

- Why put-call parity is a useful pricing sanity check.
- Why vega, theta and rho conventions need to be explicit.
- How Brent and Newton implied-vol solvers fail differently.
- Why CRR trees converge to Black-Scholes for European options.
- How antithetic and control variates reduce Monte Carlo estimator variance.
- What full truncation Euler does in Heston variance simulation.
- Why Heston calibration can fit prices but still be parameter-degenerate.
- How hedge frequency, transaction costs and model assumptions interact.
- Why validation reports should show actual benchmark outcomes rather than claims.
