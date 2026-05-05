# Validation Report

This report is generated from live benchmark checks in `derivatives_engine.utils.validation`.

## Executive Summary

- **Overall result:** 26/26 benchmark checks passed.
- **Scope:** analytical pricing, Greeks, implied volatility, tree convergence, Monte Carlo error bars, Heston simulation/calibration and delta hedging.
- **Interpretation:** the engine matches known analytical references and passes numerical sanity checks; stochastic and calibration outputs are reported with appropriate limitations.
- **Model governance note:** this is a benchmark suite for a portfolio project, not a formal model approval document.

## Headline Results

| area | result | evidence |
| --- | --- | --- |
| Black-Scholes | Call/put textbook benchmarks within tolerance | Call error 1.64e-05; put error 2.60e-05 |
| Parity | Put-call parity holds for analytical prices | Absolute error 0.0 |
| Implied volatility | Brent and Newton recover known volatilities | All IV recovery checks passed |
| Binomial tree | CRR converges to Black-Scholes | 1000-step call/put error about 0.0020 |
| Monte Carlo | Vanilla estimates pass statistical targets | 100k path call error 0.0068; put error 0.0740 |
| Variance reduction | Both methods reduce standard error | Plain SE 0.04662; antithetic 0.03303; control 0.01783 |
| Heston calibration | Synthetic calibration fits deterministic Heston prices | RMSE 3.12e-05 |
| Delta hedging | Daily rebalancing reduces hedging error dispersion | Daily std 0.4257; monthly std 1.9104 |

## Benchmark Summary

| category | benchmark | metric | value | target | passed | notes |
| --- | --- | --- | --- | --- | --- | --- |
| Black-Scholes | Textbook call benchmark | absolute_error | 1.6427814e-05 | < 1e-4 | True | price=10.45058357 |
| Black-Scholes | Textbook put benchmark | absolute_error | 2.6022257e-05 | < 1e-4 | True | price=5.57352602 |
| Black-Scholes | Put-call parity | absolute_error | 0 | < 1e-8 | True |  |
| Implied volatility | Brent recovers sigma=0.15 | absolute_error | 3.0531133e-16 | < 1e-6 | True |  |
| Implied volatility | Newton recovers sigma=0.15 | absolute_error | 5.1680882e-14 | < 1e-5 | True |  |
| Implied volatility | Brent recovers sigma=0.20 | absolute_error | 0 | < 1e-6 | True |  |
| Implied volatility | Newton recovers sigma=0.20 | absolute_error | 2.7755576e-17 | < 1e-5 | True |  |
| Implied volatility | Brent recovers sigma=0.35 | absolute_error | 3.2474023e-14 | < 1e-6 | True |  |
| Implied volatility | Newton recovers sigma=0.35 | absolute_error | 2.7755576e-16 | < 1e-5 | True |  |
| Greeks | Analytical vs finite-difference delta | absolute_error | 8.2740148e-11 | < 0.0001 | True | Theta uses annualised calendar -dV/dT convention. |
| Greeks | Analytical vs finite-difference gamma | absolute_error | 2.6811328e-10 | < 0.0001 | True | Theta uses annualised calendar -dV/dT convention. |
| Greeks | Analytical vs finite-difference vega | relative_error | 8.1143472e-09 | < 0.001 | True | Theta uses annualised calendar -dV/dT convention. |
| Greeks | Analytical vs finite-difference theta | absolute_error | 4.9198068e-09 | < 0.002 | True | Theta uses annualised calendar -dV/dT convention. |
| Greeks | Analytical vs finite-difference rho | relative_error | 1.4917859e-10 | < 0.001 | True | Theta uses annualised calendar -dV/dT convention. |
| Binomial tree | CRR call convergence at 1000 steps | absolute_error | 0.0019994684 | < 0.02 | True |  |
| Binomial tree | CRR put convergence at 1000 steps | absolute_error | 0.0019994684 | < 0.02 | True |  |
| Monte Carlo | European call vs Black-Scholes | absolute_error | 0.0067662449 | < 0.15 or analytical inside 95% CI | True | price=10.457350, analytical=10.450584, CI=(10.365946, 10.548753) |
| Monte Carlo | European put vs Black-Scholes | absolute_error | 0.073980242 | < 0.15 or analytical inside 95% CI | True | price=5.647506, analytical=5.573526, CI=(5.593478, 5.701535) |
| Monte Carlo | antithetic standard error reduction | standard_error | 0.033028026 | < plain SE 0.046623 | True |  |
| Monte Carlo | control_variate_stock standard error reduction | standard_error | 0.017834804 | < plain SE 0.046623 | True |  |
| Heston | Simulation output shapes | passed | True | True | True |  |
| Heston | Full truncation non-negative variance | passed | True | True | True |  |
| Heston | Monte Carlo option price finite | price | 9.1002025 | finite and > 0 | True | SE=0.080951 |
| Heston calibration | Synthetic chain pricing fit | RMSE | 3.1242831e-05 | < 0.05 | True | Synthetic calibration is deterministic but can remain non-unique. |
| Heston calibration | Optimizer success flag | success | True | True | True | `gtol` termination condition is satisfied. |
| Delta hedging | Daily vs monthly hedging error | std_pnl | 0.42570136 | < monthly std 1.910362 | True | No transaction costs; same random seed by frequency. |

## Greek Convention

Vega is reported per 1.00 volatility change, with helper output available
per one volatility point. Rho is per 1.00 rate change, with helper
output available per basis point. Theta is annualised calendar theta
using the `-dV/dT` convention; daily theta divides by calendar days.

## Binomial Convergence

European CRR prices converge toward Black-Scholes as the number of steps
increases. The benchmark checks the 1000-step error target for calls
and puts.

| steps | binomial_price | black_scholes_price | absolute_error |
| --- | --- | --- | --- |
| 10 | 10.253409 | 10.450584 | 0.19717453 |
| 25 | 10.520966 | 10.450584 | 0.070382052 |
| 50 | 10.410692 | 10.450584 | 0.039892031 |
| 100 | 10.430612 | 10.450584 | 0.01997191 |
| 250 | 10.442589 | 10.450584 | 0.0079948596 |
| 500 | 10.446585 | 10.450584 | 0.0039984357 |
| 1000 | 10.448584 | 10.450584 | 0.0019994684 |

## Monte Carlo Variance Reduction

| method | price | standard_error | ci_lower | ci_upper | standard_error_reduction_pct |
| --- | --- | --- | --- | --- | --- |
| plain | 10.505731 | 0.046623193 | 10.414352 | 10.597111 | 0 |
| antithetic | 10.449595 | 0.033028026 | 10.384861 | 10.514329 | 29.159666 |
| control_variate_stock | 10.468043 | 0.017834804 | 10.433088 | 10.502999 | 61.746928 |

## Heston Calibration

Synthetic Heston calibration uses deterministic characteristic-function
prices. The recovered parameters should be interpreted cautiously
because Heston calibration can be non-unique and sensitive to
strike/maturity coverage, objective scaling, and market data quality.

| parameter | true | recovered | absolute_error |
| --- | --- | --- | --- |
| v0 | 0.04 | 0.039999308 | 6.9220509e-07 |
| kappa | 1.4 | 1.3996339 | 0.00036605988 |
| theta | 0.04 | 0.039998944 | 1.0563064e-06 |
| sigma_v | 0.35 | 0.34989156 | 0.00010843999 |
| rho | -0.55 | -0.55008448 | 8.4480255e-05 |

## Delta Hedging

The hedging benchmark compares rebalancing frequencies under
Black-Scholes assumptions with zero transaction costs. More frequent
rebalancing should generally reduce hedging error dispersion in this
idealised setting.

| rebalances_per_year | rebalance_every_steps | mean_pnl | std_pnl | p05 | p50 | p95 | mean_transaction_costs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 21 | 0.03131783 | 1.9103616 | -3.0660951 | 0.044874315 | 3.1919661 | 0 |
| 52 | 5 | 0.0168025 | 0.91732945 | -1.4486237 | 0.0014866628 | 1.5633548 | 0 |
| 252 | 1 | 0.0052236678 | 0.42570136 | -0.71222 | 0.0040084948 | 0.69665196 | 0 |

## Limitations

- Monte Carlo checks are deterministic by seed but still represent statistical estimators.
- Heston Monte Carlo uses full truncation Euler; discretisation bias is possible.
- Heston calibration is intentionally demonstrated on synthetic data and should not be treated as live-market evidence.
- The validation suite checks credible benchmark behaviour; it is not a model approval document.
