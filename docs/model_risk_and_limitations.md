# Model Risk and Limitations

This repository is benchmark-tested and validation-aware, but it is still a
portfolio-focused research project. The checks included here improve confidence
in the implementation, but they are not a substitute for formal model approval,
independent review or production risk governance.

## Black-Scholes Assumptions

The Black-Scholes-Merton model assumes:

- lognormal underlying dynamics;
- constant volatility;
- continuous trading;
- frictionless markets;
- continuous risk-free borrowing/lending;
- continuous dividend yield;
- no jumps or trading halts.

These assumptions are useful for analytical benchmarks but are simplified
relative to real equity derivatives markets.

## Constant Volatility Limitation

Black-Scholes uses a single volatility input. Real option markets often show
volatility smiles and skews across strike and maturity. A flat volatility
assumption cannot explain these patterns without using different implied vols
for different options.

## Lognormal Returns Limitation

GBM implies continuous paths and lognormal terminal prices. Real returns may
show jumps, heavy tails, volatility clustering and liquidity-driven gaps. These
features can materially affect short-dated, barrier and tail-sensitive payoffs.

## Liquidity and Transaction Costs

The delta hedging simulator includes optional simplified transaction costs, but
it does not model:

- bid-ask spreads dynamically;
- market impact;
- liquidity constraints;
- funding haircuts;
- discrete exchange trading hours;
- execution slippage.

The hedging examples should therefore be interpreted as model experiments under
controlled assumptions rather than trading P&L forecasts.

## Monte Carlo Sampling Error

Monte Carlo prices are statistical estimates. Deterministic random seeds make
examples reproducible, but they do not remove sampling error. Prices should be
read together with standard errors and confidence intervals.

## Tree Discretisation Error

Binomial trees approximate continuous-time dynamics on a finite grid. Increasing
the number of steps generally improves convergence for European options, but
finite step counts still introduce discretisation error.

## Heston Discretisation Bias

The Heston simulation uses full truncation Euler to keep variance non-negative.
This is practical and common for demonstrations, but it can introduce
discretisation bias. More advanced schemes may be needed for sensitive
production use cases.

## Heston Calibration Instability

Heston calibration can be unstable and non-unique. Different combinations of
\(v_0\), \(\kappa\), \(\theta\), \(\sigma_v\) and \(\rho\) may produce similar
prices, especially with sparse maturities, limited strike coverage or noisy
quotes.

The synthetic calibration workflow in this repository is useful for validating
the mechanics of calibration, not for proving that the recovered parameters are
unique or market-stable.

## Synthetic Data Limitation

The sample option chain and calibration workflows use synthetic data. This keeps
the project reproducible without external APIs, but it does not capture live
market microstructure, stale quotes, corporate actions, borrow costs or data
cleaning issues.

## Validation Is Not Formal Approval

The project validates implementations against analytical benchmarks, convergence
checks, confidence intervals and synthetic calibration tests. These checks are
important, but formal model approval would require broader independent review,
stress testing, data controls, governance documentation and production
monitoring.
