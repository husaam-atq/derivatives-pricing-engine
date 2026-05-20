# Mathematical Foundations

This note explains the mathematical ideas behind the pricing engine. It is
written for the equity-option setting used in this repository: continuously
compounded rates, continuous dividend yield, European-style analytical
benchmarks, simulation under risk-neutral dynamics, and model validation against
known numerical checks.

## 1. Risk-Neutral Pricing

Modern derivatives pricing starts from the no-arbitrage idea that two
self-financing portfolios with the same future payoff should have the same
value today. In a complete market with suitable assumptions, derivative prices
can be represented as discounted expectations under a risk-neutral probability
measure, usually denoted by \(Q\).

For a European payoff depending on \(S_T\), the simplified pricing equation is:

```text
V_0 = exp(-rT) E^Q[payoff(S_T)]
```

Here:

- \(V_0\) is the option value today.
- \(r\) is the continuously compounded risk-free rate.
- \(T\) is time to maturity in years.
- \(E^Q[\cdot]\) is expectation under the risk-neutral measure.

This is the conceptual foundation used by the Black-Scholes formulas, Monte
Carlo estimators, binomial trees and Heston pricing workflows in this project.
For clarity, this note focuses on the equity-option case rather than the full
generality of arbitrage pricing theory.

## 2. Geometric Brownian Motion

The Black-Scholes model assumes that the underlying follows geometric Brownian
motion (GBM). Under the real-world measure \(P\), the dynamics are:

```text
dS_t = mu S_t dt + sigma S_t dW_t
```

where:

- \(S_t\) is the underlying price.
- \(\mu\) is the real-world expected return.
- \(\sigma\) is the volatility.
- \(W_t\) is Brownian motion.

For pricing, the drift is changed under the risk-neutral measure \(Q\). With a
continuous dividend yield \(q\), the dynamics become:

```text
dS_t = (r - q) S_t dt + sigma S_t dW_t^Q
```

The risk-neutral drift \(r - q\) reflects the cost of carry for a dividend-paying
equity. Volatility remains the key driver of option convexity and uncertainty,
while Brownian motion provides the random shock process.

## 3. Itô's Lemma

If an option value is a smooth function \(V(S,t)\), Itô's lemma gives the
stochastic differential of \(V\) when \(S_t\) follows GBM:

```text
dV = (partial V / partial t) dt
   + (partial V / partial S) dS
   + 0.5 (partial^2 V / partial S^2) (dS)^2
```

Substituting \(dS = \mu S dt + \sigma S dW_t\) and using the Itô rule
\((dW_t)^2 = dt\), this becomes:

```text
dV = [V_t + mu S V_S + 0.5 sigma^2 S^2 V_SS] dt
   + sigma S V_S dW_t
```

The second-order term matters because Brownian motion has quadratic variation:
\((dW_t)^2\) contributes at order \(dt\), while terms such as \(dt^2\) and
\(dt dW_t\) vanish in the limit.

## 4. Black-Scholes PDE Derivation

Let \(V(S,t)\) be the option value. Under GBM:

```text
dS = (r - q) S dt + sigma S dW_t^Q
```

Applying Itô's lemma:

```text
dV = [V_t + (r - q) S V_S + 0.5 sigma^2 S^2 V_SS] dt
   + sigma S V_S dW_t^Q
```

Construct a delta-hedged portfolio:

```text
Pi = V - Delta S
```

Choose:

```text
Delta = V_S
```

The stochastic \(dW_t^Q\) term is eliminated, leaving a locally riskless
portfolio. By no arbitrage, the hedged portfolio must earn the risk-free rate.
Accounting for continuous dividends, the Black-Scholes-Merton PDE is:

```text
partial V / partial t
+ (r - q) S partial V / partial S
+ 0.5 sigma^2 S^2 partial^2 V / partial S^2
- rV = 0
```

Equivalently:

```text
∂V/∂t + (r - q)S ∂V/∂S + 0.5 sigma^2 S^2 ∂²V/∂S² - rV = 0
```

This PDE is the theoretical source of the closed-form Black-Scholes prices
implemented in `src/derivatives_engine/models/black_scholes.py`.

## 5. Closed-Form Black-Scholes Formula

For a European call:

```text
C = S exp(-qT) N(d1) - K exp(-rT) N(d2)
```

For a European put:

```text
P = K exp(-rT) N(-d2) - S exp(-qT) N(-d1)
```

where:

```text
d1 = [ln(S/K) + (r - q + 0.5 sigma^2)T] / [sigma sqrt(T)]
d2 = d1 - sigma sqrt(T)
```

Interpretation:

- \(N(d1)\) is related to the option's hedge ratio under the dividend-adjusted
  model.
- \(N(d2)\) is related to the risk-neutral exercise probability.
- \(K exp(-rT)\) discounts the strike payment.
- \(S exp(-qT)\) adjusts the spot for continuous dividend yield.

These formulas are used as analytical benchmarks throughout the project.

## 6. Greeks and Sensitivities

Greeks measure how option value changes with respect to model inputs:

```text
Delta = partial V / partial S
Gamma = partial^2 V / partial S^2
Vega  = partial V / partial sigma
Theta = -partial V / partial T
Rho   = partial V / partial r
```

Conceptually:

- Delta measures first-order spot exposure.
- Gamma measures curvature and hedge instability.
- Vega measures volatility sensitivity.
- Theta measures calendar time decay under the convention used in this repo.
- Rho measures interest-rate sensitivity.

The package implements analytical Greeks and finite-difference Greeks.
Finite-difference Greeks are useful because they independently validate
closed-form expressions through bump-and-revalue tests.

## 7. Monte Carlo Pricing

Monte Carlo pricing estimates the risk-neutral expectation directly. Under GBM:

```text
S_T = S_0 exp((r - q - 0.5 sigma^2)T + sigma sqrt(T) Z)
```

with \(Z \sim N(0,1)\). A vanilla option estimator is:

```text
V_hat = exp(-rT) (1 / N) sum_i payoff(S_T_i)
```

The standard error is estimated from the discounted payoff samples:

```text
SE = sample_standard_deviation / sqrt(N)
```

A 95% confidence interval is approximately:

```text
V_hat +/- 1.96 SE
```

Monte Carlo convergence is slow but general:

```text
error = O(1 / sqrt(N))
```

This is why the repo reports both price estimates and confidence intervals in
`src/derivatives_engine/models/monte_carlo.py`.

## 8. Variance Reduction

Variance reduction aims to reduce estimator variance without changing the
pricing objective.

Antithetic variates simulate paired shocks \(Z\) and \(-Z\). For monotonic or
approximately monotonic payoffs, the paired payoff average can have lower
variance than two independent paths.

Control variates use a related random variable \(Y\) with known expectation
\(E[Y]\). A control-variate estimator has the form:

```text
X_cv = X - beta (Y - E[Y])
```

where \(X\) is the discounted payoff and \(\beta\) is chosen to reduce variance.
In the repo, the terminal stock value is used as a control because its
discounted expectation is known under the risk-neutral measure.

## 9. Heston Stochastic Volatility

The Heston model allows variance itself to be stochastic:

```text
dS_t = (r - q)S_t dt + sqrt(v_t)S_t dW_t^S
dv_t = kappa(theta - v_t)dt + sigma_v sqrt(v_t)dW_t^v
corr(dW_t^S, dW_t^v) = rho
```

Parameter meanings:

- \(v_0\): initial variance.
- \(\kappa\): speed of variance mean reversion.
- \(\theta\): long-run variance level.
- \(\sigma_v\): volatility of variance.
- \(\rho\): correlation between spot and variance shocks.

Constant-volatility Black-Scholes produces a flat implied volatility surface.
Stochastic volatility can generate smiles and skews because volatility varies
through time and can be correlated with spot moves.

## 10. Calibration Objective

Calibration fits model parameters to option data by minimising pricing errors.
A common least-squares objective is:

```text
min_theta sum_i (model_price_i(theta) - market_price_i)^2
```

Calibration can be unstable or non-unique because different parameter
combinations can produce similar option prices, especially with sparse strikes,
limited maturities, noisy market quotes or poorly scaled objectives. The Heston
calibration workflow in this repo therefore reports fit errors and warns about
identifiability rather than overclaiming parameter precision.

## 11. Summary

This project combines closed-form pricing, stochastic simulation, numerical
methods, calibration and validation. The implementation is intentionally
portfolio-focused and benchmark-tested: it connects the mathematical pricing
framework to executable Python modules, reports, notebooks and dashboard views.
