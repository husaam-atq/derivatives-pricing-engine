# Numerical Methods and Convergence

This note explains the numerical methods used to validate and extend the
closed-form models in the pricing engine. It focuses on methods currently
implemented in the repository and clearly marks PDE finite-difference pricing as
a future extension.

## 1. Why Numerical Methods Matter

Closed-form formulas are valuable, but they apply only under restrictive
assumptions. Numerical methods are essential when:

- early exercise features matter;
- payoffs are path-dependent;
- volatility is stochastic or state-dependent;
- analytical Greeks need independent validation;
- sampling uncertainty must be quantified.

This repository uses binomial trees, Monte Carlo simulation, variance reduction
and finite-difference Greeks to complement analytical Black-Scholes benchmarks.

## 2. Binomial Tree Convergence

The Cox-Ross-Rubinstein (CRR) tree approximates GBM by allowing the stock to move
up or down over each small time step. For time step \(dt = T / n\):

```text
u = exp(sigma sqrt(dt))
d = 1 / u
p = [exp((r-q)dt) - d] / (u - d)
```

Here \(p\) is the risk-neutral probability of an up move. Terminal payoffs are
computed at maturity, then discounted backward through the tree:

```text
V = exp(-r dt) [p V_up + (1-p) V_down]
```

For American options, each node compares continuation value with immediate
exercise value. For European options, CRR prices converge toward
Black-Scholes-Merton prices as the number of steps increases.

## 3. Monte Carlo Error

Monte Carlo estimates a discounted expectation using simulated payoffs:

```text
V_hat = exp(-rT) (1 / N) sum_i payoff_i
```

The sampling error is measured by the standard error:

```text
SE = sample_standard_deviation / sqrt(N)
```

An approximate 95% confidence interval is:

```text
V_hat +/- 1.96 SE
```

The convergence rate is:

```text
O(1 / sqrt(N))
```

This slow convergence means reducing standard error by a factor of 10 requires
roughly 100 times as many paths, unless variance reduction is used.

## 4. Antithetic Variates

Antithetic variates simulate paired shocks:

```text
Z and -Z
```

The estimator averages the two corresponding discounted payoffs. If the payoff
is monotonic or negatively correlated across paired shocks, the average can have
lower variance than independent sampling while preserving the same expectation.

## 5. Control Variates

A control variate uses a related random variable with known expectation. If
\(X\) is the target discounted payoff and \(Y\) has known expectation \(E[Y]\):

```text
X_cv = X - beta (Y - E[Y])
```

The repo's Monte Carlo module uses the discounted terminal stock value as a
control variate for vanilla options. Under the risk-neutral measure:

```text
E[exp(-rT) S_T] = S_0 exp(-qT)
```

Because this expectation is known, deviations in the simulated terminal stock
can be used to reduce estimator variance.

## 6. Finite-Difference Greeks

Finite-difference Greeks use bump-and-revalue calculations. Central differences
are used where possible:

```text
Delta ~= [V(S+h) - V(S-h)] / (2h)
Gamma ~= [V(S+h) - 2V(S) + V(S-h)] / h^2
Vega  ~= [V(sigma+h) - V(sigma-h)] / (2h)
Rho   ~= [V(r+h) - V(r-h)] / (2h)
```

The bump size \(h\) must balance truncation error and floating-point error. If
\(h\) is too large, the approximation is crude; if \(h\) is too small, numerical
rounding can dominate.

## 7. Optional PDE Finite-Difference Pricing

The repository derives and uses the Black-Scholes-Merton PDE conceptually, but
it does not currently implement a finite-difference PDE pricing solver.

A PDE finite-difference solver would discretise the option value over spot and
time grids, then step backward from terminal payoff to present value. This is a
natural future extension for comparing explicit, implicit and Crank-Nicolson
schemes against the analytical Black-Scholes benchmark.

## 8. Numerical Validation Philosophy

The repository validates numerical methods using multiple independent checks:

- textbook Black-Scholes call and put benchmarks;
- put-call parity;
- analytical Greeks compared with finite-difference Greeks;
- CRR convergence against Black-Scholes;
- Monte Carlo estimates with confidence intervals;
- variance-reduction standard error comparisons;
- synthetic Heston calibration with known true parameters;
- delta hedging simulations under idealised Black-Scholes assumptions.

The goal is not to claim formal model approval. The goal is to show disciplined,
reproducible, validation-aware quantitative engineering.
