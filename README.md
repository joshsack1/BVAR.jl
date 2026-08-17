# BayesianVectorAutoregressions.jl

[![CI](https://github.com/joshsack1/BayesianVectorAutoregressions.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/joshsack1/BayesianVectorAutoregressions.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/joshsack1/BayesianVectorAutoregressions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/joshsack1/BayesianVectorAutoregressions.jl)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://joshsack1.github.io/BayesianVectorAutoregressions.jl/dev/)
[![Documentation](https://github.com/joshsack1/BayesianVectorAutoregressions.jl/actions/workflows/Documenter.yml/badge.svg)](https://github.com/joshsack1/BayesianVectorAutoregressions.jl/actions/workflows/Documenter.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Bayesian Vector Autoregressions in Julia*

`BayesianVectorAutoregressions.jl` provides an end-to-end framework for Vector Autoregressive (VAR) and Bayesian Vector Autoregressive (BVAR) modeling in Julia.
Designed for econometricians and data scientists, the package enables users to bring a standard `DataFrame` and proceed sequentially through every stage of time-series modeling:

> *On authorship: the code in this package was drafted by a language model from mathematical
> specifications written by the author, then reviewed against them; the documentation was
> almost entirely model-written. See [How this package was built](#how-this-package-was-built)
> below, or the [fuller statement in the docs](https://joshsack1.github.io/BayesianVectorAutoregressions.jl/dev/provenance/).*

1. **Unit Root & Cointegration Testing** (ADF tests, Johansen trace test)
2. **Lag Selection & Information Criteria** (AIC, BIC, HQ, FPE)
3. **Frequentist VAR Estimation** (Stacked OLS and Equation-by-Equation Fixed-Effect Models)
4. **Bayesian VAR Estimation** (9 prior families: Minnesota, Conjugate Normal-Wishart, Dummy-Observation priors, Independent NIW via Gibbs sampling, Asymmetric Conjugate, and Baumeister-Hamilton reference priors)
5. **Structural Identification & Impulse Responses** (Cholesky short-run identification, sign restrictions, and general Baumeister-Hamilton structural priors with short-run, sign, and long-run restrictions)

---

## Documentation

- [**Documentation (dev)**](https://joshsack1.github.io/BayesianVectorAutoregressions.jl/dev/) — the full manual and API
  reference
- [The five-stage pipeline](https://joshsack1.github.io/BayesianVectorAutoregressions.jl/dev/#The-five-stage-pipeline) —
  what each stage hands to the next, and the cross-stage contracts. Worth reading first.
- [Guide](https://joshsack1.github.io/BayesianVectorAutoregressions.jl/dev/guide/pre-estimation/) — a worked, executed
  walkthrough of every stage
- [API Reference](https://joshsack1.github.io/BayesianVectorAutoregressions.jl/dev/api/data-testing/) — generated from the
  docstrings

## Installation

Install from the General registry:

```julia
using Pkg
Pkg.add("BayesianVectorAutoregressions")
```

Julia 1.12 or later is required.

---

## Quickstart

Simulated macroeconomic data (`:gdp`, `:cpi`, `:ffr`) through the conjugate path. The
package name is long by design (registry naming rules); `import
BayesianVectorAutoregressions as BVAR` gives a short alias for qualified calls.

```julia
using BayesianVectorAutoregressions
using DataFrames
using Random

Random.seed!(123)
df = DataFrame(
    gdp = cumsum(0.5 .+ randn(150)),   # I(1) output
    cpi = cumsum(0.2 .+ randn(150)),   # I(1) price level
    ffr = 2.0 .+ 0.5 .* randn(150),    # policy rate
)
end_vars = [:gdp, :cpi, :ffr]

# Stage 1: unit-root and cointegration testing
adf_res = adf_tests(df, end_vars)
trace_stats, eigenvals = johansen_trace_test(df, end_vars, 2)

# Stage 2: lag selection. Note that these two helpers are not exported.
Y = BayesianVectorAutoregressions.get_endogenous(df, end_vars)
for p in 1:3
    r = BayesianVectorAutoregressions.generate_VARresult(Y, p)
    println("p=", p, "  AIC=", aic(r), "  BIC=", bic(r))
end

# Stage 3: reduced-form VAR(2)
est = estimate_var(df, end_vars, 2; include_constant = true, method = :ols)

# Stage 4: conjugate Normal-Wishart prior, then i.i.d. draws from the exact posterior
prior_nw = build_prior(df, end_vars, est, :normal_wishart)
draws_nw = sample_posterior(prior_nw, est; ndraws = 1000)

# Stage 5: recursive (Cholesky) identification and impulse responses
irf_short = identify_short_run(draws_nw; horizon = 20)
```

The remaining paths — the independent-NIW Gibbs sampler, sign restrictions, and the general
Baumeister-Hamilton structural framework with short-run, sign and long-run restrictions — are
covered in the [Guide](https://joshsack1.github.io/BayesianVectorAutoregressions.jl/dev/guide/pre-estimation/), where
every example is executed as part of the documentation build.

---

## Capability Comparison: `BayesianVectorAutoregressions.jl` vs BVAR (R)

`BayesianVectorAutoregressions.jl` was compared against [BVAR](https://github.com/nk027/bvar) (CRAN v1.0.5; Kuschnig & Vashold 2021, *Journal of Statistical Software* 100(14), [doi:10.18637/jss.v100.i14](https://doi.org/10.18637/jss.v100.i14)) — the standard hierarchical-Minnesota BVAR package in R. The comparison covers capabilities, numerical parity on a matched conjugate prior, and speed on matched tasks. The harness is fully reproducible and lives in `benchmark/rbvar-comparison/` (run `./run.sh`; requires R with the `BVAR` package installed).

### Capability Matrix

| Feature / Capability | `BayesianVectorAutoregressions.jl` | BVAR (R) |
|:---|:---:|:---:|
| **Pre-Estimation & Lag Selection** | | |
| ADF Unit-Root Tests | ✓ | — |
| Johansen Trace Test | ✓ | — |
| Information Criteria (AIC, BIC, HQ, FPE) | ✓ | — |
| Standalone OLS / FEM Reduced-Form VAR | ✓ | — |
| **Prior Families** | | |
| Prior Families (count) | **9 families** | **1 family** (hierarchical Minnesota-NIW + dummy extensions) |
| Minnesota Normal-Wishart Conjugate Prior | ✓ | ✓ |
| Sum-of-Coefficients / Dummy-Initial-Observation Priors | ✓ (composable — entire prior derived from stacked dummies) | ✓ (dummy rows appended to the parametric Minnesota prior, à la Theil mixed estimation)* |
| Asymmetric Conjugate Prior (Chan 2022) | ✓ | — |
| Baumeister-Hamilton (2019) Structural Prior | ✓ | — |
| Hyperparameter Selection | ◐ (point-optimizes the marginal likelihood via golden-section search) | ✓ (full hierarchical posterior via Metropolis-Hastings; Giannone-Lenza-Primiceri 2015) |
| **MCMC & Samplers** | | |
| Independent Normal-Inverse-Wishart Gibbs Sampler | ✓ | — |
| Exact i.i.d. Draws from the Closed-Form Conjugate Posterior | ✓ | — |
| Parallel Chains + `coda` Integration | — | ✓ |
| **Structural Identification & Post-Processing** | | |
| Recursive / Cholesky Identification | ✓ | ✓ |
| Sign Restrictions | ✓ | ✓ |
| Zero-and-Sign Restrictions (Arias-Rubio-Ramírez-Waggoner 2018) | — | ✓ |
| Long-Run Multiplier Restrictions (Blanchard-Quah, via restriction closures) | ✓ | — |
| Baumeister-Hamilton $p(A)$ Structural Framework | ✓ | — |
| **Forecasting & Diagnostics** | | |
| Unconditional & Conditional Forecasts (Waggoner-Zha 1999) | — | ✓ |
| Forecast Error Variance Decomposition (FEVD) | — | ✓ |
| Historical Decomposition | — | ✓ |
| Model Diagnostics (WAIC, Log Predictive Scores, RMSE) | — | ✓ |
| Plotting Methods | — | ✓ |
| Bundled FRED-QD/MD Data with Transforms | — | ✓ |

\* Composition semantics differ: `BayesianVectorAutoregressions.jl` derives the entire prior from stacked dummy observations, while R's `BVAR` appends dummy rows to the parametric Minnesota prior in the style of Theil mixed estimation. Both are ✓ but the two are not interchangeable implementations of the same construction.

### Posterior Parity

The two packages were matched on the plain Minnesota Normal-Wishart conjugate prior with fixed hyperparameters: overall tightness λ1 ↔ `lambda` (0.2, both defaults), lag decay λ3 ↔ `alpha` (`alpha = 2λ3`, both defaults), prior scale σ̂ⱼ² computed in Julia and passed to R explicitly as `psi`, constant-term variance λ4² = `var` = 1e7, a random-walk prior mean, and identical Wishart degrees of freedom. Both packages use the same design-matrix layout (constant, then lag-major blocks), so posterior moments compare elementwise with no permutation needed.

R's `BVAR` cannot run in a fully non-hierarchical mode — it requires at least one hierarchical hyperparameter — so λ was pinned via a near-degenerate hyperprior (sd 1e-6, with the Metropolis-Hastings proposal scaled to match); the realized λ draws had sd ≈ 5e-11 around 0.2, with an 80% MH acceptance rate.

With 20,000 stored R draws per model size, every posterior mean of β and Σ agrees with the Julia **closed-form** posterior within Monte-Carlo error: max |z| = 3.20 across all elements and all three model sizes (gate: 5), max absolute deviation ≤ 1.2 × 10⁻³. A secondary two-sample check of R's Monte-Carlo means against Julia's own 20,000-draw sampler also passes (max |z| = 2.88). R's Monte-Carlo standard deviations match the analytic matrix-t / inverse-Wishart marginal sds within ±1.5%.

This parity is statistical, not bitwise — there is no shared RNG across languages — and it is scoped to this one matched conjugate configuration. It does not cover R's hierarchical posterior (which `BayesianVectorAutoregressions.jl` does not implement) or the Julia-only prior families.

### Speed

Apple M4, macOS 26.6, single-threaded BLAS on both sides, Julia 1.12.6 + OpenBLAS, R 4.6.1 + Homebrew OpenBLAS. Julia timings are `BenchmarkTools` medians; R timings are medians of 5 timed reps after warmup.

| Model size | Task | Julia | R BVAR | Ratio |
|:---|:---|---:|---:|---:|
| S (T=200, n=3, p=2) | 1000 stored posterior draws | 1.1 ms | 150 ms | 134× |
| S (T=200, n=3, p=2) | Cholesky IRFs (21 periods × 1000 draws) | 2.9 ms | 29 ms | 10.0× |
| M (T=800, n=6, p=4) | 1000 stored posterior draws | 3.2 ms | 347 ms | 108× |
| M (T=800, n=6, p=4) | Cholesky IRFs (21 periods × 1000 draws) | 23.6 ms | 104 ms | 4.4× |
| L (T=400, n=10, p=5) | 1000 stored posterior draws | 11.7 ms | 536 ms | 46× |
| L (T=400, n=10, p=5) | Cholesky IRFs (21 periods × 1000 draws) | 125 ms | 424 ms | 3.4× |

These ratios are end-to-end costs of the same user-facing task, not kernel benchmarks, and the architectures being timed are not the same shape. R's `bvar()` has no non-hierarchical mode: producing 1000 stored draws necessarily includes a one-off L-BFGS-B marginal-likelihood optimization, 1000 burn-in iterations, and a Metropolis-Hastings step with a marginal-likelihood evaluation on every iteration. `BayesianVectorAutoregressions.jl` instead draws i.i.d. from the closed-form posterior with hyperparameters taken as given. That architectural difference — not raw linear-algebra speed — accounts for most of the gap in the draws row. The IRF ratio compares against R's bare IRF computation loop; R's public `irf()` also computes quantile bands by default, which the Julia function does not, and was timed separately (slightly slower still).

---

## How this package was built

`BayesianVectorAutoregressions.jl` was written with substantial help from a large language model. The Julia General
registry does not require anyone to say so; this is here because econometric software fails
quietly, and you should be able to calibrate how much to trust this package instead of
guessing.

- **The mathematics is the author's** — every method was chosen and specified from the
  literature cited in the docstrings, not derived by a model.
- **The implementation code in `src/` was drafted by a language model** from those
  specifications and then reviewed line-by-line against them.
- **The tests were written jointly**: the author specified what to verify, the model helped
  write the assertions.
- **The documentation was almost entirely model-written.** Every code example in it is
  executed on every docs build, so the code is verified. The prose and citations were
  reviewed but not audited claim-by-claim.

The failure mode worth worrying about here is not a crash — it is a result that is
plausible, correctly typed, and wrong: a coefficient block sliced in the wrong order, a
dropped normalization. The test suite targets that class of error directly, with
hand-computed known-answer tests, reduction identities, cross-method agreement between the
`:ols`/`:fem` estimators and the `:sir`/`:mh` samplers, and simulation-based coefficient
recovery. It runs [on every push](https://github.com/joshsack1/BayesianVectorAutoregressions.jl/actions/workflows/CI.yml)
against both the declared Julia floor and latest stable, so the badge above is the evidence —
you do not have to take the claim on faith.
It has *not* been validated by replicating any cited paper's published results.
If you are using this for research, reproduce something you already know the answer to
first, and check formulas against the papers the docstrings cite.

However the code was drafted, the author is responsible for it, and intends to fix what
turns out to be broken. Bug reports — including against the prose — are welcome and
unusually valuable here. The full statement is
[in the documentation](https://joshsack1.github.io/BayesianVectorAutoregressions.jl/dev/provenance/).
