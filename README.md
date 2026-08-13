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

`BayesianVectorAutoregressions.jl` is not registered in the General registry, so install it by URL:

```julia
using Pkg
Pkg.add(url = "https://github.com/joshsack1/BayesianVectorAutoregressions.jl")
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

## Capability Comparison: `BayesianVectorAutoregressions.jl` vs [`BayesianVARs.jl`](https://github.com/elenev/BayesianVARs.jl)

`BayesianVectorAutoregressions.jl` was evaluated head-to-head against `BayesianVARs.jl` across functionality, posterior precision, and benchmark execution speed. Both packages were run on identical simulated data with matched priors and draw counts; timings come from `BenchmarkTools` and the parity check compares posterior moments elementwise. The benchmark harness lives in `benchmark/` and is run with `julia --project=benchmark benchmark/benchmarks.jl`. A summary of the findings follows.

### Capability Matrix

| Feature / Capability | `BayesianVectorAutoregressions.jl` | `BayesianVARs.jl` |
|:---|:---:|:---:|
| **Pre-Estimation & Lag Selection** | | |
| ADF Unit-Root Tests | ✓ (3 deterministic specs) | — |
| Johansen Trace Test | ✓ (statistics & eigenvalues) | — |
| Information Criteria (AIC, BIC, HQ, FPE) | ✓ | — |
| Standalone OLS / FEM Reduced-Form VAR | ✓ | — |
| **Prior Families** | **9 families** | **1 family** (Minnesota) |
| Minnesota Prior (Litterman 1986) | ✓ (with cross-equation λ₂ asymmetry) | ✓ (3 structural variants) |
| Natural-Conjugate Normal-Wishart | ✓ | ✓ |
| Dummy-Observation Minnesota (BGR 2010) | ✓ (composable) | — |
| Sum-of-Coefficients (DLS 1984) | ✓ (composable) | — |
| Dummy Initial Observation (Sims 1993) | ✓ (composable) | — |
| Prior for the Long Run (GLP 2019) | ✓ (composable) | — |
| Independent Normal-Inverse-Wishart | ✓ (Gibbs sampler) | ✓ (Gibbs sampler) |
| Asymmetric Conjugate (Chan 2022) | ✓ (closed form) | — |
| Baumeister-Hamilton (2019) Reference Prior | ✓ (reduced & structural) | — |
| Automatic Hyperparameter Selection | ✓ (Golden-section search) | — |
| **MCMC & Samplers** | | |
| Closed-Form Conjugate Posterior Sampling | ✓ (i.i.d. draws) | ✓ (distribution objects) |
| Gibbs Sampler Performance | **3.0× – 13× faster** | Baseline (re-collapses data every sweep) |
| **Structural Identification & Post-Processing** | | |
| Recursive / Cholesky Identification | ✓ | ◐ (IRF method only) |
| Sign Restrictions (Uhlig 2005; RWZ 2010) | ✓ (QR rotations) | — |
| Baumeister-Hamilton $p(A)$ Structural Framework | ✓ (SIR & MH with ESS/Acceptance) | — |
| Long-Run Multiplier Restrictions (Blanchard-Quah) | ✓ (via restriction closures) | — |
| Posterior IRF Calculation Speed | **2.1× – 4.2× faster** | Baseline |

### Key Highlights

1. **Broad Prior Ecosystem**: `BayesianVectorAutoregressions.jl` supports 9 distinct prior specifications (including composable dummy observation priors, Chan's asymmetric conjugate prior, and Baumeister-Hamilton reference priors), compared to `BayesianVARs.jl`'s Minnesota-only scope.
2. **Speed & Efficiency**:
   - **Gibbs Sampler**: `BayesianVectorAutoregressions.jl`'s Gibbs driver executes analytic conditional draws directly, achieving **3.0×–13× faster** runtime compared to `BayesianVARs.jl`'s data re-collapse loop.
   - **IRF Transformation**: Computing IRFs over posterior draws is **2.1×–4.2× faster**.
3. **Exact Posterior Parity**: Across all matched conjugate priors, `BayesianVectorAutoregressions.jl` agrees with `BayesianVARs.jl` to machine precision ($\max |\Delta| \le 3.3 \times 10^{-10}$), confirming exact analytical equivalence.

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
