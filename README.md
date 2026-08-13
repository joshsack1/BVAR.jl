# BVAR.jl

*Bayesian Vector Autoregressions in Julia*

`BVAR.jl` provides an end-to-end framework for Vector Autoregressive (VAR) and Bayesian Vector Autoregressive (BVAR) modeling in Julia.
Designed for econometricians and data scientists, the package enables users to bring a standard `DataFrame` and proceed sequentially through every stage of time-series modeling:

1. **Unit Root & Cointegration Testing** (ADF tests, Johansen trace test)
2. **Lag Selection & Information Criteria** (AIC, BIC, HQ, FPE)
3. **Frequentist VAR Estimation** (Stacked OLS and Equation-by-Equation Fixed-Effect Models)
4. **Bayesian VAR Estimation** (9 prior families: Minnesota, Conjugate Normal-Wishart, Dummy-Observation priors, Independent NIW via Gibbs sampling, Asymmetric Conjugate, and Baumeister-Hamilton reference priors)
5. **Structural Identification & Impulse Responses** (Cholesky short-run identification, sign restrictions, and general Baumeister-Hamilton structural priors with short-run, sign, and long-run restrictions)

---

## Quickstart & Minimal Example

The following example demonstrates the complete workflow using simulated macroeconomic data (`:gdp`, `:cpi`, `:ffr`).

```julia
using BVAR
using DataFrames
using Distributions
using LinearAlgebra
using Random

# -----------------------------------------------------------------------------
# 1. Prepare Data
# -----------------------------------------------------------------------------
Random.seed!(123)
n_obs = 150
df = DataFrame(
    gdp = cumsum(0.5 .+ randn(n_obs)),   # I(1) output
    cpi = cumsum(0.2 .+ randn(n_obs)),   # I(1) price level
    ffr = 2.0 .+ 0.5 .* randn(n_obs),   # Policy rate
)
end_vars = [:gdp, :cpi, :ffr]

# -----------------------------------------------------------------------------
# 2. Pre-Estimation: Unit Root & Cointegration Testing + Lag Selection
# -----------------------------------------------------------------------------
# Augmented Dickey-Fuller (ADF) tests (returns Type 1, 2, and 4 test objects)
adf_res = adf_tests(df, end_vars)

# Johansen Trace Test for Cointegration
trace_stats, eigenvals = johansen_trace_test(df, end_vars, 2)

# Information Criteria for Lag Selection
Y = get_endogenous(df, end_vars)
var_res_p1 = generate_VARresult(Y, 1)
var_res_p2 = generate_VARresult(Y, 2)

println("AIC (p=1): ", aic(var_res_p1), " | AIC (p=2): ", aic(var_res_p2))
println("BIC (p=1): ", bic(var_res_p1), " | BIC (p=2): ", bic(var_res_p2))

# -----------------------------------------------------------------------------
# 3. Frequentist Reduced-Form VAR
# -----------------------------------------------------------------------------
est = estimate_var(df, end_vars, 2; include_constant = true, method = :ols)

# -----------------------------------------------------------------------------
# 4. Bayesian VAR Estimation
# -----------------------------------------------------------------------------
# Path A: Conjugate Normal-Wishart (Direct, i.i.d. draws from exact posterior)
prior_nw = build_prior(df, end_vars, est, :normal_wishart)
draws_nw = sample_posterior(prior_nw, est; ndraws = 1000)

# Path B: Independent Normal-Inverse-Wishart (Sampled via Turing Gibbs sampler)
prior_niw = build_prior(
    df,
    end_vars,
    est,
    :independent_niw;
    hyperparameter_method = :fixed,
    hyperparameters = (λ1 = 0.2, λ2 = 0.5, λ3 = 1.0, λ4 = 1e5),
)
draws_niw = sample_posterior(prior_niw, est; ndraws = 1000)

# -----------------------------------------------------------------------------
# 5. Impulse Response Functions & Structural Identification
# -----------------------------------------------------------------------------

# (a) Short-Run Recursive Identification (Cholesky / Sims 1980)
irf_short = identify_short_run(draws_nw; horizon = 20)

# (b) Sign Restrictions Identification (Uhlig 2005 / RWZ 2010)
# Example pattern: shock 1 has positive contemporaneous impact on gdp and cpi
sign_pattern = [
     1  0  0;  # gdp response to shock 1 > 0
     1  0  0;  # cpi response to shock 1 > 0
     0  0  0   # ffr response unrestricted
]
irf_sign = identify_sign_restrictions(draws_nw, sign_pattern; horizon = 20)

# (c) General Baumeister-Hamilton Structural Identification (Short-Run, Sign & Long-Run)
rf_prior = build_prior(df, end_vars, est, :hamilton_baumeister)

n = length(end_vars)
template = Matrix(1.0I, n, n)
free = falses(n, n)
free[2, 1] = true  # Allow cpi to respond contemporaneously to gdp shock

# Component prior with sign/bound restriction on A[2,1]
component = Dict((2, 1) => Truncated(Normal(0.0, 1.0), -2.0, 0.0))

# Long-run sign restriction: cumulative long-run effect of shock 1 on gdp is positive
lr_restriction = long_run_sign_restriction(1, 1, 1, est.lags, est.include_constant)

A_prior = structural_prior(
    template,
    free,
    component;
    restrictions = Function[lr_restriction],
    names = end_vars,
)
struct_prior = hamilton_structural_prior(rf_prior, A_prior, Y)

# Sample structural posterior (A, B, D) via Metropolis-Hastings or SIR
s_draws, diagnostics = sample_structural(struct_prior, est; ndraws = 1000, method = :mh)
irf_struct = impulse_response(s_draws; horizon = 20)
```

---

## Capability Comparison: `BVAR.jl` vs [`BayesianVARs.jl`](https://github.com/elenev/BayesianVARs.jl)

`BVAR.jl` was evaluated head-to-head against `BayesianVARs.jl` across functionality, posterior precision, and benchmark execution speed. Below is a summary of the findings (for full benchmark details, see `BVAR-Pkg-Comparison-2026-08-13.qmd`).

### Capability Matrix

| Feature / Capability | `BVAR.jl` | `BayesianVARs.jl` |
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

1. **Broad Prior Ecosystem**: `BVAR.jl` supports 9 distinct prior specifications (including composable dummy observation priors, Chan's asymmetric conjugate prior, and Baumeister-Hamilton reference priors), compared to `BayesianVARs.jl`'s Minnesota-only scope.
2. **Speed & Efficiency**:
   - **Gibbs Sampler**: `BVAR.jl`'s Gibbs driver executes analytic conditional draws directly, achieving **3.0×–13× faster** runtime compared to `BayesianVARs.jl`'s data re-collapse loop.
   - **IRF Transformation**: Computing IRFs over posterior draws is **2.1×–4.2× faster**.
3. **Exact Posterior Parity**: Across all matched conjugate priors, `BVAR.jl` agrees with `BayesianVARs.jl` to machine precision ($\max |\Delta| \le 3.3 \times 10^{-10}$), confirming exact analytical equivalence.
