```@meta
CurrentModule = BVAR
```

# BVAR.jl

*Bayesian Vector Autoregressions in Julia*

`BVAR.jl` provides an end-to-end framework for Vector Autoregressive (VAR) and Bayesian
Vector Autoregressive (BVAR) modeling. You bring a standard `DataFrame` and proceed
sequentially through every stage of time-series modeling:

1. **Unit root & cointegration testing** — [`adf_tests`](@ref), [`johansen_trace_test`](@ref)
2. **Lag selection** — [`aic`](@ref), [`bic`](@ref), [`hq`](@ref), [`fpe`](@ref)
3. **Frequentist VAR estimation** — [`estimate_var`](@ref), by stacked OLS or
   equation-by-equation fixed-effect models
4. **Bayesian VAR estimation** — [`build_prior`](@ref) and [`sample_posterior`](@ref), across
   nine prior families
5. **Structural identification & impulse responses** — [`identify_short_run`](@ref),
   [`identify_sign_restrictions`](@ref), [`sample_structural`](@ref),
   [`impulse_response`](@ref)

## Installation

`BVAR.jl` is not registered in the General registry, so install it by URL:

```julia
using Pkg
Pkg.add(url = "https://github.com/joshsack1/BVAR.jl")
```

Or, to hack on it:

```julia
using Pkg
Pkg.develop(url = "https://github.com/joshsack1/BVAR.jl")
```

Julia 1.11 or later is required.

## A single flat namespace

Despite the subdirectory layout of `src/`, `BVAR` defines **no submodules**. Every name —
exported or not — lives directly in `BVAR`. Unexported helpers are reached as
`BVAR.get_endogenous`, and they are documented in the "Internals" section of each
[API Reference](@ref api-data-testing) page.

## The five-stage pipeline

Each stage consumes the object the previous stage returned. The chain is:

```
DataFrame + Vector{Symbol}
        │
        ├─▶ adf_tests / johansen_trace_test        (stage 1, diagnostic only)
        ├─▶ aic / bic / hq / fpe  ← VARresult      (stage 2, diagnostic only)
        │
        ▼   estimate_var
   VARestimate
        │
        ▼   build_prior
   MinnesotaPrior | NormalWishartPrior | IndependentNIWPrior
   AsymmetricConjugatePrior | BaumeisterHamiltonPrior     (all <: AbstractVARPrior)
        │
        ▼   sample_posterior
    BVARdraws ──────────────┬──▶ identify_short_run          ─┐
        │                   └──▶ identify_sign_restrictions   ─┤
        ▼   hamilton_structural_prior + sample_structural      │
  StructuralDraws ──────────────▶ impulse_response           ─┤
                                                              ▼
                                                          IRFdraws
```

Stages 1 and 2 are diagnostic: nothing downstream consumes their output, so you can skip them
if you already know your lag order and integration orders.

### Contracts worth knowing before you start

These are the places where a mistake is silent rather than loud.

- **`build_prior` re-derives statistics from `df` that a `VARestimate` does not carry.** It
  needs the univariate AR residual variances and the sample means, so it takes `df` and
  `end_vec` *alongside* `est`. All three must describe the same data: `end_vec` must equal
  `est.names` **in the same order**, and `df` must have the same number of rows as the data
  `est` was fit on. Otherwise the two sources of statistics are mismatched.

- **Coefficient matrices are stored `k × n`, variables in columns.** The row order is: the
  constant first (if `include_constant`), then lag 1 of *every* variable, then lag 2 of every
  variable, and so on — so `k == include_constant + lags * vars`. This single convention is
  shared by `VARestimate.β_hat`, `BVARdraws.β`, and the structural `B`, and
  [`lag_blocks`](@ref) is what converts it into the ``\Phi_1,\ldots,\Phi_p`` matrices of the
  textbook column-vector form. Slicing those rows by hand is the easiest way to get a wrong
  answer that still runs.

- **`sample_posterior` is one entry point over two very different mechanisms.** Which you get
  is determined entirely by the prior type you hand it — see
  [4b. Posterior Sampling](@ref "Posterior Sampling: Closed Form and Gibbs").

- **`ndraws` does not always mean the same thing.** For the conjugate families it is a count
  of independent draws from the exact posterior; for `IndependentNIWPrior` it is a count of
  Gibbs sweeps, which are autocorrelated.

- **Some helpers the README-style workflow needs are not exported.**
  `BVAR.get_endogenous` and `BVAR.generate_VARresult` must be qualified.

## Quickstart

```@setup quickstart
include("plot-theme.jl")
```

```@example quickstart
using BVAR
using DataFrames
using Random

Random.seed!(123)
df = DataFrame(
    gdp = cumsum(0.5 .+ randn(150)),   # I(1) output
    cpi = cumsum(0.2 .+ randn(150)),   # I(1) price level
    ffr = 2.0 .+ 0.5 .* randn(150),    # policy rate
)
end_vars = [:gdp, :cpi, :ffr]

# Stage 3: reduced-form VAR(2)
est = estimate_var(df, end_vars, 2; include_constant = true, method = :ols)

# Stage 4: conjugate Normal-Wishart prior, then i.i.d. draws from the exact posterior
prior = build_prior(df, end_vars, est, :normal_wishart)
draws = sample_posterior(prior, est; ndraws = 500)

# Stage 5: recursive (Cholesky) identification
irf = identify_short_run(draws; horizon = 20)

(est.obs, est.lags, est.vars, length(draws.β), irf.horizon)
```

And the resulting impulse responses:

```@example quickstart
using Plots, Statistics

med(i, j) = [median(d[h + 1][i, j] for d in irf.H) for h in 0:irf.horizon]

plot(
    0:irf.horizon,
    [med(1, 1) med(2, 1) med(3, 1)];
    label = ["gdp" "cpi" "ffr"],
    xlabel = "horizon",
    ylabel = "response",
    title = "Response to shock 1 (posterior median)",
)
```

## Where to go next

- The **Guide** walks each stage in order. Start with
  [1-2. Pre-Estimation](@ref "Pre-Estimation: Unit Roots, Cointegration and Lag Selection")
  and read
  [The five-stage pipeline](@ref "The five-stage pipeline") for the objects that cross stage
  boundaries and the contracts between them — that section is the one most worth reading
  before you write any code.
- The **API Reference** is generated from the docstrings, split by stage, with exported names
  first and internal helpers below.
- The **Bibliography** collects the papers the docstrings cite.
