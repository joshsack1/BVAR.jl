```@meta
CurrentModule = BVAR
```

# Posterior Sampling: Closed Form and Gibbs

Stage 4b. [`sample_posterior`](@ref) takes a prior and a `VARestimate` and returns a
`BVARdraws`. It has six methods, and **which sampling mechanism you get depends entirely on the
prior type you hand it** — the call site looks identical either way.

```@setup bayes
include("plot-theme.jl")
using BVAR, DataFrames, Random
Random.seed!(123)
df = DataFrame(
    gdp = cumsum(0.5 .+ randn(150)),
    cpi = cumsum(0.2 .+ randn(150)),
    ffr = 2.0 .+ 0.5 .* randn(150),
)
end_vars = [:gdp, :cpi, :ffr]
est = estimate_var(df, end_vars, 2; include_constant = true, method = :ols)
```

| Prior type | Mechanism | What `ndraws` counts |
|:--|:--|:--|
| `MinnesotaPrior` | Closed form, ``\Sigma`` plugged in | Independent draws |
| `NormalWishartPrior` | Closed form (Matrix-Normal / Inverse-Wishart) | Independent draws |
| `AsymmetricConjugatePrior` | Closed form, per equation (Normal-Gamma) | Independent draws |
| `BaumeisterHamiltonPrior` | Closed form, per equation (Normal-Gamma) | Independent draws |
| `IndependentNIWPrior` | **Gibbs sampler** (Turing) | Correlated sweeps |

That distinction matters for how you treat the output: for the four conjugate families the
draws are i.i.d. from the *exact* posterior, so there is no burn-in, no thinning and no
convergence question. For `IndependentNIWPrior` they are an autocorrelated Markov chain.

## The closed-form path

```@example bayes
prior_nw = build_prior(df, end_vars, est, :normal_wishart)
draws_nw = sample_posterior(prior_nw, est; ndraws = 500)
(family = draws_nw.family, ndraws = length(draws_nw.β), β_size = size(draws_nw.β[1]))
```

`BVARdraws` stores the draws as vectors of matrices rather than a single stacked array:

| Field | Type | Meaning |
|:--|:--|:--|
| `β` | `Vector{Matrix{T}}` | One `k × vars` coefficient matrix per draw, same row convention as `VARestimate.β_hat` |
| `Σ` | `Vector{Matrix{T}}` | One `vars × vars` covariance per draw |
| `family` | `Symbol` | Which prior family produced these draws |
| `lags`, `vars`, `names`, `include_constant` | | Carried forward for the structural stage |

Reduce over it with ordinary Julia:

```@example bayes
using Statistics

β_mean = sum(draws_nw.β) / length(draws_nw.β)
round.(β_mean, digits = 4)
```

Comparing that against the OLS point estimate shows the shrinkage the prior applied:

```@example bayes
round.(β_mean .- est.β_hat, digits = 4)
```

## The Gibbs path

`IndependentNIWPrior` has no closed-form joint posterior, so `sample_posterior` runs a Gibbs
sampler. Its two conditionals ([`niw_cond_β`](@ref) and [`niw_cond_Σ`](@ref)) are exactly the
same Matrix-Normal and Inverse-Wishart distributions the conjugate families draw from directly
— only *alternating* between them requires a sampler at all. Marginal-likelihood tuning is
unavailable for this family, so hyperparameters must be fixed:

```@example bayes
prior_niw = build_prior(
    df, end_vars, est, :independent_niw;
    hyperparameter_method = :fixed,
    hyperparameters = (λ1 = 0.2, λ2 = 0.5, λ3 = 1.0, λ4 = 1e5),
)
draws_niw = sample_posterior(prior_niw, est; ndraws = 200)
(family = draws_niw.family, ndraws = length(draws_niw.β))
```

The likelihood is expressed purely through the Gram blocks ``(X'X, X'Y, Y'Y)`` rather than the
raw data ([`niw_var_model`](@ref)), which is what keeps this path consistent with the other
five methods and avoids re-collapsing the data on every sweep.

Because these draws are correlated, inspect the trace before trusting a posterior summary:

```@example bayes
using Plots

trace_own_lag = [b[2, 1] for b in draws_niw.β]     # gdp equation, own first lag
plot(
    trace_own_lag;
    label = "β[gdp lag 1, gdp]",
    xlabel = "Gibbs sweep",
    ylabel = "draw",
    title = "Gibbs trace",
)
```

## Comparing the two

The posterior for a single coefficient under both priors, on the same data:

```@example bayes
own_nw = [b[2, 1] for b in draws_nw.β]     # 500 i.i.d. draws
own_niw = [b[2, 1] for b in draws_niw.β]   # 200 Gibbs sweeps

# Separate calls, because the two sets deliberately differ in length.
histogram(
    own_nw;
    label = "Normal-Wishart (i.i.d.)",
    xlabel = "β[gdp lag 1, gdp]",
    ylabel = "density",
    title = "Posterior of the own first-lag coefficient",
    alpha = 0.55,
    bins = 30,
    normalize = :pdf,
)
histogram!(own_niw; label = "Independent NIW (Gibbs)", alpha = 0.55, bins = 30, normalize = :pdf)
vline!([est.β_hat[2, 1]]; label = "OLS", linestyle = :dash, linewidth = 2)
```

The two need not agree: `IndependentNIWPrior` can represent cross-equation asymmetry that the
Kronecker structure of `NormalWishartPrior` cannot, which is the whole reason to pay for the
Gibbs sampler.
