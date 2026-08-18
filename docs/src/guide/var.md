```@meta
CurrentModule = BayesianVectorAutoregressions
```

# Frequentist VAR Estimation

Stage 3. [`estimate_var`](@ref) fits the reduced-form VAR(p)

```math
Y_t = c + \Phi_1 Y_{t-1} + \cdots + \Phi_p Y_{t-p} + \varepsilon_t
```

and returns a `VARestimate`, the object every Bayesian stage builds on.

```@setup var
include("plot-theme.jl")
using BayesianVectorAutoregressions, DataFrames, Random
Random.seed!(123)
df = DataFrame(
    gdp = cumsum(0.5 .+ randn(150)),
    cpi = cumsum(0.2 .+ randn(150)),
    ffr = 2.0 .+ 0.5 .* randn(150),
)
end_vars = [:gdp, :cpi, :ffr]
```

```@example var
est = estimate_var(df, end_vars, 2; include_constant = true, method = :ols)
(obs = est.obs, lags = est.lags, vars = est.vars, β_size = size(est.β_hat))
```

## `:ols` versus `:fem`

Two estimators are available and they agree to numerical precision, because equation-by-equation
least squares on a common regressor matrix *is* stacked least squares:

```@example var
est_fem = estimate_var(df, end_vars, 2; include_constant = true, method = :fem)
maximum(abs.(est.β_hat .- est_fem.β_hat))
```

Use `method = :ols` (the default) unless you specifically want the
[FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl) machinery: it is
roughly thirty times faster, since it solves one stacked system instead of `vars` separate
regressions.

`estimate_var` validates its inputs with assertions rather than returning a degenerate fit, so
expect a thrown error — not a silent `NaN` — if `method` is not `:ols` or `:fem`, if a name in
`end_vec` is not a column of `df`, if any value is `missing` or non-finite, or if `lags` is not
strictly between `0` and the number of observations.

## [The `VARestimate` object](@id varestimate-object)

`VARestimate` bundles the coefficient matrix, the residual covariance, standard errors, and the
regressor Gram matrix together with the shape metadata (`obs`, `lags`, `vars`, `names`,
`include_constant`) that later stages need to interpret them. See [`VARestimate`](@ref) in the
[API Reference](@ref api-var-estimation) for the full field-by-field description.

## The coefficient row convention

This is the single most important layout fact in the package, and it is shared by
`VARestimate.β_hat`, `BVARdraws.β`, and the structural `B`:

> **Row order:** the constant first (only if `include_constant`), then lag 1 of *every*
> variable, then lag 2 of every variable, and so on. **Columns are equations**, in `names`
> order. So `k == include_constant + lags * vars`.

For our VAR(2) in three variables with a constant, `β_hat` is `7 × 3`:

```@example var
using DataFrames

rows = ["constant"; ["$(v) lag $(ℓ)" for ℓ in 1:est.lags for v in est.names]]
DataFrame(
    "term" => rows,
    (string(v) => round.(est.β_hat[:, j], digits = 4) for (j, v) in enumerate(est.names))...,
)
```

Note the ordering carefully: rows 2–4 are lag 1 of `gdp`, `cpi`, `ffr`; rows 5–7 are lag 2 of
`gdp`, `cpi`, `ffr`. It is **not** grouped by variable.

Rather than slicing those rows yourself, use [`lag_blocks`](@ref), which converts the stored
matrix into the ``\Phi_\ell`` matrices of the equation above and asserts that the row count is
consistent with `lags` and `include_constant`:

```@example var
Φ = BayesianVectorAutoregressions.lag_blocks(est.β_hat, est.lags, est.include_constant)
length(Φ), size(Φ[1])
```

[`companion_matrix`](@ref) then stacks those into the ``np \times np`` companion form whose
powers generate the impulse responses. Its eigenvalues are the usual stability check:

```@example var
F = BayesianVectorAutoregressions.companion_matrix(Φ)
using LinearAlgebra
round(maximum(abs.(eigvals(F))), digits = 4)
```

A spectral radius at or above 1 is expected here — `gdp` and `cpi` were built as random walks.
