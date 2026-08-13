```@meta
CurrentModule = BayesianVectorAutoregressions
```

# Impulse Responses and Long-Run Multipliers

All three identification routes converge on one output type, `IRFdraws`, so everything on this
page applies regardless of how the shocks were identified.

```@setup irf
include("plot-theme.jl")
using BayesianVectorAutoregressions, DataFrames, LinearAlgebra, Random, Statistics
Random.seed!(123)
df = DataFrame(
    gdp = cumsum(0.5 .+ randn(150)),
    cpi = cumsum(0.2 .+ randn(150)),
    ffr = 2.0 .+ 0.5 .* randn(150),
)
end_vars = [:gdp, :cpi, :ffr]
est = estimate_var(df, end_vars, 2; include_constant = true, method = :ols)
prior_nw = build_prior(df, end_vars, est, :normal_wishart)
draws_nw = sample_posterior(prior_nw, est; ndraws = 500)
irf = identify_short_run(draws_nw; horizon = 20)
```

## The `IRFdraws` layout

```@example irf
(
    ndraws = length(irf.H),          # one entry per posterior draw
    horizons = length(irf.H[1]),     # horizon + 1, because s = 0 is included
    per_horizon = size(irf.H[1][1]), # vars × vars
    method = irf.method,
)
```

So `irf.H[d][s + 1][i, j]` is the response of variable `i` to shock `j` at horizon `s`, in draw
`d`. Horizon `0` lives at index `1` — the impact response is included, not dropped.

| Field | Meaning |
|:--|:--|
| `H` | `Vector` over draws of `Vector` over horizons of `vars × vars` matrices |
| `horizon` | Maximum horizon `s` |
| `lags`, `vars`, `names` | Carried through from the reduced form |
| `method` | Which identification produced these (`:short_run`, `:sign_restriction`, `:hamilton_structural`) |

Underneath, [`impulse_responses`](@ref) builds ``\Psi_0 = I``,
``\Psi_s = \sum_{\ell=1}^{\min(s,p)} \Phi_\ell \Psi_{s-\ell}`` from the companion form
([`companion_matrix`](@ref), [`nonorthogonalized_irf`](@ref)) and post-multiplies by the impact
matrix — a Cholesky factor, its rotation, or ``A^{-1}``, depending on the route.

## Posterior summaries

Reduce across draws for each `(i, j, s)`:

```@example irf
function irf_quantiles(irf, i, j; probs = (0.16, 0.5, 0.84))
    paths = [[d[s + 1][i, j] for d in irf.H] for s in 0:irf.horizon]
    return (
        lo = [quantile(p, probs[1]) for p in paths],
        md = [quantile(p, probs[2]) for p in paths],
        hi = [quantile(p, probs[3]) for p in paths],
    )
end

q = irf_quantiles(irf, 1, 1)
round.(q.md[1:5], digits = 4)
```

## Fan charts

A 68% credible band around the posterior median, for every variable-shock pair:

```@example irf
using Plots

panels = map(Iterators.product(1:irf.vars, 1:irf.vars)) do (i, j)
    q = irf_quantiles(irf, i, j)
    plot(
        0:irf.horizon, q.md;
        ribbon = (q.md .- q.lo, q.hi .- q.md),
        fillalpha = 0.25,
        label = false,
        title = "$(irf.names[i]) ← shock $(j)",
        titlefontsize = 9,
        xlabel = i == irf.vars ? "horizon" : "",
        ylabel = j == 1 ? "response" : "",
    )
end

plot(
    permutedims(panels)...;
    layout = (irf.vars, irf.vars),
    size = (900, 700),
    plot_title = "Posterior impulse responses (median, 68% band)",
    plot_titlefontsize = 12,
    legend = false,
)
```

Rows are responding variables, columns are shocks — the same orientation as the `sign_pattern`
matrix in [Structural Identification](@ref api-structural). The recursive ordering is visible in the top row:
`gdp` is ordered first, so shocks 2 and 3 have exactly zero impact effect on it.

## Long-run multipliers

[`long_run_multiplier`](@ref) computes ``\Xi = (A - \sum_{\ell} B_\ell)^{-1}``, the cumulative
effect of a unit structural shock. It takes the **raw matrices**, not an `IRFdraws`, so apply it
per draw. At the reduced form ``A = I``, giving the familiar
``(I - \Phi_1 - \cdots - \Phi_p)^{-1}``:

```@example irf
A = Matrix(1.0I, est.vars, est.vars)

Ξ = [
    long_run_multiplier(A, b, est.lags, est.include_constant)
    for b in draws_nw.β
]

round.(sum(Ξ) / length(Ξ), digits = 3)
```

These are large here by construction — two of the three series are random walks, so their
long-run multipliers are near-singular. On real data, an implausibly large ``\Xi`` is a useful
signal that the VAR is close to non-stationary and that a specification in levels may be the
wrong choice.

Because a restriction on ``\Xi`` is just a function of ``(A, B)``, it plugs into the structural
prior through the same `restrictions` mechanism as everything else — that is what
[`long_run_sign_restriction`](@ref) wraps.
