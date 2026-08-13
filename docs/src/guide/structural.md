```@meta
CurrentModule = BVAR
```

# Structural Identification

Stage 5. Reduced-form draws pin down ``\Sigma`` but not the structural shocks; identification is
the extra assumption that separates them. `BVAR.jl` offers three routes, two of which act
directly on a `BVARdraws`, and one — the general Baumeister-Hamilton framework — which samples
its own structural posterior.

```@setup struct
include("plot-theme.jl")
using BVAR, DataFrames, Distributions, LinearAlgebra, Random
Random.seed!(123)
df = DataFrame(
    gdp = cumsum(0.5 .+ randn(150)),
    cpi = cumsum(0.2 .+ randn(150)),
    ffr = 2.0 .+ 0.5 .* randn(150),
)
end_vars = [:gdp, :cpi, :ffr]
est = estimate_var(df, end_vars, 2; include_constant = true, method = :ols)
Y = BVAR.get_endogenous(df, end_vars)
prior_nw = build_prior(df, end_vars, est, :normal_wishart)
draws_nw = sample_posterior(prior_nw, est; ndraws = 500)
```

## 1. Recursive (Cholesky) identification

The classic Sims (1980) ordering assumption: the impact matrix is the lower Cholesky factor of
``\Sigma``, so variable `i` does not respond contemporaneously to shocks `j > i`. The ordering
is the ordering of `end_vec`, which makes it an economic assumption disguised as an argument
order — worth stating explicitly in any write-up.

```@example struct
irf_short = identify_short_run(draws_nw; horizon = 20)
(method = irf_short.method, horizon = irf_short.horizon, ndraws = length(irf_short.H))
```

## 2. Sign restrictions

[`identify_sign_restrictions`](@ref) implements Uhlig (2005) / Rubio-Ramírez, Waggoner & Zha
(2010): for each reduced-form draw it rotates the Cholesky factor ``P`` by random orthogonal
matrices ``Q`` until the impact matrix ``PQ`` matches the requested sign pattern.

!!! warning "Two things about `sign_pattern` that are easy to get wrong"
    1. It is `vars × vars` with **rows = variables, columns = shocks**, and entries `+1`, `-1`
       or `0`, where `0` means *unrestricted*.
    2. The pattern is checked **only against the impact matrix at horizon 0** — not across
       horizons. If you are used to multi-horizon sign restrictions, this is narrower than you
       may expect.

```@example struct
# Column 1 = the first shock. Require it to raise gdp and cpi on impact,
# and leave its effect on ffr unrestricted.
sign_pattern = [
     1  0  0;   # gdp response to shock 1 > 0
     1  0  0;   # cpi response to shock 1 > 0
     0  0  0    # ffr unrestricted
]

irf_sign = identify_sign_restrictions(draws_nw, sign_pattern; horizon = 20)
(method = irf_sign.method, ndraws = length(irf_sign.H))
```

Rejection sampling can fail: if no rotation satisfying the pattern is found within
`max_attempts` (default `10_000`) for *any* draw, the function throws rather than quietly
returning fewer draws than you asked for. A pattern that is too restrictive therefore produces a
loud error, not a silently biased sample.

## 3. The general Baumeister-Hamilton framework

Baumeister & Hamilton (2019) place a prior directly on the structural matrix ``A``, rather than
imposing exact restrictions. This subsumes both routes above: short-run zero restrictions are
the limit where entries of ``A`` are fixed, and sign restrictions become truncated component
priors. It also admits restrictions the other two cannot express, notably long-run ones.

The prior factors as

```math
p(A) \propto \prod_{(i,j)\ \text{free}} p_{ij}(A_{ij}) \cdot \prod_r \exp\big(r(A,B)\big),
```

built by [`structural_prior`](@ref) from three pieces plus optional joint restrictions:

- `template` — the fixed entries of ``A``, read off wherever `free` is `false`. Never sampled.
- `free` — a `Bool` matrix marking the entries that *are* sampled.
- `component` — one `UnivariateDistribution` per free entry. Must have exactly one key per
  free entry; a missing or extra key is an assertion error. A `Truncated` distribution is how
  you impose a sign or bound on a single entry.
- `restrictions` — a `Vector{Function}`, each with signature `r(A, B) -> Real`, returning an
  additive log-weight: `0.0` if satisfied, `-Inf` if violated, or a genuine log-density for a
  soft restriction.

```@example struct
n = length(end_vars)
template = Matrix(1.0I, n, n)
free = falses(n, n)
free[2, 1] = true        # let cpi respond contemporaneously to the gdp shock

# A bound restriction on that one free entry.
component = Dict((2, 1) => Truncated(Normal(0.0, 1.0), -2.0, 0.0))
nothing # hide
```

Two restriction closures ship with the package. [`det_sign_restriction`](@ref) fixes the sign of
``\det A``; [`long_run_sign_restriction`](@ref) constrains an entry of the long-run multiplier
matrix ``\Xi = (A - \sum_\ell B_\ell)^{-1}`` — the structural analogue of a Blanchard & Quah
(1989) restriction, expressed through the very same mechanism as short-run and sign
restrictions.

```@example struct
lr = long_run_sign_restriction(1, 1, 1, est.lags, est.include_constant)

A_prior = structural_prior(
    template, free, component;
    restrictions = Function[lr],
    names = end_vars,
)

rf_prior = build_prior(df, end_vars, est, :hamilton_baumeister)
struct_prior = hamilton_structural_prior(rf_prior, A_prior, Y)
typeof(struct_prior)
```

### Sampling, and reading the diagnostics

[`sample_structural`](@ref) draws the triple ``(A, B, D)``. Candidate ``A``'s come from the
component priors, so that factor cancels from both the importance weight and the
Metropolis-Hastings ratio, leaving only the likelihood ratio and any joint restrictions;
``B, D \mid A`` then come from the exact conjugate posterior.

The second return value is a `NamedTuple` of diagnostics, and **it is different for each
method** — always look at it:

```@example struct
s_mh, diag_mh = sample_structural(struct_prior, est; ndraws = 500, method = :mh)
diag_mh
```

```@example struct
s_sir, diag_sir = sample_structural(struct_prior, est; ndraws = 500, method = :sir)
diag_sir
```

`:mh` (the default) reports `acceptance_rate`; `:sir` reports `ess`, the effective sample size.
`:mh` is the more robust choice: both cost about the same per candidate, but `:sir`'s failure
mode is importance-weight collapse, which is easy to miss unless you check `ess` — and it is a
real risk here, because the marginal posterior of ``A`` carries a steep ``T/2`` exponent. A low
MH acceptance rate (the run above is in the low single-digit percent) means the chain is moving
slowly and you should raise `ndraws` accordingly.

Both methods return a `StructuralDraws`, which [`impulse_response`](@ref) turns into an
`IRFdraws`:

```@example struct
irf_struct = impulse_response(s_mh; horizon = 20)
(method = irf_struct.method, ndraws = length(irf_struct.H))
```

Continue to [Impulse Responses](@ref "Impulse Responses and Long-Run Multipliers") for reading
and plotting the result.
