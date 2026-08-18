```@meta
CurrentModule = BayesianVectorAutoregressions
```

# Structural Identification

Stage 5. Reduced-form draws pin down ``\Sigma`` but not the structural shocks; identification is
the extra assumption that separates them. `BayesianVectorAutoregressions.jl` offers three routes, two of which act
directly on a `BVARdraws`, and one — the general Baumeister-Hamilton framework — which samples
its own structural posterior.

```@setup struct
include("plot-theme.jl")
using BayesianVectorAutoregressions, DataFrames, Distributions, LinearAlgebra, Random
Random.seed!(123)
df = DataFrame(
    gdp = cumsum(0.5 .+ randn(150)),
    cpi = cumsum(0.2 .+ randn(150)),
    ffr = 2.0 .+ 0.5 .* randn(150),
)
end_vars = [:gdp, :cpi, :ffr]
est = estimate_var(df, end_vars, 2; include_constant = true, method = :ols)
Y = BayesianVectorAutoregressions.get_endogenous(df, end_vars)
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
  free entry; a missing or extra key is an assertion error. A `truncated` distribution is how
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
component = Dict((2, 1) => truncated(Normal(0.0, 1.0), -2.0, 0.0))
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

### Parametric priors on economic parameters

`structural_prior` puts an independent prior on each free entry of ``A``. Baumeister & Hamilton
(2019)'s own examples usually don't: they elicit priors on interpretable economic quantities —
elasticities, multipliers — that then enter several entries of ``A`` at once, often nonlinearly.
[`parametric_structural_prior`](@ref) supports exactly that: independent marginal priors on a
parameter vector ``\theta``, a user-supplied map ``\theta \to A``, and any number of
`extra_logprior` terms for priors on *functions* of ``(\theta, A)`` that are not simply one
``\theta_k``'s marginal — the paper's asymmetric-``t`` prior on ``\det(\tilde A)`` is the
canonical example.

```@example struct
θ_prior = UnivariateDistribution[
    truncated(TDist(3), 0.0, Inf),    # supply elasticity, sign-restricted
    truncated(TDist(3), -Inf, 0.0),   # demand elasticity, sign-restricted
]
A_map(θ) = [1.0 -θ[1]; 1.0 -θ[2]]     # θ enters column 2 of both equations

param_prior = parametric_structural_prior(
    θ_prior,
    A_map,
    2;
    extra_logprior = Function[(θ, A) -> logpdf(Normal(1.0, 2.0), det(A))],
    names = [:gdp, :cpi],
)

# A parametric prior's vars must match its reduced-form prior's, so this
# 2-variable supply/demand block gets its own reduced-form prior — built the
# same way as rf_prior above, from the same df.
sd_vars = [:gdp, :cpi]
sd_est = estimate_var(df, sd_vars, 2; include_constant = true, method = :ols)
sd_Y = BayesianVectorAutoregressions.get_endogenous(df, sd_vars)
rf_prior_sd = build_prior(df, sd_vars, sd_est, :hamilton_baumeister)
struct_prior_param = hamilton_structural_prior(rf_prior_sd, param_prior, sd_Y)
typeof(struct_prior_param)
```

Unlike the ``\theta`` marginals — which candidate draws come from under the independence
proposal, so they cancel out of both `:sir`'s importance weight and `:mh`'s acceptance ratio —
`extra_logprior` terms never cancel out of *any* sampler's acceptance ratio and are always
evaluated in full.

### Sampling, and reading the diagnostics

[`sample_structural`](@ref) draws the triple ``(A, B, D)`` by one of three methods. Under `:mh`
(the default) and `:sir`, candidate ``A``'s come from the component/marginal priors, so that
factor cancels from both the Metropolis-Hastings ratio and the importance weight, leaving only
the likelihood ratio and any joint restrictions; ``B, D \mid A`` then come from the exact
conjugate posterior. `:rwmh` is different — see the next section. All three methods take
`sample_structural`'s keywords explicitly (`ndraws`, `rng`, `method`, `burn_in`, `oversample`,
`θ₀`, `proposal_scale`, `ξ`, `proposal_df`); a keyword a given method doesn't use is silently
ignored.

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

`:mh` reports `acceptance_rate`; `:sir` reports `ess`, the effective sample size. `:mh` is the
more robust choice: both cost about the same per candidate, but `:sir`'s failure mode is
importance-weight collapse, which is easy to miss unless you check `ess` — and it is a real risk
here, because the marginal posterior of ``A`` carries a steep ``T/2`` exponent. A low MH
acceptance rate (the run above is in the low single-digit percent) means the chain is moving
slowly; raising `ndraws` helps, but when the free-parameter count grows the whole-``A``-at-once
proposal used by `:mh` and `:sir` degrades further still. `method = :rwmh`, next, is the
paper's own remedy.

### Random-walk Metropolis-Hastings

`method = :rwmh` is Baumeister & Hamilton (2019)'s own sampler: a random walk directly on the
collapsed marginal posterior of ``\theta`` (`structural_log_posterior`, ``B, D`` integrated out),
so ``p(A)`` no longer cancels and enters the acceptance ratio explicitly. The step is a
multivariate Student-``t`` with `proposal_df` degrees of freedom (2, the paper's own choice, by
default) rather than a Gaussian, so occasional long jumps keep the chain from getting stuck under
fat-tailed marginals.

By default the proposal is tuned the way the paper tunes it: BFGS (with `ForwardDiff` gradients
through `structural_log_posterior`) finds the mode of the marginal posterior starting from
`θ₀` (default: the marginals' medians), and the proposal scale is ``\xi \cdot`` the inverse
Hessian there. `ξ` is yours to adjust — follow the paper's own advice and tune it until
`acceptance_rate` lands near 30-35%. Supplying `proposal_scale` skips the BFGS/Hessian step
entirely, using `θ₀` as the chain's starting point exactly as given; supplying `θ₀` alone only
seeds that BFGS search — the mode it finds still becomes the starting point, overwriting `θ₀`.

```@example struct
s_rwmh, diag_rwmh = sample_structural(
    struct_prior_param, sd_est;
    ndraws = 1500,
    burn_in = 500,
    method = :rwmh,
    ξ = 1.0,
)
diag_rwmh.acceptance_rate
```

Beyond `acceptance_rate`, `:rwmh`'s diagnostics carry `θ`, the post-burn-in `ndraws × c` matrix
of parameter draws (`c = length(θ_prior)`) — useful for trace plots or for checking a marginal
against its prior.

Tuning by hand — maximizing `structural_log_posterior`'s closure with any optimizer, then
passing the result in as `θ₀`/`proposal_scale` — skips the package's own BFGS step entirely:

```julia
using Optim, ForwardDiff

logpost = structural_log_posterior(struct_prior_param, sd_est)
neg(θ) = -logpost(θ)
neg_grad!(G, θ) = copyto!(G, ForwardDiff.gradient(neg, θ))
result = Optim.optimize(neg, neg_grad!, θ_start, Optim.BFGS())

θ0 = Optim.minimizer(result)
H = ForwardDiff.hessian(neg, θ0)
s_hand, diag_hand = sample_structural(
    struct_prior_param, sd_est;
    method = :rwmh, θ₀ = θ0, proposal_scale = inv(H),
)
```

All three methods return a `StructuralDraws`, which [`impulse_response`](@ref) turns into an
`IRFdraws`:

```@example struct
irf_struct = impulse_response(s_mh; horizon = 20)
(method = irf_struct.method, ndraws = length(irf_struct.H))
```

Continue to [Impulse Responses](@ref "Impulse Responses and Long-Run Multipliers") for reading
and plotting the result.
