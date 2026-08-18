# Shared types for the structural-identification stage (stage 7): the
# general Baumeister-Hamilton (2019, AER) prior on the structural rotation
# matrix A, its wrapper around a reduced-form BaumeisterHamiltonPrior, and
# the two results containers (structural draws, and the impulse-response
# functions computed from either identification path).

"""
    AbstractStructuralPrior{T<:Real}

Supertype of the two priors on the structural rotation matrix ``A``:
`StructuralPrior` (independent priors on ``A``'s free entries) and
`ParametricStructuralPrior` (priors on a parameter vector ``\\theta`` plus a
map ``\\theta\\to A``). Defines the interface every structural sampler
dispatches on — `nparams`, `marginals`, `draw_theta`, `theta_to_A`,
`marginal_log_prior`, `extra_log_prior` — so that both parameterizations
share one sampler.
"""
abstract type AbstractStructuralPrior{T<:Real} end

"""
    StructuralPrior{T<:Real}

General prior on the structural rotation matrix ``A`` of Baumeister &
Hamilton (2019, AER, "Structural Interpretation of Vector Autoregressions
with Incomplete Identification"), built by `structural_prior`, factored as
independent priors on ``A``'s free entries. See that function's docstring for
the full mathematical description; see `ParametricStructuralPrior` for the
alternative parameterization by an economic parameter vector.

- `template::Matrix{T}` — value used at every fixed (non-free) entry of `A`
  (the zero-restriction/point-mass case; a `1` on the diagonal normalizes
  the corresponding equation).
- `free::BitMatrix` — `true` where `A[i,j]` is drawn from `component[(i,j)]`.
- `component::Dict{Tuple{Int,Int},UnivariateDistribution}` — one
  `Distributions.jl` object per free entry (a `Truncated(dist, lo, hi)`
  expresses a sign or bound restriction on that entry).
- `restrictions::Vector{<:Function}` — joint restrictions on `(A,B)`, each
  ``(A,B)\\to\\mathbb{R}`` returning an additive log-weight.
- `vars::Int`, `names::Vector{Symbol}`.
"""
struct StructuralPrior{T<:Real} <: AbstractStructuralPrior{T}
    template::Matrix{T}
    free::BitMatrix
    component::Dict{Tuple{Int,Int},UnivariateDistribution}
    restrictions::Vector{<:Function}
    vars::Int
    names::Vector{Symbol}
end

"""
    ParametricStructuralPrior{T<:Real}

Parametric prior on the structural rotation matrix ``A``: independent priors
on a parameter vector ``\\theta`` together with a user-supplied map
``\\theta\\to A``, built by `parametric_structural_prior`. This is Baumeister
& Hamilton (2019, AER)'s own setup — a handful of economic parameters
(elasticities, multipliers) entering ``A``'s entries nonlinearly — as opposed
to `StructuralPrior`'s independent entry-by-entry priors. See
`parametric_structural_prior`'s docstring for the full mathematical
description.

- `θ_prior::Vector{UnivariateDistribution}` — independent marginal priors on
  the elements of ``\\theta``, one per parameter (a `Truncated(dist, lo, hi)`
  expresses a sign or bound restriction on that parameter).
- `map::Function` — ``\\theta\\to A``, taking an `AbstractVector` of length
  `length(θ_prior)` to an ``n\\times n`` `AbstractMatrix`.
- `extra_logprior::Vector{<:Function}` — each ``(\\theta,A)\\to\\mathbb{R}``
  returning an additive log-density. This is where priors on *functions* of
  ``A`` go: Baumeister & Hamilton's asymmetric-``t`` prior on
  ``\\det(\\tilde A)`` and their Student-``t`` prior on an entry of
  ``\\tilde A^{-1}`` are written by hand as log-densities, since
  `Distributions.jl` ships no skew-``t``. Unlike the component/marginal
  priors — which are exactly what candidate ``\\theta``'s are drawn from, so
  they cancel out of the acceptance ratio — `extra_logprior` terms never
  cancel out of *any* sampler's acceptance ratio and are always evaluated.
- `restrictions::Vector{<:Function}` — joint restrictions on ``(A,B)``, each
  ``(A,B)\\to\\mathbb{R}`` returning an additive log-weight; the same
  contract as `StructuralPrior.restrictions`.
- `vars::Int`, `names::Vector{Symbol}`.
"""
struct ParametricStructuralPrior{T<:Real} <: AbstractStructuralPrior{T}
    θ_prior::Vector{UnivariateDistribution}
    map::Function
    extra_logprior::Vector{<:Function}
    restrictions::Vector{<:Function}
    vars::Int
    names::Vector{Symbol}
end

"""
    HamiltonStructuralPrior{T<:Real,P<:AbstractStructuralPrior{T}}

Pairs a reduced-form `BaumeisterHamiltonPrior` (its per-equation ``m,M,\\kappa``
— the prior for ``B,D\\mid A``) with any `AbstractStructuralPrior` (the prior
for ``A``: a `StructuralPrior` or a `ParametricStructuralPrior`, carried in
the second type parameter `P` so the sampler specializes on it) and the
sample covariance ``\\hat S`` of univariate-AR residuals
(`ar_residual_covariance`) needed to evaluate ``\\tau_i(A)=\\kappa_i\\,a_i'
\\hat Sa_i``. Built by `hamilton_structural_prior`; consumed by
`sample_structural`. Kept as its own type — rather than a field on
`BaumeisterHamiltonPrior` — so stage 5's prior types stay untouched and
`sample_posterior`/`sample_structural` are cleanly separated by dispatch.
"""
struct HamiltonStructuralPrior{T<:Real,P<:AbstractStructuralPrior{T}}
    reduced_form::BaumeisterHamiltonPrior{T}
    A_prior::P
    Ŝ::Matrix{T}
end

"""
    StructuralDraws{T<:Real}

Posterior draws of the structural triple ``(A,B,D)`` from `sample_structural`,
following Baumeister & Hamilton (2019, AER)'s model ``Ay_t = Bx_{t-1}+u_t``,
``u_t\\sim N(0,D)``, ``D`` diagonal.

- `A::Vector{Matrix{T}}` — one ``n\\times n`` draw of the structural rotation
  matrix per entry.
- `B::Vector{Matrix{T}}` — one ``k\\times n`` draw of the structural lagged
  coefficients per entry, in the same row layout as `VARestimate.β_hat`.
- `D::Vector{Vector{T}}` — one length-``n`` draw of ``D``'s diagonal per
  entry.
- `lags::Int`, `vars::Int`, `names::Vector{Symbol}`, `include_constant::Bool`
  — carried over from the `VARestimate`/prior the draws were sampled from.

Deliberately not `BVARdraws` (whose documented contract is a uniform
reduced-form ``(\\beta,\\Sigma)`` shape across the five named families):
adding ``A`` here would break that contract. Consumed by `impulse_response`.
"""
struct StructuralDraws{T<:Real}
    A::Vector{Matrix{T}}
    B::Vector{Matrix{T}}
    D::Vector{Vector{T}}
    lags::Int
    vars::Int
    names::Vector{Symbol}
    include_constant::Bool
end

"""
    IRFdraws{T<:Real}

Structural impulse response functions, one set per posterior draw, produced
by either identification path: the Baumeister-Hamilton framework
(`impulse_response`) or the traditional short-run/sign-restriction path
(`identify_short_run`, `identify_sign_restrictions`).

- `H::Vector{Vector{Matrix{T}}}` — `H[d][s+1]` is the ``n\\times n`` response
  at horizon `s` for draw `d`.
- `horizon::Int`, `lags::Int`, `vars::Int`, `names::Vector{Symbol}`.
- `method::Symbol` — one of `:hamilton_structural`, `:cholesky`,
  `:sign_restriction`.
"""
struct IRFdraws{T<:Real}
    H::Vector{Vector{Matrix{T}}}
    horizon::Int
    lags::Int
    vars::Int
    names::Vector{Symbol}
    method::Symbol
end
