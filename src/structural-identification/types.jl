# Shared types for the structural-identification stage (stage 7): the
# general Baumeister-Hamilton (2019, AER) prior on the structural rotation
# matrix A, its wrapper around a reduced-form BaumeisterHamiltonPrior, and
# the two results containers (structural draws, and the impulse-response
# functions computed from either identification path).

"""
    StructuralPrior{T<:Real}

General prior on the structural rotation matrix ``A`` of Baumeister &
Hamilton (2019, AER, "Structural Interpretation of Vector Autoregressions
with Incomplete Identification"), built by `structural_prior`. See that
function's docstring for the full mathematical description.

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
struct StructuralPrior{T<:Real}
    template::Matrix{T}
    free::BitMatrix
    component::Dict{Tuple{Int,Int},UnivariateDistribution}
    restrictions::Vector{<:Function}
    vars::Int
    names::Vector{Symbol}
end

"""
    HamiltonStructuralPrior{T<:Real}

Pairs a reduced-form `BaumeisterHamiltonPrior` (its per-equation ``m,M,\\kappa``
— the prior for ``B,D\\mid A``) with a `StructuralPrior` (the prior for
``A``) and the sample covariance ``\\hat S`` of univariate-AR residuals
(`ar_residual_covariance`) needed to evaluate ``\\tau_i(A)=\\kappa_i\\,a_i'
\\hat Sa_i``. Built by `hamilton_structural_prior`; consumed by
`sample_structural`. Kept as its own type — rather than a field on
`BaumeisterHamiltonPrior` — so stage 5's prior types stay untouched and
`sample_posterior`/`sample_structural` are cleanly separated by dispatch.
"""
struct HamiltonStructuralPrior{T<:Real}
    reduced_form::BaumeisterHamiltonPrior{T}
    A_prior::StructuralPrior{T}
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
