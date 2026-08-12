# Shared machinery for the Bayesian VAR priors stage (stage 5): the prior
# type hierarchy plus the Gram-matrix and dummy-observation helpers every
# family builds on.

abstract type AbstractVARPrior{T<:Real} end

# Original (Litterman 1986) Minnesota prior: no prior on Σ, per-equation
# diagonal Ω0 so cross-equation shrinkage (λ2) can differ from the own-lag
# shrinkage.
struct MinnesotaPrior{T<:Real} <: AbstractVARPrior{T}
    β0::Matrix{T}
    Ω0::Matrix{T}
    σ_ar::Vector{T}
    λ::NamedTuple
    lags::Int
    vars::Int
    names::Vector{Symbol}
end

# Natural-conjugate Normal-Wishart prior shared by the direct Minnesota-
# compatible NW prior and the four composable dummy-observation families.
struct NormalWishartPrior{T<:Real} <: AbstractVARPrior{T}
    β0::Matrix{T}
    Ω0::Matrix{T}
    S0::Matrix{T}
    ν0::T
    lags::Int
    vars::Int
    names::Vector{Symbol}
    include_constant::Bool
end

# Independent Normal-Inverse-Wishart prior: β and Σ independent a priori, so
# Ω0 is a full k·n × k·n matrix rather than Kronecker-linked to Σ. No
# closed-form posterior; needs Gibbs sampling downstream.
struct IndependentNIWPrior{T<:Real} <: AbstractVARPrior{T}
    β0::Vector{T}
    Ω0::Matrix{T}
    S0::Matrix{T}
    ν0::T
    lags::Int
    vars::Int
    names::Vector{Symbol}
end

# Reduced-form implementation of the Chan (2022) asymmetric conjugate
# prior: an independent Normal-Gamma prior per equation, retaining a
# closed-form posterior despite equation-specific shrinkage.
struct AsymmetricConjugatePrior{T<:Real} <: AbstractVARPrior{T}
    β0::Vector{Vector{T}}
    Ω0::Vector{Matrix{T}}
    κ::Vector{T}
    τ::Vector{T}
    lags::Int
    vars::Int
    names::Vector{Symbol}
end

# Baumeister & Hamilton (2019, AER) reduced-form reference prior (their
# Appendix A): independent Normal-Gamma per equation, valid at A = I;
# `structural` is the extension point for the full A ≠ I prior once stage 7
# (structural identification) exists.
struct BaumeisterHamiltonPrior{T<:Real} <: AbstractVARPrior{T}
    m::Vector{Vector{T}}
    M::Vector{Matrix{T}}
    κ::Vector{T}
    τ::Vector{T}
    lags::Int
    vars::Int
    names::Vector{Symbol}
    structural::Bool
end

"""
    gram_blocks(est::VARestimate)

Recovers the cross-product blocks ``X'Y`` and ``Y'Y`` from a `VARestimate`
via the OLS identities

``X'Y = (X'X)\\hat\\beta, \\qquad Y'Y = T\\hat\\Sigma + \\hat\\beta'(X'X)\\hat\\beta,``

so that a conjugate prior/posterior can be built without re-passing the raw
data alongside `est`. Returns a named tuple `(XᵀX, XᵀY, YᵀY)`.
"""
function gram_blocks(est::VARestimate)
    @unpack β_hat, Σ, XᵀX, obs = est
    XᵀY = XᵀX * β_hat
    YᵀY = obs * Σ + β_hat' * XᵀX * β_hat
    return (XᵀX = XᵀX, XᵀY = XᵀY, YᵀY = YᵀY)
end

"""
    ar_residual_variances(Y::AbstractMatrix, lags::Int)

Fits a univariate AR(`lags`) to each column of `Y` (via `ols_var`, the same
reference OLS routine the frequentist estimator uses) and returns the vector
of residual variances ``\\hat\\sigma_j^2``. These, not the joint ``\\hat\\Sigma``
in a `VARestimate`, are the scale used by the Minnesota and
Baumeister-Hamilton priors.

Because those priors divide by ``\\hat\\sigma_j^2``, every entry must come back
strictly positive and finite; a constant, perfectly lag-collinear, or too-short
series trips an assertion here rather than silently seeding the prior with
`Inf`/`NaN`.
"""
function ar_residual_variances(Y::AbstractMatrix{T}, lags::Int) where {T<:Real}
    n = size(Y, 2)
    σ² = zeros(T, n)
    for j in 1:n
        fit = ols_var(reshape(Y[:, j], :, 1), lags, true)
        σ²[j] = fit.Σ[1, 1]
    end
    @assert all(x -> x > 0 && isfinite(x), σ²) "Non-positive or non-finite univariate AR residual variance for at least one variable: the series may be constant, perfectly collinear with its own lags, or too short relative to lags. Minnesota- and Baumeister-Hamilton-style priors require a strictly positive residual-variance scale for every variable."
    return σ²
end

"""
    ar_residual_covariance(Y::AbstractMatrix, lags::Int)

Fits a univariate AR(`lags`) to each column of `Y` (via `ols_var`, the same
reference OLS routine `ar_residual_variances` uses) and returns the
``n\\times n`` sample covariance matrix ``\\hat S`` of the resulting residual
series, ``\\hat S_{ij} = T_{\\text{eff}}^{-1}\\sum_t \\hat\\varepsilon_{it}
\\hat\\varepsilon_{jt}``. Generalizes `ar_residual_variances` (which returns
only ``\\hat S``'s diagonal) to the full matrix — needed as the scale
``\\hat S`` of Baumeister & Hamilton (2019, AER)'s structural prior,
``\\tau_i(A) = \\kappa_i\\,a_i'\\hat Sa_i`` (stage 7, `structural_prior`).
"""
function ar_residual_covariance(Y::AbstractMatrix{T}, lags::Int) where {T<:Real}
    n = size(Y, 2)
    T_eff = size(Y, 1) - lags
    E = Matrix{T}(undef, T_eff, n)
    for j in 1:n
        fit = ols_var(reshape(Y[:, j], :, 1), lags, true)
        E[:, j] = fit.ε
    end
    return (E' * E) / T_eff
end

"""
    dummy_gram(Yd::AbstractMatrix, Xd::AbstractMatrix)

Converts a block of dummy observations `(Yd, Xd)` into the moments of a
natural-conjugate Normal-Wishart prior,

``\\beta_0=(X_d'X_d)^{-1}X_d'Y_d,\\quad \\Omega_0=(X_d'X_d)^{-1},\\quad
S_0=(Y_d-X_d\\beta_0)'(Y_d-X_d\\beta_0),\\quad \\nu_0=T_d-k,``

following the "prior as dummy observations" device (Theil & Goldberger 1961;
Bańbura, Giannone & Reichlin 2010). Several dummy blocks are combined by
stacking their rows before calling this function once, since the Gram
matrices of stacked blocks are just the sum of the individual blocks' Gram
matrices. Returns a named tuple `(β0, Ω0, S0, ν0)`.
"""
function dummy_gram(Yd::AbstractMatrix{T}, Xd::AbstractMatrix{T}) where {T<:Real}
    Td, k = size(Xd)
    n = size(Yd, 2)
    XdᵀXd = Xd' * Xd
    @assert rank(XdᵀXd) == k "dummy_components produce a rank-deficient design: need at least $k linearly independent dummy rows across the combined blocks (for example, include :dummy_initial_obs alongside :minnesota when include_constant = true, so the constant term is identified)"
    β0 = XdᵀXd \ (Xd' * Yd)
    S0 = (Yd - Xd * β0)' * (Yd - Xd * β0)
    ν0 = Td - k
    @assert ν0 > n - 1 "Not enough dummy observations for a proper prior: increase the dummy sample size or reduce the number of variables"
    @assert isposdef(Symmetric(S0)) "The combined dummy observations produce a degenerate (singular) prior for Σ. This typically happens when the variables have (near) zero sample means, which makes the sum-of-coefficients/dummy-initial-observation rows uninformative — these two priors are designed for variables in meaningful levels (e.g. log GDP), not demeaned/zero-mean series. Consider using variables in levels, or adjusting the requested dummy_components."
    return (β0 = β0, Ω0 = inv(XdᵀXd), S0 = S0, ν0 = T(ν0))
end

"""
    normal_wishart_posterior(
        prior::NormalWishartPrior,
        XᵀX::AbstractMatrix,
        XᵀY::AbstractMatrix,
        YᵀY::AbstractMatrix,
        obs::Int,
    )

Closed-form natural-conjugate posterior of a `NormalWishartPrior` given the
Gram blocks of the data (from `gram_blocks`),

``\\bar\\Omega = (\\Omega_0^{-1}+X'X)^{-1}, \\quad
\\bar\\beta = \\bar\\Omega(\\Omega_0^{-1}\\beta_0+X'Y), \\quad
\\bar S = S_0+Y'Y+\\beta_0'\\Omega_0^{-1}\\beta_0-\\bar\\beta'(\\Omega_0^{-1}+X'X)\\bar\\beta, \\quad
\\bar\\nu = \\nu_0+T,``

so that ``\\Sigma\\mid Y \\sim IW(\\bar S,\\bar\\nu)`` and
``\\beta\\mid\\Sigma,Y \\sim MN(\\bar\\beta,\\bar\\Omega,\\Sigma)``. Returns a named
tuple `(β̄, Ω̄, S̄, ν̄)`. Shared by `log_marginal_likelihood` (stage 5, for
hyperparameter tuning) and `sample_posterior` (stage 6, for posterior draws).
"""
function normal_wishart_posterior(
    prior::NormalWishartPrior,
    XᵀX::AbstractMatrix,
    XᵀY::AbstractMatrix,
    YᵀY::AbstractMatrix,
    obs::Int,
)
    @unpack β0, Ω0, S0, ν0 = prior
    Ω0_inv = inv(Ω0)
    P = Ω0_inv + XᵀX
    Ω̄ = inv(P)
    β̄ = Ω̄ * (Ω0_inv * β0 + XᵀY)
    S̄ = S0 + YᵀY + β0' * Ω0_inv * β0 - β̄' * P * β̄
    ν̄ = ν0 + obs
    return (β̄ = β̄, Ω̄ = Ω̄, S̄ = S̄, ν̄ = ν̄)
end

"""
    equation_normal_gamma_posterior(
        m::AbstractVector,
        M::AbstractMatrix,
        κ::Real,
        τ::Real,
        XᵀX::AbstractMatrix,
        Xᵀy::AbstractVector,
        yᵀy::Real,
        obs::Int,
    )

Closed-form posterior of a single equation ``y=Xb+\\varepsilon``,
``\\varepsilon \\sim N(0,dI)``, ``b\\mid d \\sim N(m,dM)``, ``d^{-1}\\sim
\\text{Gamma}(\\kappa,\\tau)``:

``\\bar M=(M^{-1}+X'X)^{-1}, \\quad \\bar b=\\bar M(M^{-1}m+X'y), \\quad
\\bar\\kappa=\\kappa+T/2, \\quad \\bar\\tau=\\tau+\\frac12\\left(y'y+m'M^{-1}m-\\bar
b'(M^{-1}+X'X)\\bar b\\right),``

so that ``d^{-1}\\mid y \\sim \\text{Gamma}(\\bar\\kappa,\\bar\\tau)`` and
``b\\mid d,y \\sim N(\\bar b,d\\bar M)``. Returns a named tuple
`(b̄, M̄, κ̄, τ̄)`. Shared by `equation_log_marginal_likelihood` (stage 5) and
`sample_posterior` (stage 6) for `AsymmetricConjugatePrior` and
`BaumeisterHamiltonPrior`, both independent per-equation Normal-Gamma priors.
"""
function equation_normal_gamma_posterior(
    m::AbstractVector,
    M::AbstractMatrix,
    κ::Real,
    τ::Real,
    XᵀX::AbstractMatrix,
    Xᵀy::AbstractVector,
    yᵀy::Real,
    obs::Int,
)
    M_inv = inv(M)
    P = M_inv + XᵀX
    M̄ = inv(P)
    b̄ = M̄ * (M_inv * m + Xᵀy)
    κ̄ = κ + obs / 2
    τ̄ = τ + (yᵀy + m' * M_inv * m - b̄' * P * b̄) / 2
    return (b̄ = b̄, M̄ = M̄, κ̄ = κ̄, τ̄ = τ̄)
end

"""
    minnesota_posterior(prior::MinnesotaPrior, XᵀX::AbstractMatrix, XᵀY::AbstractMatrix)

Closed-form per-equation Normal-Normal posterior of a `MinnesotaPrior`, with
``\\Sigma`` held fixed at its prior value `Diagonal(σ_ar.^2)` (the Minnesota
prior places no prior on ``\\Sigma``, so there is nothing to update there —
unlike `normal_wishart_posterior`). For each equation ``i``,

``\\bar P_i = \\Omega_{0,i}^{-1} + X'X/\\hat\\sigma_i^2, \\qquad
\\bar\\beta_i = \\bar P_i^{-1}\\left(\\Omega_{0,i}^{-1}\\beta_{0,i} +
X'Y_{\\cdot,i}/\\hat\\sigma_i^2\\right),``

so that ``\\beta_{\\cdot,i}\\mid Y \\sim N(\\bar\\beta_i,\\bar P_i^{-1})``. Returns
a named tuple `(β̄::Matrix, P::Vector{<:AbstractMatrix})`, the posterior mean
as a `k×n` matrix and the per-equation posterior precision matrices. Used by
`sample_posterior` (stage 6); not used by `log_marginal_likelihood`, which
instead scores candidate hyperparameters via a Woodbury-identity shortcut
that avoids forming `P` at every step of `optimize_hyperparameters`' search.
"""
function minnesota_posterior(
    prior::MinnesotaPrior,
    XᵀX::AbstractMatrix,
    XᵀY::AbstractMatrix,
)
    @unpack β0, Ω0, σ_ar = prior
    k, n = size(β0)
    β̄ = similar(β0)
    P = Vector{Matrix{eltype(β0)}}(undef, n)
    for i in 1:n
        σᵢ² = σ_ar[i]^2
        Ω0ᵢ_inv = Diagonal(1 ./ Ω0[:, i])
        Pᵢ = Ω0ᵢ_inv + XᵀX ./ σᵢ²
        β̄[:, i] = Pᵢ \ (Ω0ᵢ_inv * β0[:, i] + XᵀY[:, i] ./ σᵢ²)
        P[i] = Pᵢ
    end
    return (β̄ = β̄, P = P)
end
