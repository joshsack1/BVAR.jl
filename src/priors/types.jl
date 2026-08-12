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
"""
function ar_residual_variances(Y::AbstractMatrix{T}, lags::Int) where {T<:Real}
    n = size(Y, 2)
    σ² = zeros(T, n)
    for j in 1:n
        fit = ols_var(reshape(Y[:, j], :, 1), lags, true)
        σ²[j] = fit.Σ[1, 1]
    end
    return σ²
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
