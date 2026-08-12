# Baumeister & Hamilton (2019, AER) reduced-form reference prior.

"""
    baumeister_hamilton_prior(
        Y::AbstractMatrix,
        lags::Int,
        names::Vector{Symbol},
        include_constant::Bool;
        λ0 = 0.5,
        λ1 = 1.0,
        λ3 = 100.0,
        κ0 = 2.0,
        random_walk = true,
    )

Builds the reduced-form slice of the Baumeister & Hamilton (2019, AER,
"Structural Interpretation of Vector Autoregressions with Incomplete
Identification") reference prior (their Appendix A), applicable when the
structural rotation matrix is the identity (``A = I``, i.e. before structural
identification — stage 7 — exists). Each equation ``i`` is independent a
priori:

``d_{ii}^{-1} \\sim \\text{Gamma}(\\kappa_i, \\tau_i), \\qquad \\tau_i = \\kappa_i
\\hat\\sigma_i^2, \\qquad b_i \\mid d_{ii} \\sim N(m_i, d_{ii} M),``

with the shared (equation-independent, per the paper's own assumption that
``M_i`` does not vary with ``i``) diagonal scale

``M_{\\ell j, \\ell j} = \\dfrac{\\lambda_0^2}{\\ell^{2\\lambda_1} \\hat\\sigma_j^2}``

for lag ``\\ell`` of variable ``j``, and constant-term variance
``\\lambda_0^2 \\lambda_3^2`` (a large ``\\lambda_3`` makes the constant
essentially unrestricted, per the paper). ``\\hat\\sigma_j^2`` is the
univariate AR(`lags`) residual variance of variable ``j``, playing the role
of the paper's ``\\hat s_{jj}`` at ``A = I``. Returns a `BaumeisterHamiltonPrior`
with `structural = false`; the full structural extension (``A \\neq I``,
using the equilibrium/elasticity priors in the paper's Table 1) is deferred
to stage 7 and is not implemented here.
"""
function baumeister_hamilton_prior(
    Y::AbstractMatrix{T},
    lags::Int,
    names::Vector{Symbol},
    include_constant::Bool;
    λ0 = 0.5,
    λ1 = 1.0,
    λ3 = 100.0,
    κ0 = 2.0,
    random_walk = true,
) where {T<:Real}
    n = size(Y, 2)
    σ_ar = sqrt.(ar_residual_variances(Y, lags))
    k = n * lags + (include_constant ? 1 : 0)
    offset = include_constant ? 1 : 0
    M_diag = zeros(T, k)
    if include_constant
        M_diag[1] = (λ0 * λ3)^2
    end
    for ℓ in 1:lags, j in 1:n
        M_diag[offset + (ℓ - 1) * n + j] = λ0^2 / (T(ℓ)^(2λ1) * σ_ar[j]^2)
    end
    M_shared = Matrix(Diagonal(M_diag))
    m = Vector{Vector{T}}(undef, n)
    for i in 1:n
        mi = zeros(T, k)
        random_walk && (mi[offset + i] = one(T))
        m[i] = mi
    end
    M = [copy(M_shared) for _ in 1:n]
    κ = fill(T(κ0), n)
    τ = κ .* σ_ar .^ 2
    return BaumeisterHamiltonPrior(m, M, κ, τ, lags, n, names, false)
end
