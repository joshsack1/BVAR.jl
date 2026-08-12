# Original (Litterman 1986) Minnesota prior.

"""
    minnesota_prior(
        Y::AbstractMatrix,
        lags::Int,
        names::Vector{Symbol},
        include_constant::Bool;
        λ = (λ1 = 0.2, λ2 = 0.5, λ3 = 1.0, λ4 = 1e5),
        random_walk = true,
    )

Builds the original (Litterman 1986) Minnesota prior. Each equation is
treated independently and the residual covariance is never assigned a prior
— it is plugged in at its univariate AR(`lags`) estimate ``\\hat\\sigma_j^2``
(see `ar_residual_variances`), unlike the natural-conjugate priors in
`conjugate.jl`. With `random_walk = true` (the default) the prior mean of
``\\beta`` sets the coefficient on own first lag to one and everything else to
zero; the prior variance of the coefficient on lag ``\\ell`` of variable ``j``
in equation ``i`` is

``\\Omega_{0} = \\begin{cases}
\\left(\\dfrac{\\lambda_1}{\\ell^{\\lambda_3}}\\right)^2 & i = j \\\\[6pt]
\\left(\\dfrac{\\lambda_1 \\lambda_2}{\\ell^{\\lambda_3}}\\right)^2 \\dfrac{\\hat\\sigma_i^2}{\\hat\\sigma_j^2} & i \\neq j
\\end{cases}``

with the constant (if present) given the loose variance ``(\\hat\\sigma_i
\\lambda_4)^2``. Because ``\\Omega_0`` is allowed to vary by equation ``i`` (via
``\\lambda_2``), it cannot be represented by the single shared scale matrix a
natural-conjugate `NormalWishartPrior` uses — that asymmetry is the point of
the original Minnesota prior, and of `AsymmetricConjugatePrior`/
`IndependentNIWPrior` below, which reuse it. Returns a `MinnesotaPrior`.
"""
function minnesota_prior(
    Y::AbstractMatrix{T},
    lags::Int,
    names::Vector{Symbol},
    include_constant::Bool;
    λ = (λ1 = 0.2, λ2 = 0.5, λ3 = 1.0, λ4 = 1e5),
    random_walk = true,
) where {T<:Real}
    n = size(Y, 2)
    σ_ar = sqrt.(ar_residual_variances(Y, lags))
    k = n * lags + (include_constant ? 1 : 0)
    offset = include_constant ? 1 : 0
    β0 = zeros(T, k, n)
    if random_walk
        for i in 1:n
            β0[offset + i, i] = one(T)
        end
    end
    Ω0 = zeros(T, k, n)
    if include_constant
        for i in 1:n
            Ω0[1, i] = (σ_ar[i] * λ.λ4)^2
        end
    end
    for ℓ in 1:lags, j in 1:n
        row = offset + (ℓ - 1) * n + j
        for i in 1:n
            Ω0[row, i] = if i == j
                (λ.λ1 / ℓ^λ.λ3)^2
            else
                (λ.λ1 * λ.λ2 / ℓ^λ.λ3)^2 * (σ_ar[i]^2 / σ_ar[j]^2)
            end
        end
    end
    return MinnesotaPrior(β0, Ω0, σ_ar, λ, lags, n, names)
end
