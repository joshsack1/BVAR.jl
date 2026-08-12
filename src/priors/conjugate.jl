# Natural-conjugate Normal-Wishart prior (direct and dummy-observation
# routes), the independent Normal-Inverse-Wishart prior, and the reduced-form
# asymmetric conjugate prior.

"""
    normal_wishart_prior(
        Y::AbstractMatrix,
        lags::Int,
        names::Vector{Symbol},
        include_constant::Bool;
        dummy_components::Vector{Symbol} = Symbol[],
        λ = (λ1 = 0.2, λ3 = 1.0, λ4 = 1e5, λ_soc = 1.0, λ_dio = 1.0, λ_lr = 1.0),
        H = nothing,
        ν0 = size(Y, 2) + 2,
    )

Builds a natural-conjugate Normal-Wishart prior, ``\\beta \\mid \\Sigma \\sim
MN(\\beta_0, \\Omega_0, \\Sigma)``, ``\\Sigma \\sim IW(S_0, \\nu_0)``.

With `dummy_components = Symbol[]` (the default) this is the direct
Kadiyala & Karlsson (1997) Normal-Wishart-compatible Minnesota prior: a
shared diagonal ``\\Omega_0`` with entries ``(\\lambda_1/\\ell^{\\lambda_3})^2 /
\\hat\\sigma_j^2`` for lag ``\\ell`` of variable ``j`` (and ``\\lambda_4^2`` for the
constant, if present), prior mean ``\\beta_0`` a random walk on own first
lags, and ``S_0 = \\text{diag}(\\hat\\sigma_1^2, \\ldots, \\hat\\sigma_n^2)(\\nu_0-n-1)``
so that ``E[\\Sigma] = \\text{diag}(\\hat\\sigma_j^2)`` under the prior.

Passing one or more `dummy_components` (any subset of `:minnesota`,
`:sum_of_coefficients`, `:dummy_initial_obs`, `:long_run`, from
`dummy-observations.jl`) instead builds the prior from stacked pseudo/dummy
observations (Bańbura, Giannone & Reichlin 2010; Giannone, Lenza & Primiceri
2015/2019) via `dummy_gram`, which is the standard way several of these
beliefs get *combined* into one prior in practice. `:long_run` additionally
requires the keyword `H`. Returns a `NormalWishartPrior`.
"""
function normal_wishart_prior(
    Y::AbstractMatrix{T},
    lags::Int,
    names::Vector{Symbol},
    include_constant::Bool;
    dummy_components::Vector{Symbol} = Symbol[],
    λ = (λ1 = 0.2, λ3 = 1.0, λ4 = 1e5, λ_soc = 1.0, λ_dio = 1.0, λ_lr = 1.0),
    H = nothing,
    ν0 = size(Y, 2) + 2,
) where {T<:Real}
    n = size(Y, 2)
    k = n * lags + (include_constant ? 1 : 0)
    if isempty(dummy_components)
        σ_ar = sqrt.(ar_residual_variances(Y, lags))
        offset = include_constant ? 1 : 0
        β0 = zeros(T, k, n)
        for i in 1:n
            β0[offset + i, i] = one(T)
        end
        Ω0_diag = zeros(T, k)
        if include_constant
            Ω0_diag[1] = T(λ.λ4)^2
        end
        for ℓ in 1:lags, j in 1:n
            Ω0_diag[offset + (ℓ - 1) * n + j] = (λ.λ1 / ℓ^λ.λ3)^2 / σ_ar[j]^2
        end
        Ω0 = Matrix(Diagonal(Ω0_diag))
        S0 = Matrix(Diagonal(σ_ar .^ 2)) * (ν0 - n - 1)
        return NormalWishartPrior(β0, Ω0, S0, T(ν0), lags, n, names, include_constant)
    end
    blocks = Tuple{Matrix{T},Matrix{T}}[]
    if :minnesota in dummy_components
        push!(blocks, dummy_minnesota(Y, lags, include_constant; λ1 = λ.λ1, λ3 = λ.λ3))
    end
    if :sum_of_coefficients in dummy_components
        push!(blocks, dummy_sum_of_coefficients(Y, lags, include_constant; λ_soc = λ.λ_soc))
    end
    if :dummy_initial_obs in dummy_components
        push!(blocks, dummy_initial_observation(Y, lags, include_constant; λ_dio = λ.λ_dio))
    end
    if :long_run in dummy_components
        @assert !isnothing(H) "dummy_components including :long_run requires the keyword H (the long-run combination matrix)"
        push!(blocks, dummy_long_run(Y, lags, include_constant, H; λ_lr = λ.λ_lr))
    end
    Yd = reduce(vcat, first.(blocks))
    Xd = reduce(vcat, last.(blocks))
    moments = dummy_gram(Yd, Xd)
    return NormalWishartPrior(
        moments.β0,
        moments.Ω0,
        moments.S0,
        moments.ν0,
        lags,
        n,
        names,
        include_constant,
    )
end

"""
    independent_niw_prior(
        Y::AbstractMatrix,
        lags::Int,
        names::Vector{Symbol},
        include_constant::Bool;
        λ = (λ1 = 0.2, λ2 = 0.5, λ3 = 1.0, λ4 = 1e5),
        ν0 = size(Y, 2) + 2,
    )

Builds an independent Normal-Inverse-Wishart prior, ``\\text{vec}(\\beta) \\sim
N(\\text{vec}(\\beta_0), \\Omega_0)`` independent of ``\\Sigma \\sim IW(S_0, \\nu_0)``.
Because ``\\beta`` and ``\\Sigma`` are not linked through a shared Kronecker
structure (unlike `NormalWishartPrior`), ``\\Omega_0`` is free to vary by
equation the way the Minnesota prior's cross-equation tightness ``\\lambda_2``
intends — this reuses the exact per-equation moments of `minnesota_prior`
(flattened to a ``kn \\times kn`` diagonal), rather than the shared,
equation-symmetric ``\\Omega_0`` a natural-conjugate prior is forced to use.
The price is that there is no closed-form posterior; sampling this prior's
posterior needs a Gibbs sampler once the Turing regression stage (6) exists.
Returns an `IndependentNIWPrior`.
"""
function independent_niw_prior(
    Y::AbstractMatrix{T},
    lags::Int,
    names::Vector{Symbol},
    include_constant::Bool;
    λ = (λ1 = 0.2, λ2 = 0.5, λ3 = 1.0, λ4 = 1e5),
    ν0 = size(Y, 2) + 2,
) where {T<:Real}
    n = size(Y, 2)
    minn = minnesota_prior(Y, lags, names, include_constant; λ = λ)
    β0 = vec(minn.β0)
    Ω0 = Matrix(Diagonal(vec(minn.Ω0)))
    S0 = Matrix(Diagonal(minn.σ_ar .^ 2)) * (ν0 - n - 1)
    return IndependentNIWPrior(β0, Ω0, S0, T(ν0), lags, n, names)
end

"""
    asymmetric_conjugate_prior(
        Y::AbstractMatrix,
        lags::Int,
        names::Vector{Symbol},
        include_constant::Bool;
        λ = (λ1 = 0.2, λ2 = 0.5, λ3 = 1.0, λ4 = 1e5),
        κ0 = 3.0,
    )

Builds an asymmetric conjugate prior in the spirit of Chan (2022),
"Asymmetric Conjugate Priors for Large Bayesian VARs". The natural-conjugate
`NormalWishartPrior` is forced to use one shared ``\\Omega_0`` across every
equation; Chan shows that if the system is triangularized — equation ``i``'s
regressors are extended with the *contemporaneous* values of variables
``1, \\ldots, i-1`` — a closed-form per-equation Normal-Gamma posterior
survives even though each equation now gets its own prior mean ``b_{0,i}``
and diagonal scale ``\\Omega_{0,i}``, recovering the cross-equation asymmetry
(``\\lambda_2``) that a shared-``\\Omega_0`` prior cannot represent. This is a
reduced-form implementation of that idea — reusing `minnesota_prior`'s
per-equation own/cross-lag moments, each with an independent Gamma prior on
its own (triangularized) residual variance,
``d_{ii}^{-1} \\sim \\text{Gamma}(\\kappa_i, \\tau_i)`` with ``\\tau_i = \\kappa_i
\\hat\\sigma_i^2`` — rather than a literal reproduction of Chan's triangular
derivation. Returns an `AsymmetricConjugatePrior`.
"""
function asymmetric_conjugate_prior(
    Y::AbstractMatrix{T},
    lags::Int,
    names::Vector{Symbol},
    include_constant::Bool;
    λ = (λ1 = 0.2, λ2 = 0.5, λ3 = 1.0, λ4 = 1e5),
    κ0 = 3.0,
) where {T<:Real}
    n = size(Y, 2)
    minn = minnesota_prior(Y, lags, names, include_constant; λ = λ)
    β0 = [minn.β0[:, i] for i in 1:n]
    Ω0 = [Matrix(Diagonal(minn.Ω0[:, i])) for i in 1:n]
    κ = fill(T(κ0), n)
    τ = κ .* minn.σ_ar .^ 2
    return AsymmetricConjugatePrior(β0, Ω0, κ, τ, lags, n, names)
end
