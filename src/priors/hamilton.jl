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
        m = nothing,
        η = nothing,
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
of the paper's ``\\hat s_{jj}`` at ``A = I``.

The prior mean ``m_i`` defaults to the random-walk mean (`random_walk = true`:
unity on equation ``i``'s own first lag, zero elsewhere) and can be replaced
in either of two ways, which are mutually exclusive and both require
`random_walk = false` — `random_walk` and a custom mean both specify the prior
mean of ``b_i``, so fold the own-first-lag entries into your own `m`/`η` if you
want both:

- `m`, a length-`n` vector of length-`k` vectors: one constant prior mean per
  equation, in the regressor ordering `[constant (if any); lag 1 of every
  variable; lag 2; …]`. This is the paper's baseline (their `main_BH_AER.m`),
  where every entry is zero except ``\\pm0.1`` on the first lag of the real
  oil price in the supply and demand equations.
- `η`, an ``n\\times k`` matrix: the prior mean becomes *A-dependent*,
  ``m_i(A) = \\eta'a_i`` with ``a_i' = A[i,:]``, the paper's construction for
  its KM12/KAER replications. Only `sample_structural`/
  `structural_log_posterior` (stage 7) read the ``A``-dependence; the
  ``A = I`` consumers read the stored ``m_i = \\eta'e_i`` (row ``i`` of `η`),
  which this builder materializes for them. The structural random-walk prior
  is `η` with row ``i`` the unit vector on equation ``i``'s own first lag —
  `η[i, offset + i] = 1`, where `offset = include_constant ? 1 : 0`. Note the
  ordering caveat: the paper's MATLAB code puts the constant *last*, so its
  canonical ``\\eta = [I\\ 0]`` becomes ``[0\\ I\\ 0]`` here.

Returns a `BaumeisterHamiltonPrior`, with `structural = true` exactly when `η`
is supplied. Via `build_prior` both keywords are passed inside the
`hyperparameters` NamedTuple, which requires
`hyperparameter_method = :fixed`.
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
    m::Union{Nothing,Vector{<:AbstractVector{<:Real}}} = nothing,
    η::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
) where {T<:Real}
    n = size(Y, 2)
    σ_ar = sqrt.(ar_residual_variances(Y, lags))
    k = n * lags + (include_constant ? 1 : 0)
    offset = include_constant ? 1 : 0
    @assert m === nothing || η === nothing "m and η are mutually exclusive ways of setting the prior mean: η already fixes m through its A = I value m[i] = η'eᵢ (row i of η), so pass one or the other"
    @assert (m === nothing && η === nothing) || !random_walk "a custom prior mean requires random_walk = false: random_walk and m/η both specify the prior mean of bᵢ, so fold the own-first-lag entries into your own m/η if you want both (set m[i][$(offset) + i] = 1 or η[i, $(offset) + i] = 1)"
    if m !== nothing
        @assert length(m) == n "m must have one prior mean vector per equation (got $(length(m)), expected n = $n)"
        @assert all(mᵢ -> length(mᵢ) == k, m) "every entry of m must have one element per regressor, k = $k, ordered [constant (if any); lag 1 of every variable; lag 2; …]"
        @assert all(mᵢ -> all(isfinite, mᵢ), m) "every entry of m must be finite"
    end
    if η !== nothing
        @assert size(η) == (n, k) "η must be n×k = $n×$k, its columns ordered [constant (if any); lag 1 of every variable; lag 2; …] — note this package puts the constant FIRST where Baumeister & Hamilton's MATLAB puts it last, so their canonical η = [I 0] is written here as η[i, $(offset) + i] = 1"
        @assert all(isfinite, η) "every entry of η must be finite"
    end
    M_diag = zeros(T, k)
    if include_constant
        M_diag[1] = (λ0 * λ3)^2
    end
    for ℓ in 1:lags, j in 1:n
        M_diag[offset + (ℓ - 1) * n + j] = λ0^2 / (T(ℓ)^(2λ1) * σ_ar[j]^2)
    end
    M_shared = Matrix(Diagonal(M_diag))
    # η's stored mean is its A = I evaluation, η'eᵢ = row i of η; the
    # A-dependent mᵢ(A) = η'aᵢ is formed per candidate A in stage 7. Both custom
    # paths copy, so a later mutation of the caller's arrays cannot reach here.
    η_built = η === nothing ? nothing : Matrix{T}(η)
    m_built = Vector{Vector{T}}(undef, n)
    if η_built !== nothing
        for i in 1:n
            m_built[i] = Vector{T}(η_built[i, :])
        end
    elseif m !== nothing
        for i in 1:n
            m_built[i] = Vector{T}(m[i])
        end
    else
        for i in 1:n
            mi = zeros(T, k)
            random_walk && (mi[offset + i] = one(T))
            m_built[i] = mi
        end
    end
    M = [copy(M_shared) for _ in 1:n]
    κ = fill(T(κ0), n)
    τ = κ .* σ_ar .^ 2
    return BaumeisterHamiltonPrior(m_built, M, κ, τ, lags, n, names, η_built, η !== nothing)
end
