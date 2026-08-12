# Traditional short-run/sign-restriction structural identification, usable
# by any of the five prior families' reduced-form BVARdraws — as opposed to
# the general Baumeister-Hamilton p(A) framework in hamilton-structural.jl,
# which is specific to a HamiltonStructuralPrior's own sampler.

"""
    identify_short_run(draws::BVARdraws; horizon::Int = 20)

Identifies structural shocks recursively (Cholesky/Sims 1980 short-run
identification): for each reduced-form draw, the impact matrix is the lower
Cholesky factor of `Σ`, ``P_d = \\text{chol}(\\Sigma_d)_L``, so variable 1 has
no contemporaneous response to shocks ``2,\\ldots,n``, variable 2 no response
to shocks ``3,\\ldots,n``, and so on (variable order = `draws.names`).
Computes `impulse_responses` for every draw. Returns an `IRFdraws`.
"""
function identify_short_run(draws::BVARdraws{T}; horizon::Int = 20) where {T<:Real}
    ndraws = length(draws.β)
    @assert length(draws.β) == length(draws.Σ) "draws.β and draws.Σ must have the same length"
    H = Vector{Vector{Matrix{T}}}(undef, ndraws)
    for d in 1:ndraws
        Φ = lag_blocks(draws.β[d], draws.lags, draws.include_constant)
        P = Matrix{T}(cholesky(Symmetric(draws.Σ[d])).L)
        H[d] = impulse_responses(Φ, P, horizon)
    end
    return IRFdraws(H, horizon, draws.lags, draws.vars, draws.names, :cholesky)
end

"""
    random_orthogonal(n::Int, rng::Random.AbstractRNG, ::Type{T})

Draws a random ``n\\times n`` orthogonal matrix ``Q`` via a sign-normalized
QR decomposition of a standard-normal ``n\\times n`` matrix (Rubio-Ramírez,
Waggoner & Zha 2010), uniform over the orthogonal group. Internal; used by
`identify_sign_restrictions`.
"""
function random_orthogonal(n::Int, rng::Random.AbstractRNG, ::Type{T}) where {T<:Real}
    F = qr(randn(rng, T, n, n))
    Q = Matrix{T}(F.Q)
    for j in 1:n
        F.R[j, j] < 0 && (Q[:, j] .*= -1)
    end
    return Q
end

"""
    matches_sign_pattern(impact::AbstractMatrix, sign_pattern::AbstractMatrix{<:Integer})

Checks whether every nonzero entry of `sign_pattern` (``\\pm1``; `0` means
unrestricted) matches the sign of the corresponding entry of `impact`.
Internal; used by `identify_sign_restrictions`.
"""
function matches_sign_pattern(
    impact::AbstractMatrix,
    sign_pattern::AbstractMatrix{<:Integer},
)
    n = size(impact, 1)
    for i in 1:n, j in 1:n
        s = sign_pattern[i, j]
        s == 0 && continue
        sign(impact[i, j]) == s || return false
    end
    return true
end

"""
    identify_sign_restrictions(
        draws::BVARdraws,
        sign_pattern::AbstractMatrix{<:Integer};
        horizon::Int = 20,
        max_attempts::Int = 10_000,
        rng::Random.AbstractRNG = Random.default_rng(),
    )

Identifies structural shocks via sign restrictions (Uhlig 2005;
Rubio-Ramírez, Waggoner & Zha 2010): for each reduced-form draw, repeatedly
rotates the Cholesky factor `P` by a random orthogonal matrix ``Q``
(`random_orthogonal`) until the impact responses ``PQ`` match
`sign_pattern` (entries ``-1/0/1``, ``0`` meaning unrestricted;
`matches_sign_pattern`), then keeps that rotation. Fails loudly via
`@assert` if `max_attempts` rotations are exhausted for any draw, rather
than silently returning fewer draws than requested. Returns an `IRFdraws`.
"""
function identify_sign_restrictions(
    draws::BVARdraws{T},
    sign_pattern::AbstractMatrix{<:Integer};
    horizon::Int = 20,
    max_attempts::Int = 10_000,
    rng::Random.AbstractRNG = Random.default_rng(),
) where {T<:Real}
    n = draws.vars
    @assert size(sign_pattern) == (n, n) "sign_pattern must be n×n"
    @assert all(s -> s in (-1, 0, 1), sign_pattern) "sign_pattern entries must be -1, 0, or 1 (got a value outside that range)"
    ndraws = length(draws.β)
    @assert length(draws.β) == length(draws.Σ) "draws.β and draws.Σ must have the same length"
    H = Vector{Vector{Matrix{T}}}(undef, ndraws)
    for d in 1:ndraws
        Φ = lag_blocks(draws.β[d], draws.lags, draws.include_constant)
        P = Matrix{T}(cholesky(Symmetric(draws.Σ[d])).L)
        matched = false
        attempts = 0
        while !matched && attempts < max_attempts
            attempts += 1
            impact = P * random_orthogonal(n, rng, T)
            if matches_sign_pattern(impact, sign_pattern)
                H[d] = impulse_responses(Φ, impact, horizon)
                matched = true
            end
        end
        @assert matched "identify_sign_restrictions: no rotation satisfying sign_pattern found within max_attempts = $max_attempts for draw $d"
    end
    return IRFdraws(H, horizon, draws.lags, draws.vars, draws.names, :sign_restriction)
end
