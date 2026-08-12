# Direct (non-MCMC) posterior sampling for the four closed-form prior
# families: NormalWishartPrior, MinnesotaPrior, AsymmetricConjugatePrior, and
# BaumeisterHamiltonPrior. Each draws i.i.d. straight from the family's exact
# posterior, computed by the `*_posterior` helpers in `src/priors/types.jl`.

"""
    sample_posterior(
        prior::NormalWishartPrior,
        gram::NamedTuple,
        obs::Int,
        include_constant::Bool;
        ndraws::Int = 1000,
        rng::Random.AbstractRNG = Random.default_rng(),
    )

Draws `ndraws` i.i.d. samples from the natural-conjugate posterior
``\\Sigma\\mid Y \\sim IW(\\bar S,\\bar\\nu)``, ``\\beta\\mid\\Sigma,Y \\sim
MN(\\bar\\beta,\\bar\\Omega,\\Sigma)`` (`normal_wishart_posterior`). No MCMC is
needed — the posterior is exact. Internal; called by the public
`sample_posterior(prior, est; ...)` entry point.
"""
function sample_posterior(
    prior::NormalWishartPrior,
    gram::NamedTuple,
    obs::Int,
    include_constant::Bool;
    ndraws::Int = 1000,
    rng::Random.AbstractRNG = Random.default_rng(),
    kwargs...,
)
    T = eltype(prior.β0)
    post = normal_wishart_posterior(prior, gram.XᵀX, gram.XᵀY, gram.YᵀY, obs)
    Ω̄ = Symmetric(post.Ω̄)
    S̄ = Matrix(Symmetric(post.S̄))
    β = Vector{Matrix{T}}(undef, ndraws)
    Σ = Vector{Matrix{T}}(undef, ndraws)
    for d in 1:ndraws
        Σd = Matrix{T}(rand(rng, InverseWishart(post.ν̄, S̄)))
        β[d] = Matrix{T}(rand(rng, MatrixNormal(post.β̄, Ω̄, Symmetric(Σd))))
        Σ[d] = Σd
    end
    return BVARdraws(
        β,
        Σ,
        :normal_wishart,
        prior.lags,
        prior.vars,
        prior.names,
        include_constant,
    )
end

"""
    sample_posterior(
        prior::MinnesotaPrior,
        gram::NamedTuple,
        obs::Int,
        include_constant::Bool;
        ndraws::Int = 1000,
        rng::Random.AbstractRNG = Random.default_rng(),
    )

Draws `ndraws` i.i.d. samples of ``\\beta`` from the per-equation Normal-Normal
posterior `minnesota_posterior` (``\\Sigma`` is never given a prior, so every
draw's ``\\Sigma`` is the same fixed `Diagonal(σ_ar.^2)`). No MCMC is needed.
Internal; called by the public `sample_posterior(prior, est; ...)` entry
point.
"""
function sample_posterior(
    prior::MinnesotaPrior,
    gram::NamedTuple,
    obs::Int,
    include_constant::Bool;
    ndraws::Int = 1000,
    rng::Random.AbstractRNG = Random.default_rng(),
    kwargs...,
)
    T = eltype(prior.β0)
    k, n = size(prior.β0)
    post = minnesota_posterior(prior, gram.XᵀX, gram.XᵀY)
    Ω̄ = [Symmetric(inv(post.P[i])) for i in 1:n]
    Σ_fixed = Matrix{T}(Diagonal(prior.σ_ar .^ 2))
    β = Vector{Matrix{T}}(undef, ndraws)
    Σ = Vector{Matrix{T}}(undef, ndraws)
    for d in 1:ndraws
        βd = Matrix{T}(undef, k, n)
        for i in 1:n
            βd[:, i] = rand(rng, MvNormal(post.β̄[:, i], Ω̄[i]))
        end
        β[d] = βd
        Σ[d] = copy(Σ_fixed)
    end
    return BVARdraws(
        β,
        Σ,
        :minnesota,
        prior.lags,
        prior.vars,
        prior.names,
        include_constant,
    )
end

"""
    equation_normal_gamma_draws(
        means::Vector{<:AbstractVector},
        scales::Vector{<:AbstractMatrix},
        κ::Vector,
        τ::Vector,
        gram::NamedTuple,
        obs::Int,
        ndraws::Int,
        rng::Random.AbstractRNG,
    )

Shared sampling routine for the two independent per-equation Normal-Gamma
families (`AsymmetricConjugatePrior`, `BaumeisterHamiltonPrior`): for each
equation `i`, draws ``d_{ii}^{-1}\\sim\\text{Gamma}(\\bar\\kappa_i,\\bar\\tau_i)``
then ``b_i\\mid d_{ii}\\sim N(\\bar b_i,d_{ii}\\bar M_i)`` from
`equation_normal_gamma_posterior`. Since neither family places a prior on
cross-equation covariance, every `Σ` draw is diagonal. Returns
`(β::Vector{Matrix}, Σ::Vector{Matrix})`. Internal; shared by the
`sample_posterior` methods for both families below.
"""
function equation_normal_gamma_draws(
    means::Vector{<:AbstractVector{T}},
    scales::Vector{<:AbstractMatrix{T}},
    κ::Vector{T},
    τ::Vector{T},
    gram::NamedTuple,
    obs::Int,
    ndraws::Int,
    rng::Random.AbstractRNG,
) where {T<:Real}
    n = length(means)
    k = length(means[1])
    posts = [
        equation_normal_gamma_posterior(
            means[i],
            scales[i],
            κ[i],
            τ[i],
            gram.XᵀX,
            gram.XᵀY[:, i],
            gram.YᵀY[i, i],
            obs,
        ) for i in 1:n
    ]
    M̄ = [Symmetric(posts[i].M̄) for i in 1:n]
    β = Vector{Matrix{T}}(undef, ndraws)
    Σ = Vector{Matrix{T}}(undef, ndraws)
    for d in 1:ndraws
        βd = Matrix{T}(undef, k, n)
        dvar = Vector{T}(undef, n)
        for i in 1:n
            dvar[i] = 1 / rand(rng, Gamma(posts[i].κ̄, 1 / posts[i].τ̄))
            βd[:, i] = rand(rng, MvNormal(posts[i].b̄, Symmetric(dvar[i] * M̄[i])))
        end
        β[d] = βd
        Σ[d] = Matrix{T}(Diagonal(dvar))
    end
    return β, Σ
end

"""
    sample_posterior(
        prior::AsymmetricConjugatePrior,
        gram::NamedTuple,
        obs::Int,
        include_constant::Bool;
        ndraws::Int = 1000,
        rng::Random.AbstractRNG = Random.default_rng(),
    )

Draws `ndraws` i.i.d. samples from the per-equation Normal-Gamma posterior
via `equation_normal_gamma_draws`. No MCMC is needed. Internal; called by the
public `sample_posterior(prior, est; ...)` entry point.
"""
function sample_posterior(
    prior::AsymmetricConjugatePrior,
    gram::NamedTuple,
    obs::Int,
    include_constant::Bool;
    ndraws::Int = 1000,
    rng::Random.AbstractRNG = Random.default_rng(),
    kwargs...,
)
    β, Σ = equation_normal_gamma_draws(
        prior.β0,
        prior.Ω0,
        prior.κ,
        prior.τ,
        gram,
        obs,
        ndraws,
        rng,
    )
    return BVARdraws(
        β,
        Σ,
        :asymmetric_conjugate,
        prior.lags,
        prior.vars,
        prior.names,
        include_constant,
    )
end

"""
    sample_posterior(
        prior::BaumeisterHamiltonPrior,
        gram::NamedTuple,
        obs::Int,
        include_constant::Bool;
        ndraws::Int = 1000,
        rng::Random.AbstractRNG = Random.default_rng(),
    )

Draws `ndraws` i.i.d. samples from the per-equation Normal-Gamma posterior
via `equation_normal_gamma_draws`. No MCMC is needed. Internal; called by the
public `sample_posterior(prior, est; ...)` entry point.
"""
function sample_posterior(
    prior::BaumeisterHamiltonPrior,
    gram::NamedTuple,
    obs::Int,
    include_constant::Bool;
    ndraws::Int = 1000,
    rng::Random.AbstractRNG = Random.default_rng(),
    kwargs...,
)
    β, Σ = equation_normal_gamma_draws(
        prior.m,
        prior.M,
        prior.κ,
        prior.τ,
        gram,
        obs,
        ndraws,
        rng,
    )
    return BVARdraws(
        β,
        Σ,
        :hamilton_baumeister,
        prior.lags,
        prior.vars,
        prior.names,
        include_constant,
    )
end
