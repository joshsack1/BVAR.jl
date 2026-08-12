# Turing-based Gibbs sampler for IndependentNIWPrior — the one family with
# no closed-form joint posterior. Its two conditionals are exactly the same
# Matrix-Normal/Inverse-Wishart distributions the closed-form families draw
# from directly; only alternating between them needs a sampler at all.

"""
    niw_var_model(gram::NamedTuple, obs::Int, prior::IndependentNIWPrior)

The reduced-form VAR likelihood, expressed purely through the Gram blocks
`gram = (XᵀX, XᵀY, YᵀY)` (no raw data needed — the closed-form conditionals
below need nothing more, and reusing the Gram blocks keeps this consistent
with the other four `sample_posterior` methods),

``\\ln p(Y\\mid\\beta,\\Sigma) = -\\frac{Tn}{2}\\ln(2\\pi) - \\frac T2\\ln|\\Sigma|
- \\frac12\\text{tr}\\!\\left(\\Sigma^{-1}\\left(Y'Y-B'X'Y-Y'XB+B'X'XB\\right)\\right),``

where ``B=\\text{reshape}(\\beta,k,n)``, added via `Turing.@addlogprob!` on top
of the prior `β ~ MvNormal(β0,Ω0)`, `Σ ~ InverseWishart(ν0,S0)`. Internal;
built by `sample_posterior(prior::IndependentNIWPrior, ...)`, and sampled
entirely through `niw_cond_β`/`niw_cond_Σ` (see below) rather than by
evaluating this likelihood via generic MCMC.
"""
@model function niw_var_model(gram::NamedTuple, obs::Int, prior::IndependentNIWPrior)
    n = prior.vars
    k = size(gram.XᵀX, 1)
    β ~ MvNormal(prior.β0, prior.Ω0)
    Σ ~ InverseWishart(prior.ν0, Matrix(Symmetric(prior.S0)))
    Bmat = reshape(β, k, n)
    resid = gram.YᵀY - Bmat' * gram.XᵀY - gram.XᵀY' * Bmat + Bmat' * gram.XᵀX * Bmat
    Turing.@addlogprob! -(obs * n / 2) * log(2π) - (obs / 2) * logdet(Symmetric(Σ)) -
                        tr(Symmetric(Σ) \ Symmetric(resid)) / 2
end

"""
    niw_cond_β(c)

Analytical conditional posterior ``\\beta\\mid\\Sigma,Y``, derived from the same
Gram-block identity `sample_posterior`'s other methods use
(``\\text{vec}(X'Y\\Sigma^{-1})`` is the GLS linear term of the stacked
regression ``\\text{vec}(Y)=(I_n\\otimes X)\\beta+\\varepsilon``,
``\\varepsilon\\sim N(0,\\Sigma\\otimes I_T)``):

``\\bar\\Omega^{-1} = \\Omega_0^{-1} + \\Sigma^{-1}\\otimes X'X, \\qquad
\\bar\\beta = \\bar\\Omega\\left(\\Omega_0^{-1}\\beta_0 +
\\text{vec}(X'Y\\Sigma^{-1})\\right),``

so that ``\\beta\\mid\\Sigma,Y\\sim N(\\bar\\beta,\\bar\\Omega)``. Reads `gram`,
`prior`, and the current `Σ` off the `Gibbs` conditioning context; passed to
`Turing.GibbsConditional`. Internal.
"""
function niw_cond_β(c)
    gram = c[@varname(gram)]
    prior = c[@varname(prior)]
    Σ = c[@varname(Σ)]
    Σ_inv = inv(Symmetric(Σ))
    Ω0_inv = inv(prior.Ω0)
    P = Ω0_inv + kron(Σ_inv, gram.XᵀX)
    Ω̄ = inv(Symmetric(P))
    β̄ = Ω̄ * (Ω0_inv * prior.β0 + vec(gram.XᵀY * Σ_inv))
    return MvNormal(β̄, Symmetric(Ω̄))
end

"""
    niw_cond_Σ(c)

Analytical conditional posterior ``\\Sigma\\mid\\beta,Y``, the same
Inverse-Wishart update `normal_wishart_posterior` uses, but with ``\\beta``
held fixed at its current Gibbs value rather than integrated out:

``\\bar S = S_0 + Y'Y - B'X'Y - Y'XB + B'X'XB, \\qquad \\bar\\nu = \\nu_0+T,``

``B=\\text{reshape}(\\beta,k,n)``, so that ``\\Sigma\\mid\\beta,Y\\sim
IW(\\bar S,\\bar\\nu)``. Reads `gram`, `obs`, `prior`, and the current `β` off
the `Gibbs` conditioning context; passed to `Turing.GibbsConditional`.
Internal.
"""
function niw_cond_Σ(c)
    gram = c[@varname(gram)]
    obs = c[@varname(obs)]
    prior = c[@varname(prior)]
    β = c[@varname(β)]
    n = prior.vars
    k = size(gram.XᵀX, 1)
    Bmat = reshape(β, k, n)
    resid = gram.YᵀY - Bmat' * gram.XᵀY - gram.XᵀY' * Bmat + Bmat' * gram.XᵀX * Bmat
    S̄ = Matrix(Symmetric(prior.S0 + resid))
    ν̄ = prior.ν0 + obs
    return InverseWishart(ν̄, S̄)
end

"""
    sample_posterior(
        prior::IndependentNIWPrior,
        gram::NamedTuple,
        obs::Int,
        include_constant::Bool;
        ndraws::Int = 1000,
        rng::Random.AbstractRNG = Random.default_rng(),
        burn_in::Int = ndraws,
    )

Draws `ndraws` samples from the posterior of an `IndependentNIWPrior`, which
has no closed form, via a Turing `Gibbs` sampler alternating the exact
conditionals `niw_cond_β` and `niw_cond_Σ` (`Turing.GibbsConditional` —
no generic MCMC proposal is used for either block, since both conditionals
are known analytically). Runs `burn_in + ndraws` iterations and discards the
first `burn_in` (defaulting to `ndraws`, so calling with only `ndraws` set
works the same as for the four closed-form families). Internal; called by
the public `sample_posterior(prior, est; ...)` entry point.
"""
function sample_posterior(
    prior::IndependentNIWPrior,
    gram::NamedTuple,
    obs::Int,
    include_constant::Bool;
    ndraws::Int = 1000,
    rng::Random.AbstractRNG = Random.default_rng(),
    burn_in::Int = ndraws,
)
    @assert ndraws > 0 "ndraws must be positive"
    @assert burn_in >= 0 "burn_in must be non-negative"
    T = eltype(prior.β0)
    n = prior.vars
    k = size(gram.XᵀX, 1)
    model = niw_var_model(gram, obs, prior)
    sampler = Turing.Gibbs(
        @varname(β) => Turing.GibbsConditional(niw_cond_β),
        @varname(Σ) => Turing.GibbsConditional(niw_cond_Σ),
    )
    chain = Turing.sample(rng, model, sampler, burn_in + ndraws; progress = false)
    β_chain = chain[@varname(β)]
    Σ_chain = chain[@varname(Σ)]
    β = Vector{Matrix{T}}(undef, ndraws)
    Σ = Vector{Matrix{T}}(undef, ndraws)
    for (i, it) in enumerate((burn_in + 1):(burn_in + ndraws))
        β[i] = Matrix{T}(reshape(β_chain[it, 1], k, n))
        Σ[i] = Matrix{T}(Σ_chain[it, 1])
    end
    return BVARdraws(
        β,
        Σ,
        :independent_niw,
        prior.lags,
        prior.vars,
        prior.names,
        include_constant,
    )
end
