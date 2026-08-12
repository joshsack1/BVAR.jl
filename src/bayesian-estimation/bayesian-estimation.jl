# Bayesian VAR estimation (stage 6): draws from the posterior implied by one
# of the five prior families built in `priors/`.
include("types.jl")
include("conjugate-sampling.jl")
include("independent-niw-gibbs.jl")

"""
    sample_posterior(
        prior::AbstractVARPrior,
        est::VARestimate;
        ndraws::Int = 1000,
        rng::Random.AbstractRNG = Random.default_rng(),
        kwargs...,
    )

Draws `ndraws` samples from the posterior implied by `prior` (the output of
`build_prior`) and the data summarized by `est` (the output of
`estimate_var`), returning a `BVARdraws`.

Four of the five families — `MinnesotaPrior`, `NormalWishartPrior`,
`AsymmetricConjugatePrior`, `BaumeisterHamiltonPrior` — have closed-form
posteriors, so this draws directly and i.i.d. from a known
Normal/Inverse-Wishart/Gamma distribution; no MCMC is involved. The fifth,
`IndependentNIWPrior`, has no closed-form joint posterior, so this instead
runs a Turing `Gibbs`/`GibbsConditional` sampler (see
`bayesian-estimation/independent-niw-gibbs.jl`) alternating its two exact
conditional distributions — which happen to be the same Matrix-Normal and
Inverse-Wishart distributions the other four families draw from directly.
This distinction is entirely internal: every family is called the same way
and returns the same `BVARdraws` shape. `kwargs` are forwarded to the
family-specific method — currently only meaningful for `IndependentNIWPrior`,
whose `burn_in` (defaulting to `ndraws`) can be overridden this way.
"""
function sample_posterior(
    prior::AbstractVARPrior,
    est::VARestimate;
    ndraws::Int = 1000,
    rng::Random.AbstractRNG = Random.default_rng(),
    kwargs...,
)
    @assert prior.lags == est.lags && prior.vars == est.vars "prior and est must come from the same model specification (matching lags/vars)"
    gram = gram_blocks(est)
    return sample_posterior(
        prior,
        gram,
        est.obs,
        est.include_constant;
        ndraws = ndraws,
        rng = rng,
        kwargs...,
    )
end
