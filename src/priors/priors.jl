# Bayesian VAR priors (stage 5): builds a prior from a VARestimate for the
# not-yet-built Turing regression stage (6) to consume.
include("types.jl")
include("minnesota.jl")
include("dummy-observations.jl")
include("conjugate.jl")
include("hamilton.jl")
include("hyperparameters.jl")

"""
    build_prior(
        df::DataFrame,
        end_vec::Vector{Symbol},
        est::VARestimate,
        family::Symbol;
        dummy_components::Vector{Symbol} = Symbol[],
        hyperparameter_method::Symbol = :marginal_likelihood,
        hyperparameters = nothing,
        H = nothing,
    )

Builds a Bayesian VAR prior for the reduced-form VAR summarized by `est`
(the output of `estimate_var`). `df`/`end_vec` are needed alongside `est`
because several families require statistics a `VARestimate` does not carry
forward — the univariate AR(`est.lags`) residual variances used as scale by
the Minnesota and Baumeister-Hamilton families, and the sample means used to
anchor the dummy-observation priors. The Gram blocks a conjugate update
needs (`X'X`, already in `est`, plus `X'Y` and `Y'Y`) are recovered instead
of re-derived, via `gram_blocks`. `df`/`end_vec` must therefore describe the
same data `est` was fit on: `end_vec` must equal `est.names` in the same
order and `df` must have the same number of rows, or the two sources of
statistics would be silently mismatched.

`family` selects one of five prior objects covering nine named prior
families from the literature:

- `:minnesota` — the original (Litterman 1986) Minnesota prior
  (`minnesota_prior`), returning a `MinnesotaPrior`. No prior is placed on
  ``\\Sigma``; it is plugged in at its univariate AR estimate.
- `:normal_wishart` — a natural-conjugate Normal-Wishart prior
  (`normal_wishart_prior`), returning a `NormalWishartPrior`. With
  `dummy_components = Symbol[]` (default) this is the direct
  Kadiyala-Karlsson (1997) Normal-Wishart-compatible Minnesota prior.
  Passing one or more of `:minnesota`, `:sum_of_coefficients`,
  `:dummy_initial_obs`, `:long_run` in `dummy_components` instead builds the
  prior from stacked pseudo/dummy observations (Bańbura, Giannone &
  Reichlin 2010; Giannone, Lenza & Primiceri 2015/2019) — several may be
  combined at once, matching how they are used together in practice.
  `:long_run` additionally requires the keyword `H`.
- `:independent_niw` — an independent Normal-Inverse-Wishart prior
  (`independent_niw_prior`), returning an `IndependentNIWPrior`; unlike
  `:normal_wishart` this can reproduce the original Minnesota prior's
  cross-equation asymmetry, at the cost of having no closed-form posterior
  (Gibbs sampling only). Marginal-likelihood hyperparameter tuning is not
  implemented for this family — `hyperparameter_method` must be `:fixed`.
- `:asymmetric_conjugate` — a reduced-form implementation of the Chan
  (2022) asymmetric conjugate prior (`asymmetric_conjugate_prior`),
  returning an `AsymmetricConjugatePrior`: equation-specific Minnesota-style
  shrinkage with an independent Gamma prior per equation, retaining a
  closed-form posterior.
- `:hamilton_baumeister` — the reduced-form (``A=I``) slice of the
  Baumeister & Hamilton (2019, AER) independent Normal-Gamma reference
  prior (`baumeister_hamilton_prior`), returning a `BaumeisterHamiltonPrior`;
  its full structural extension is deferred to stage 7.

Hyperparameters default (`hyperparameter_method = :marginal_likelihood`) to
maximizing the family's closed-form log marginal likelihood via hand-rolled
golden-section coordinate ascent (`optimize_hyperparameters`). Passing
`hyperparameter_method = :fixed` together with an explicit `hyperparameters`
`NamedTuple` (whose expected keys match the corresponding `*_prior`
function's keyword arguments — e.g. `(λ1, λ2, λ3, λ4)` for `:minnesota`, or
`(λ0, λ1, λ3, κ0, random_walk)` for `:hamilton_baumeister`) uses those values
directly instead; `hyperparameters` may only be passed alongside
`hyperparameter_method = :fixed`.
"""
function build_prior(
    df::DataFrame,
    end_vec::Vector{Symbol},
    est::VARestimate,
    family::Symbol;
    dummy_components::Vector{Symbol} = Symbol[],
    hyperparameter_method::Symbol = :marginal_likelihood,
    hyperparameters = nothing,
    H = nothing,
)
    @assert family in (
        :minnesota,
        :normal_wishart,
        :independent_niw,
        :asymmetric_conjugate,
        :hamilton_baumeister,
    ) "family must be one of :minnesota, :normal_wishart, :independent_niw, :asymmetric_conjugate, :hamilton_baumeister"
    @assert hyperparameter_method in (:marginal_likelihood, :fixed) "hyperparameter_method must be :marginal_likelihood or :fixed"
    @assert family != :independent_niw || hyperparameter_method == :fixed "marginal-likelihood tuning for :independent_niw is not implemented; use hyperparameter_method = :fixed"
    @assert isempty(dummy_components) || family == :normal_wishart "dummy_components is only used when family == :normal_wishart"
    @assert dummy_components ⊆
            (:minnesota, :sum_of_coefficients, :dummy_initial_obs, :long_run) "dummy_components must be a subset of (:minnesota, :sum_of_coefficients, :dummy_initial_obs, :long_run)"
    @assert hyperparameter_method == :marginal_likelihood || !isnothing(hyperparameters) "hyperparameter_method = :fixed requires an explicit `hyperparameters` NamedTuple"
    @assert hyperparameter_method == :fixed || isnothing(hyperparameters) "hyperparameters is ignored unless hyperparameter_method = :fixed; pass hyperparameter_method = :fixed to use your explicit hyperparameters, or omit hyperparameters to use marginal-likelihood optimization"

    Y = get_endogenous(df, end_vec)
    @assert all(y -> !ismissing(y) && isfinite(y), Y) "Endogenous data cannot contain missing or non-finite values"
    @unpack lags, names, include_constant = est
    @assert end_vec == names "end_vec must match est.names — build_prior requires the same variables in the same order as the VARestimate est was built from"
    @assert size(Y, 1) == est.obs + lags "df must be the same data (same number of rows) used to produce est: got $(size(Y, 1)) observations but est.obs + lags = $(est.obs + lags)"

    if hyperparameter_method == :fixed
        if family == :minnesota
            return minnesota_prior(Y, lags, names, include_constant; λ = hyperparameters)
        elseif family == :normal_wishart
            return normal_wishart_prior(
                Y,
                lags,
                names,
                include_constant;
                dummy_components = dummy_components,
                λ = hyperparameters,
                H = H,
            )
        elseif family == :independent_niw
            return independent_niw_prior(
                Y,
                lags,
                names,
                include_constant;
                λ = hyperparameters,
            )
        elseif family == :asymmetric_conjugate
            return asymmetric_conjugate_prior(
                Y,
                lags,
                names,
                include_constant;
                λ = hyperparameters,
            )
        else
            return baumeister_hamilton_prior(
                Y,
                lags,
                names,
                include_constant;
                hyperparameters...,
            )
        end
    end

    gram = gram_blocks(est)
    return optimize_hyperparameters(
        family,
        Y,
        lags,
        names,
        include_constant,
        gram.XᵀX,
        gram.XᵀY,
        gram.YᵀY,
        est.obs;
        dummy_components = dummy_components,
        H = H,
    )
end
