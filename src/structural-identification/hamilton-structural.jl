# Baumeister & Hamilton (2019, AER) structural extension (A ≠ I) of the
# reduced-form-only prior built in src/priors/hamilton.jl. The closed-form
# per-equation conjugate posterior already used by stage 6
# (equation_normal_gamma_posterior) turns out to be exactly the conditional
# posterior of (D,B) given A once the regression target for equation i is
# transformed by A's row i — see log_likelihood_weight — so no new
# closed-form derivation is needed, only a sampler over A.

"""
    structural_prior(
        template::AbstractMatrix,
        free::AbstractMatrix{Bool},
        component::Dict{Tuple{Int,Int},<:UnivariateDistribution};
        restrictions::Vector{<:Function} = Function[],
        names::Vector{Symbol} = [Symbol(:y, i) for i in 1:size(template, 1)],
    )

Builds a general prior on the structural rotation matrix ``A`` of
Baumeister & Hamilton (2019, AER, "Structural Interpretation of Vector
Autoregressions with Incomplete Identification"), factored as independent
component priors on the free entries of ``A`` (fixed entries — the
zero-restriction/point-mass case — are read off `template` and never
sampled) times any number of nonlinear joint restrictions on ``(A,B)``,

``p(A) \\propto \\prod_{(i,j)\\ \\text{free}} p_{ij}(A_{ij}) \\cdot
\\prod_r \\exp\\!\\big(r(A,B)\\big).``

Short-run zero restrictions (traditional Cholesky/recursive identification)
are the degenerate limit where every off-diagonal entry above/below the
diagonal is fixed; sign or bound restrictions on a single free entry are
`Truncated(dist, lo, hi)` (Kilian & Murphy 2012-style). Restrictions that mix
several entries of ``A`` (e.g. ``\\det(A)``'s sign, `det_sign_restriction`) or
that also involve ``B`` (e.g. a sign restriction on the long-run multiplier
matrix, `long_run_sign_restriction`) go in `restrictions`, each with
signature ``r(A,B)\\to\\mathbb{R}`` returning an additive log-weight (``0``
if satisfied, ``-\\infty`` if violated, or a genuine log-density for a soft
prior). Consumed by `hamilton_structural_prior`/`sample_structural`.
"""
function structural_prior(
    template::AbstractMatrix{T},
    free::AbstractMatrix{Bool},
    component::Dict{Tuple{Int,Int},<:UnivariateDistribution};
    restrictions::Vector{<:Function} = Function[],
    names::Vector{Symbol} = [Symbol(:y, i) for i in 1:size(template, 1)],
) where {T<:Real}
    n = size(template, 1)
    @assert size(template) == (n, n) "template must be square"
    @assert size(free) == (n, n) "free must be the same size as template"
    @assert Set(keys(component)) == Set(Tuple.(findall(free))) "component must have exactly one distribution per free entry of A (missing or extra keys)"
    @assert length(names) == n "names must have length n"
    return StructuralPrior(
        Matrix{T}(template),
        BitMatrix(free),
        Dict{Tuple{Int,Int},UnivariateDistribution}(component),
        restrictions,
        n,
        names,
    )
end

"""
    parametric_structural_prior(
        θ_prior::Vector{<:UnivariateDistribution},
        map::Function,
        vars::Int;
        extra_logprior::Vector{<:Function} = Function[],
        restrictions::Vector{<:Function} = Function[],
        names::Vector{Symbol} = [Symbol(:y, i) for i in 1:vars],
    )

Builds a *parametric* prior on the structural rotation matrix ``A`` of
Baumeister & Hamilton (2019, AER): independent priors on a parameter vector
``\\theta`` (`θ_prior`, one marginal per element) together with a map
``A=A(\\theta)`` (`map`, taking a length-`length(θ_prior)` vector to an
``n\\times n`` matrix, ``n=`` `vars`), times any number of extra log-prior
terms on functions of ``(\\theta,A)`` and any number of nonlinear joint
restrictions on ``(A,B)``,

``p(\\theta) \\propto \\prod_k p_k(\\theta_k) \\cdot \\prod_g
\\exp\\!\\big(g(\\theta,A(\\theta))\\big) \\cdot \\prod_r
\\exp\\!\\big(r(A(\\theta),B)\\big).``

This is the paper's own parameterization — a handful of economic quantities
(elasticities, multipliers) entering ``A`` nonlinearly, so that the priors
are elicited on interpretable objects rather than on ``A``'s entries as
`structural_prior` does. `extra_logprior` carries priors on *functions* of
``A`` that are not marginals of any single ``\\theta_k`` (the paper's
asymmetric-``t`` prior on ``\\det(\\tilde A)`` and Student-``t`` prior on an
entry of ``\\tilde A^{-1}``, each written by hand as a log-density since
`Distributions.jl` ships no skew-``t``); unlike the ``\\theta`` marginals,
which candidates are drawn from and which therefore cancel, these never
cancel out of any sampler's acceptance ratio. `restrictions` has exactly the
same ``r(A,B)\\to\\mathbb{R}`` contract as in `structural_prior`. The map is
smoke-checked once at ``\\theta=`` `median.(θ_prior)` (the median, not the
mean: a `TDist(ν≤2)` marginal has no finite mean, and `mean` has no method
for a generic `Truncated`) to catch a wrong output shape or element type at
construction rather than deep inside the sampler. Consumed by
`hamilton_structural_prior`/`sample_structural`.

# Examples
```julia
# Baumeister-Hamilton-style supply/demand block: two elasticities, θ = (α, β)
θ_prior = UnivariateDistribution[
    Truncated(TDist(3), 0.0, Inf),    # supply elasticity, sign-restricted
    Truncated(TDist(3), -Inf, 0.0),   # demand elasticity, sign-restricted
]
A_map(θ) = [1.0 -θ[1]; 1.0 -θ[2]]
prior = parametric_structural_prior(
    θ_prior,
    A_map,
    2;
    extra_logprior = Function[(θ, A) -> logpdf(TDist(3), det(A))],
    names = [:quantity, :price],
)
```
"""
function parametric_structural_prior(
    θ_prior::Vector{<:UnivariateDistribution},
    map::Function,
    vars::Int;
    extra_logprior::Vector{<:Function} = Function[],
    restrictions::Vector{<:Function} = Function[],
    names::Vector{Symbol} = [Symbol(:y, i) for i in 1:vars],
)
    @assert !isempty(θ_prior) "θ_prior must contain at least one marginal prior (a ParametricStructuralPrior with no parameters has nothing to sample)"
    @assert length(names) == vars "names must have length vars"
    # Smoke-check the map once, at the prior medians: the median is always
    # defined, whereas mean is Inf/undefined for TDist(ν≤2) and has no method
    # for a generic Truncated — both routine choices for a θ marginal here.
    θ_med = median.(θ_prior)
    A_med = map(θ_med)
    @assert size(A_med) == (vars, vars) "map must return a vars×vars matrix (got $(size(A_med)) for vars = $vars)"
    @assert eltype(A_med) <: Real "map must return a matrix of Reals (got eltype $(eltype(A_med)))"
    return ParametricStructuralPrior{Float64}(
        Vector{UnivariateDistribution}(θ_prior),
        map,
        extra_logprior,
        restrictions,
        vars,
        names,
    )
end

"""
    hamilton_structural_prior(
        reduced_form::BaumeisterHamiltonPrior,
        A_prior::AbstractStructuralPrior,
        Y::AbstractMatrix,
    )

Builds a `HamiltonStructuralPrior` pairing the reduced-form per-equation
``B,D\\mid A`` prior `reduced_form` (from `baumeister_hamilton_prior`) with
the structural prior `A_prior` on ``A`` (from `structural_prior` or
`parametric_structural_prior`), computing
``\\hat S`` (`ar_residual_covariance`) from the same data `Y` and lag order
`reduced_form.lags` used to build `reduced_form`. Checks that the two priors
name the same variables in the same order — otherwise ``A``'s rows would be
silently paired with the wrong reduced-form equations — and that `Y` has one
column per variable.
"""
function hamilton_structural_prior(
    reduced_form::BaumeisterHamiltonPrior{T},
    A_prior::AbstractStructuralPrior{T},
    Y::AbstractMatrix{T},
) where {T<:Real}
    @assert reduced_form.vars == A_prior.vars "reduced_form and A_prior must have the same number of variables"
    @assert reduced_form.names == A_prior.names "reduced_form and A_prior must describe the same variables in the same order"
    @assert size(Y, 2) == reduced_form.vars "Y must have one column per variable in reduced_form (got $(size(Y, 2)) columns, expected $(reduced_form.vars))"
    Ŝ = ar_residual_covariance(Y, reduced_form.lags)
    return HamiltonStructuralPrior(reduced_form, A_prior, Ŝ)
end

function assemble_A(
    prior::StructuralPrior{T},
    values::Dict{Tuple{Int,Int},T},
) where {T<:Real}
    A = copy(prior.template)
    for (idx, v) in values
        A[idx[1], idx[2]] = v
    end
    return A
end

# The θ interface both structural priors share: θ is the vector of quantities
# a sampler actually moves in (A's free entries, or the economic parameters),
# and theta_to_A is the only place that knows how θ becomes A.

nparams(p::StructuralPrior) = count(p.free)
nparams(p::ParametricStructuralPrior) = length(p.θ_prior)

# Column-major order (findall on a BitMatrix is deterministic). This ordering
# is the canonical θ ordering for per-entry priors — marginals, theta_to_A and
# any proposal covariance must all agree on it, so it must never be reordered.
free_indices(p::StructuralPrior) = Tuple.(findall(p.free))

marginals(p::StructuralPrior) = [p.component[idx] for idx in free_indices(p)]
marginals(p::ParametricStructuralPrior) = p.θ_prior

# Independent draw from the marginals, in canonical θ order. Annotated with the
# prior's T (as the old Dict-based draw_A was): the empty case — a fully fixed
# A, no free entries — would otherwise infer Vector{Any} and poison A's eltype.
draw_theta(p::AbstractStructuralPrior{T}, rng::Random.AbstractRNG) where {T<:Real} =
    T[rand(rng, d) for d in marginals(p)]

# Never convert θ's elements to the prior's T: a ForwardDiff.Dual θ must
# survive into A (an autodiff proposal tuner differentiates through this).
function theta_to_A(p::StructuralPrior, θ::AbstractVector)
    A = Matrix{promote_type(eltype(p.template), eltype(θ))}(p.template)
    for (k, idx) in enumerate(free_indices(p))
        A[idx[1], idx[2]] = θ[k]
    end
    return A
end

function theta_to_A(p::ParametricStructuralPrior, θ::AbstractVector)
    A = p.map(θ)
    @assert size(A) == (p.vars, p.vars) "the parametric prior's map returned a $(size(A)) matrix, expected ($(p.vars), $(p.vars))"
    return Matrix(A)  # unannotated: A's eltype is whatever the map produced
end

marginal_log_prior(p::AbstractStructuralPrior, θ::AbstractVector) =
    sum(logpdf(d, θ[k]) for (k, d) in enumerate(marginals(p)); init = 0.0)

extra_log_prior(p::StructuralPrior, θ::AbstractVector, A::AbstractMatrix) = 0.0
extra_log_prior(p::ParametricStructuralPrior, θ::AbstractVector, A::AbstractMatrix) =
    isempty(p.extra_logprior) ? 0.0 : sum(g(θ, A) for g in p.extra_logprior)

draw_A(prior::AbstractStructuralPrior, rng::Random.AbstractRNG) =
    theta_to_A(prior, draw_theta(prior, rng))

"""
    log_likelihood_weight(
        A::AbstractMatrix,
        m::Vector{<:AbstractVector},
        M::Vector{<:AbstractMatrix},
        κ::Vector,
        Ŝ::AbstractMatrix,
        gram::NamedTuple,
        Σ̂::AbstractMatrix,
        obs::Int,
    )

Computes Baumeister & Hamilton (2019, AER)'s marginal (quasi-)likelihood of
a candidate structural rotation matrix ``A`` (their eq. 12, with the
``p(A)`` factor omitted — see `sample_structural`'s docstring for why),

``\\ln p(Y\\mid A) = \\frac{T}{2}\\ln\\det(A\\hat\\Omega_TA') +
\\sum_i\\kappa_i\\ln\\tau_i(A) - \\sum_i\\kappa_i^*\\ln\\!\\big((2/T)\\tau_i^*(A)\\big),``

where ``\\hat\\Omega_T`` (`Σ̂`) is the reduced-form MLE residual covariance,
``\\tau_i(A)=\\kappa_i\\,a_i'\\hat Sa_i`` (``a_i'=A[i,:]``), and
``\\tau_i^*(A)``/``\\kappa_i^*`` come from `equation_normal_gamma_posterior`
applied to the transformed dependent variable ``a_i'y_t``, via
``X'y_i(A)=X'Y\\cdot a_i``, ``y_i(A)'y_i(A)=a_i'Y'Ya_i``. Returns
`(logw, posts)`, `posts[i]` the per-equation posterior so that a `(B,D)\\mid A`
draw reuses it directly rather than recomputing it. ``A``'s element type is
deliberately independent of the prior's — everything here promotes — so a
`ForwardDiff.Dual`-valued ``A`` can meet `Float64` prior/data arguments.
Internal; called by `evaluate_candidate`.
"""
function log_likelihood_weight(
    A::AbstractMatrix{S},
    m::Vector{<:AbstractVector{T}},
    M::Vector{<:AbstractMatrix{T}},
    κ::Vector{T},
    Ŝ::AbstractMatrix{T},
    gram::NamedTuple,
    Σ̂::AbstractMatrix{T},
    obs::Int,
) where {S<:Real,T<:Real}
    n = size(A, 1)
    logw = (obs / 2) * logdet(Symmetric(A * Σ̂ * A'))
    posts = Vector{NamedTuple}(undef, n)
    for i in 1:n
        aᵢ = A[i, :]
        τᵢ = κ[i] * dot(aᵢ, Ŝ, aᵢ)
        Xᵀyᵢ = gram.XᵀY * aᵢ
        yᵀyᵢ = dot(aᵢ, gram.YᵀY, aᵢ)
        post =
            equation_normal_gamma_posterior(m[i], M[i], κ[i], τᵢ, gram.XᵀX, Xᵀyᵢ, yᵀyᵢ, obs)
        posts[i] = post
        logw += κ[i] * log(τᵢ) - post.κ̄ * log((2 / obs) * post.τ̄)
    end
    return logw, posts
end

"""
    evaluate_candidate(
        prior::HamiltonStructuralPrior,
        θ::AbstractVector,
        gram::NamedTuple,
        Σ̂::AbstractMatrix,
        obs::Int,
        rng::Random.AbstractRNG,
    )

Scores an *already drawn* parameter vector `θ`: maps it to ``A``
(`theta_to_A`), draws ``(B,D)\\mid A`` from the exact conjugate posterior
(`log_likelihood_weight`), then adds every extra log-prior term on
``(\\theta,A)`` and every joint restriction in `prior.A_prior.restrictions`.
Returns `(A, B, d, logw)`, `d` the diagonal of `D`. Split out from
`draw_candidate` so a sampler that proposes ``\\theta`` itself (a random-walk
chain) can reuse the identical weight; internal.
"""
function evaluate_candidate(
    prior::HamiltonStructuralPrior{T},
    θ::AbstractVector,
    gram::NamedTuple,
    Σ̂::AbstractMatrix{T},
    obs::Int,
    rng::Random.AbstractRNG,
) where {T<:Real}
    rf = prior.reduced_form
    A = theta_to_A(prior.A_prior, θ)
    logw, posts = log_likelihood_weight(A, rf.m, rf.M, rf.κ, prior.Ŝ, gram, Σ̂, obs)
    # Extra log-prior terms on (θ,A) never cancel from any acceptance ratio,
    # unlike the marginal θ priors the independence proposal draws from.
    logw += extra_log_prior(prior.A_prior, θ, A)
    n = rf.vars
    k = length(rf.m[1])
    B = Matrix{T}(undef, k, n)
    d = Vector{T}(undef, n)
    for i in 1:n
        d[i] = 1 / rand(rng, Gamma(posts[i].κ̄, 1 / posts[i].τ̄))
        B[:, i] = rand(rng, MvNormal(posts[i].b̄, Symmetric(d[i] * posts[i].M̄)))
    end
    for r in prior.A_prior.restrictions
        logw += r(A, B)
    end
    return A, B, d, logw
end

"""
    draw_candidate(
        prior::HamiltonStructuralPrior,
        gram::NamedTuple,
        Σ̂::AbstractMatrix,
        obs::Int,
        rng::Random.AbstractRNG,
    )

Draws one candidate structural triple ``(A,B,D)``: ``\\theta`` from
`prior.A_prior`'s marginal priors (its component priors on ``A``'s free
entries, or its priors on the economic parameters), then scores it with
`evaluate_candidate`. Returns `(A, B, d, logw)`, `d` the diagonal of `D`.
Internal; the shared draw step of both `sample_structural_sir` and
`sample_structural_mh`.
"""
draw_candidate(
    prior::HamiltonStructuralPrior{T},
    gram::NamedTuple,
    Σ̂::AbstractMatrix{T},
    obs::Int,
    rng::Random.AbstractRNG,
) where {T<:Real} = evaluate_candidate(
    prior,
    draw_theta(prior.A_prior, rng),
    gram,
    Σ̂,
    obs,
    rng,
)

"""
    sample_structural_sir(
        prior::HamiltonStructuralPrior,
        gram::NamedTuple,
        Σ̂::AbstractMatrix,
        obs::Int,
        include_constant::Bool;
        ndraws::Int = 1000,
        rng::Random.AbstractRNG = Random.default_rng(),
        oversample::Int = 10,
    )

Draws `ndraws` samples of ``(A,B,D)`` via sampling-importance-resampling:
draws `ndraws * oversample` candidates (`draw_candidate`), then resamples
`ndraws` of them with replacement in proportion to their
(softmax-normalized) importance weights. Returns
`(draws::StructuralDraws, diagnostics::NamedTuple)` with
`diagnostics = (ess = ...,)`, the effective sample size
``1/\\sum_\\ell\\tilde w_\\ell^2`` — inspect it before trusting the draws; a
small `ess` relative to `ndraws` signals importance-weight collapse (see
`sample_structural`'s docstring for when to prefer `method = :mh` instead).
Errors if *every* candidate is rejected (all weights ``-\\infty``), which
would otherwise normalize to `NaN` weights and silently resample garbage.
Internal; called by the public `sample_structural(prior, est; ...)` entry
point.
"""
function sample_structural_sir(
    prior::HamiltonStructuralPrior{T},
    gram::NamedTuple,
    Σ̂::AbstractMatrix{T},
    obs::Int,
    include_constant::Bool;
    ndraws::Int = 1000,
    rng::Random.AbstractRNG = Random.default_rng(),
    oversample::Int = 10,
) where {T<:Real}
    @assert ndraws > 0 "ndraws must be positive"
    @assert oversample > 0 "oversample must be positive"
    ncandidates = ndraws * oversample
    As = Vector{Matrix{T}}(undef, ncandidates)
    Bs = Vector{Matrix{T}}(undef, ncandidates)
    ds = Vector{Vector{T}}(undef, ncandidates)
    logw = Vector{T}(undef, ncandidates)
    for c in 1:ncandidates
        As[c], Bs[c], ds[c], logw[c] = draw_candidate(prior, gram, Σ̂, obs, rng)
    end
    @assert any(isfinite, logw) "sample_structural_sir: every one of the $ncandidates candidate draws violates the structural prior's restrictions (all importance weights are -Inf); the restrictions in prior.A_prior.restrictions may be jointly infeasible given the component priors, or oversample may need to be much larger"
    logw .-= maximum(logw)
    w = exp.(logw)
    w ./= sum(w)
    ess = 1 / sum(abs2, w)
    cw = cumsum(w)
    idx = [searchsortedfirst(cw, rand(rng)) for _ in 1:ndraws]
    draws = StructuralDraws(
        As[idx],
        Bs[idx],
        ds[idx],
        prior.reduced_form.lags,
        prior.reduced_form.vars,
        prior.reduced_form.names,
        include_constant,
    )
    return draws, (ess = ess,)
end

"""
    sample_structural_mh(
        prior::HamiltonStructuralPrior,
        gram::NamedTuple,
        Σ̂::AbstractMatrix,
        obs::Int,
        include_constant::Bool;
        ndraws::Int = 1000,
        rng::Random.AbstractRNG = Random.default_rng(),
        burn_in::Int = ndraws,
    )

Draws `ndraws` samples of ``(A,B,D)`` via an independence-chain
Metropolis-Hastings sampler: at each iteration, draws a candidate
(`draw_candidate`) and accepts it in place of the current draw with
probability ``\\min(1,w_{\\text{new}}/w_{\\text{cur}})`` (the same importance
weight `sample_structural_sir` uses). Runs a fixed, bounded `burn_in +
ndraws` iterations and discards the first `burn_in` (defaulting to
`ndraws`). Returns `(draws::StructuralDraws, diagnostics::NamedTuple)` with
`diagnostics = (acceptance_rate = ...,)`. Internal; called by the public
`sample_structural(prior, est; ...)` entry point.
"""
function sample_structural_mh(
    prior::HamiltonStructuralPrior{T},
    gram::NamedTuple,
    Σ̂::AbstractMatrix{T},
    obs::Int,
    include_constant::Bool;
    ndraws::Int = 1000,
    rng::Random.AbstractRNG = Random.default_rng(),
    burn_in::Int = ndraws,
) where {T<:Real}
    @assert ndraws > 0 "ndraws must be positive"
    @assert burn_in >= 0 "burn_in must be non-negative"
    total = burn_in + ndraws
    A_cur, B_cur, d_cur, logw_cur = draw_candidate(prior, gram, Σ̂, obs, rng)
    As = Vector{Matrix{T}}(undef, ndraws)
    Bs = Vector{Matrix{T}}(undef, ndraws)
    ds = Vector{Vector{T}}(undef, ndraws)
    naccept = 0
    for iter in 1:total
        A_prop, B_prop, d_prop, logw_prop = draw_candidate(prior, gram, Σ̂, obs, rng)
        if log(rand(rng)) < logw_prop - logw_cur
            A_cur, B_cur, d_cur, logw_cur = A_prop, B_prop, d_prop, logw_prop
            naccept += 1
        end
        if iter > burn_in
            As[iter - burn_in] = A_cur
            Bs[iter - burn_in] = B_cur
            ds[iter - burn_in] = d_cur
        end
    end
    draws = StructuralDraws(
        As,
        Bs,
        ds,
        prior.reduced_form.lags,
        prior.reduced_form.vars,
        prior.reduced_form.names,
        include_constant,
    )
    return draws, (acceptance_rate = naccept / total,)
end

"""
    sample_structural(
        prior::HamiltonStructuralPrior,
        est::VARestimate;
        ndraws::Int = 1000,
        rng::Random.AbstractRNG = Random.default_rng(),
        method::Symbol = :mh,
        kwargs...,
    )

Draws `ndraws` samples of the structural triple ``(A,B,D)`` from the
posterior implied by `prior` and the data summarized by `est`, following
Baumeister & Hamilton (2019, AER)'s marginal posterior of ``A`` (their eq.
12),

``p(A\\mid Y_T) \\propto p(A)\\,\\frac{\\big[\\det(A\\hat\\Omega_TA')\\big]^{T/2}}
{\\prod_i\\big[(2/T)\\tau_i^*(A)\\big]^{\\kappa_i^*}}\\,\\prod_i \\tau_i(A)^{\\kappa_i}.``

Candidate ``A``'s are drawn from `prior.A_prior`'s component priors — which
are exactly the ``p(A)`` factor above restricted to `A_prior`'s free entries
— so that factor cancels out of both the importance weight (`method =
:sir`) and the Metropolis-Hastings acceptance ratio (`method = :mh`, the
default), leaving only the likelihood ratio above (`log_likelihood_weight`)
and any nonlinear `A_prior.restrictions`; ``B,D\\mid A`` are then drawn from
the exact conjugate posterior, which cancels the same way. This holds
regardless of whether a restriction depends on ``A`` alone (short-run/sign)
or on ``(A,B)`` jointly (long-run) — one uniform mechanism handles all three.

`method = :mh` (default) is the more robust choice: both methods share
essentially the same per-candidate cost, but `:sir`'s failure mode
(importance-weight collapse — a real risk here, since the posterior's
``T/2`` exponent is steep) can go unnoticed unless the returned `ess`
diagnostic is checked, whereas `:mh`'s failure mode (a low
`acceptance_rate`) is immediately visible. `:sir` is provided for direct
comparison against the paper's own (importance-sampling-based) algorithm.
`kwargs` are forwarded to the chosen method (`oversample` for `:sir`,
`burn_in` for `:mh`). Returns `(draws::StructuralDraws,
diagnostics::NamedTuple)`.
"""
function sample_structural(
    prior::HamiltonStructuralPrior,
    est::VARestimate;
    ndraws::Int = 1000,
    rng::Random.AbstractRNG = Random.default_rng(),
    method::Symbol = :mh,
    kwargs...,
)
    @assert prior.reduced_form.lags == est.lags && prior.reduced_form.vars == est.vars "prior and est must come from the same model specification (matching lags/vars)"
    @assert method in (:sir, :mh) "method must be :sir or :mh"
    gram = gram_blocks(est)
    Σ̂ = est.Σ
    sampler = method == :sir ? sample_structural_sir : sample_structural_mh
    return sampler(
        prior,
        gram,
        Σ̂,
        est.obs,
        est.include_constant;
        ndraws = ndraws,
        rng = rng,
        kwargs...,
    )
end

"""
    det_sign_restriction(sgn::Int)

Returns a joint restriction closure, for `StructuralPrior`'s `restrictions`,
enforcing ``\\mathrm{sign}(\\det A) = `` `sgn` (``\\pm1``): ``0`` (satisfied)
or ``-\\infty`` (violated). Baumeister & Hamilton (2019, Section IV.A) place
an analogous *soft* (asymmetric-``t``) prior on ``\\det(A)``'s sign rather
than a hard cutoff; this is the simpler hard-restriction special case.
"""
function det_sign_restriction(sgn::Int)
    @assert sgn in (-1, 1) "sgn must be -1 or 1"
    return (A, B) -> sign(det(A)) == sgn ? 0.0 : -Inf
end

"""
    long_run_sign_restriction(i::Int, j::Int, sgn::Int, lags::Int, include_constant::Bool)

Returns a joint restriction closure, for `StructuralPrior`'s `restrictions`,
enforcing a sign restriction on the ``(i,j)`` entry of the long-run
multiplier matrix (`long_run_multiplier`) — the structural analogue of a
Blanchard & Quah (1989) long-run restriction, expressed through the same
mechanism used for short-run and sign restrictions. `sgn` is ``\\pm1``.
"""
function long_run_sign_restriction(
    i::Int,
    j::Int,
    sgn::Int,
    lags::Int,
    include_constant::Bool,
)
    @assert sgn in (-1, 1) "sgn must be -1 or 1"
    return (A, B) ->
        sign(long_run_multiplier(A, B, lags, include_constant)[i, j]) == sgn ? 0.0 : -Inf
end
