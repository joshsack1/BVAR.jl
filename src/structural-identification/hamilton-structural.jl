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
`truncated(dist, lo, hi)` (Kilian & Murphy 2012-style). Restrictions that mix
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
    truncated(TDist(3), 0.0, Inf),    # supply elasticity, sign-restricted
    truncated(TDist(3), -Inf, 0.0),   # demand elasticity, sign-restricted
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
        rf::BaumeisterHamiltonPrior,
        Ŝ::AbstractMatrix,
        gram::NamedTuple,
        Σ̂::AbstractMatrix,
        obs::Real,
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
``X'y_i(A)=X'Y\\cdot a_i``, ``y_i(A)'y_i(A)=a_i'Y'Ya_i``. When the
reduced-form prior carries an `η` (`rf.η !== nothing`) equation ``i``'s prior
mean is itself ``A``-dependent, ``m_i(A)=\\eta'a_i`` (the paper's structural
random-walk construction when ``\\eta``'s rows are the unit vectors ``e_i`` on
each equation's own first lag); otherwise it is the constant `rf.m[i]`. Returns
`(logw, posts)`, `posts[i]` the per-equation posterior so that a `(B,D)\\mid A`
draw reuses it directly rather than recomputing it. ``A``'s element type is
deliberately independent of the prior's — everything here promotes — so a
`ForwardDiff.Dual`-valued ``A`` can meet `Float64` prior/data arguments.
Internal; called by `evaluate_candidate`.
"""
function log_likelihood_weight(
    A::AbstractMatrix{S},
    rf::BaumeisterHamiltonPrior{T},
    Ŝ::AbstractMatrix{T},
    gram::NamedTuple,
    Σ̂::AbstractMatrix{T},
    obs::Real,
) where {S<:Real,T<:Real}
    n = size(A, 1)
    # Hoisted out of the loop so the Union split on the A-dependent-mean case
    # happens once per candidate A rather than once per equation.
    η = rf.η
    # A near-singular A can make the (mathematically PSD) product numerically
    # indefinite or exactly singular; logdet would throw a DomainError there
    # and crash a chain mid-run. A candidate with zero density should instead
    # carry a -Inf weight so the sampler rejects it.
    ld, ld_sign = logabsdet(Symmetric(A * Σ̂ * A'))
    logw = ld_sign > 0 ? (obs / 2) * ld : oftype(ld, -Inf)
    posts = Vector{NamedTuple}(undef, n)
    for i in 1:n
        aᵢ = A[i, :]
        τᵢ = rf.κ[i] * dot(aᵢ, Ŝ, aᵢ)
        # η' is a lazy Adjoint: no transpose is materialized, and a Dual-valued
        # aᵢ promotes mᵢ rather than being narrowed to the prior's T.
        mᵢ = η === nothing ? rf.m[i] : η' * aᵢ
        Xᵀyᵢ = gram.XᵀY * aᵢ
        yᵀyᵢ = dot(aᵢ, gram.YᵀY, aᵢ)
        post = equation_normal_gamma_posterior(
            mᵢ,
            rf.M[i],
            rf.κ[i],
            τᵢ,
            gram.XᵀX,
            Xᵀyᵢ,
            yᵀyᵢ,
            obs,
        )
        posts[i] = post
        # τ and τ̄ are positive for any nondegenerate A (Ŝ and the posterior
        # sum of squares are PSD); at a degenerate A rounding can push them to
        # 0 or barely below, where log would throw — zero density again.
        if τᵢ > 0 && post.τ̄ > 0
            logw += rf.κ[i] * log(τᵢ) - post.κ̄ * log((2 / obs) * post.τ̄)
        else
            logw = oftype(logw, -Inf)
        end
    end
    return logw, posts
end

"""
    evaluate_candidate(
        prior::HamiltonStructuralPrior,
        θ::AbstractVector,
        gram::NamedTuple,
        Σ̂::AbstractMatrix,
        obs::Real,
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
    obs::Real,
    rng::Random.AbstractRNG,
) where {T<:Real}
    rf = prior.reduced_form
    A = theta_to_A(prior.A_prior, θ)
    logw, posts = log_likelihood_weight(A, rf, prior.Ŝ, gram, Σ̂, obs)
    # Extra log-prior terms on (θ,A) never cancel from any acceptance ratio,
    # unlike the marginal θ priors the independence proposal draws from.
    logw += extra_log_prior(prior.A_prior, θ, A)
    n = rf.vars
    k = length(rf.m[1])
    # A zero-density candidate (singular A, degenerate posterior scale, or a
    # -Inf/NaN extra term) can carry posts the Gamma/MvNormal draws below
    # would throw on, and can never be accepted or resampled anyway — reject
    # it outright rather than drawing (B, D) | A from garbage.
    isfinite(logw) || return A, zeros(T, k, n), zeros(T, n), T(-Inf)
    B = Matrix{T}(undef, k, n)
    d = Vector{T}(undef, n)
    for i in 1:n
        d[i] = 1 / rand(rng, Gamma(posts[i].κ̄, 1 / posts[i].τ̄))
        B[:, i] = rand(rng, MvNormal(posts[i].b̄, Symmetric(d[i] * posts[i].M̄)))
    end
    for r in prior.A_prior.restrictions
        logw += r(A, B)
    end
    # A NaN from a user restriction would silently poison :sir's softmax
    # (maximum(logw) turns NaN); make it an explicit rejection instead.
    isnan(logw) && (logw = T(-Inf))
    return A, B, d, logw
end

"""
    draw_candidate(
        prior::HamiltonStructuralPrior,
        gram::NamedTuple,
        Σ̂::AbstractMatrix,
        obs::Real,
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
    obs::Real,
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
        obs::Real,
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
    obs::Real,
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
        obs::Real,
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
    obs::Real,
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
    sample_structural_rwmh(
        prior::HamiltonStructuralPrior,
        gram::NamedTuple,
        Σ̂::AbstractMatrix,
        obs::Real,
        include_constant::Bool;
        ndraws::Int = 1000,
        rng::Random.AbstractRNG = Random.default_rng(),
        burn_in::Int = ndraws,
        θ₀::Union{Nothing,AbstractVector} = nothing,
        proposal_scale::Union{Nothing,AbstractMatrix} = nothing,
        ξ::Real = 1.0,
        proposal_df::Real = 2.0,
    )

Draws `ndraws` samples of ``(A,B,D)`` via Baumeister & Hamilton (2019,
AER)'s own algorithm: a random-walk Metropolis-Hastings chain on the
marginal posterior of ``\\theta`` (their eq. 12, ``B,D`` integrated out),
with the paper's fat-tailed proposal

``\\theta' = \\theta + \\sqrt{\\xi}\\,L z/\\sqrt{u/\\nu}, \\qquad
z\\sim N(0,I_c),\\; u\\sim\\chi^2_\\nu,``

a multivariate Student-``t`` with ``\\nu = `` `proposal_df` degrees of
freedom (their choice: 2) and scale ``\\xi\\cdot`` `proposal_scale`
(``L`` its Cholesky factor) — one shared mixing variable ``u`` per proposal,
so the step is a genuine multivariate ``t``, and symmetric, so the proposal
density cancels from the acceptance ratio. What does *not* cancel here —
unlike in `sample_structural_mh`, whose candidates are drawn from the prior
itself — is the marginal prior of ``\\theta``: the acceptance ratio is
``(\\text{loglik} + \\ln p(\\theta))_{\\text{prop}} - (\\cdot)_{\\text{cur}}``,
with `extra_logprior` terms and restrictions inside the first factor as
always. Restrictions that depend on the per-iteration ``B`` draw keep the
same pseudo-marginal semantics as `sample_structural_mh` (the current
state's weight is recycled until the next acceptance).

By default (`proposal_scale = nothing`) the proposal is tuned the way the
paper tunes it: `tune_rwmh_proposal` maximizes the marginal log posterior
from `θ₀` (or the marginals' medians) and the chain starts at the mode with
`proposal_scale` set to the inverse Hessian there. Supplying
`proposal_scale` (with or without `θ₀`) skips tuning entirely. `ξ` is the
paper's hand-tuned step multiplier: shrink it if the `acceptance_rate`
diagnostic is far below their 30-35% target, grow it if far above; the
posterior covariance of the returned `θ` draws is the natural
`proposal_scale` for a retuned second run.

Prefer this over `sample_structural_mh` when the independence chain's
acceptance rate collapses — its whole-``A``-at-once prior draws are accepted
ever more rarely as the number of free parameters grows, while a local
random walk keeps moving — and for full B&H replication with a
`ParametricStructuralPrior`. Returns `(draws::StructuralDraws,
diagnostics::NamedTuple)` with `diagnostics = (acceptance_rate = ..., θ =
...)`, `θ` the `ndraws × c` matrix of post-burn-in ``\\theta`` draws in
canonical order (`free_indices` order for a `StructuralPrior`, `θ_prior`
order for a `ParametricStructuralPrior`). Internal; called by the public
`sample_structural(prior, est; ...)` entry point.
"""
function sample_structural_rwmh(
    prior::HamiltonStructuralPrior{T},
    gram::NamedTuple,
    Σ̂::AbstractMatrix{T},
    obs::Real,
    include_constant::Bool;
    ndraws::Int = 1000,
    rng::Random.AbstractRNG = Random.default_rng(),
    burn_in::Int = ndraws,
    θ₀::Union{Nothing,AbstractVector} = nothing,
    proposal_scale::Union{Nothing,AbstractMatrix} = nothing,
    ξ::Real = 1.0,
    proposal_df::Real = 2.0,
) where {T<:Real}
    @assert ndraws > 0 "ndraws must be positive"
    @assert burn_in >= 0 "burn_in must be non-negative"
    @assert ξ > 0 "ξ must be positive"
    @assert proposal_df > 0 "proposal_df must be positive"
    c = nparams(prior.A_prior)
    @assert c > 0 "the structural prior has no free parameters, so a random-walk chain has nothing to move; use method = :mh or :sir, which reduce to the fixed-A path"
    θ_cur = θ₀ === nothing ? median.(marginals(prior.A_prior)) : collect(float.(θ₀))
    @assert length(θ_cur) == c "θ₀ must have one entry per free parameter (got $(length(θ_cur)), expected $c)"
    local scale
    if proposal_scale === nothing
        θ_cur, scale = tune_rwmh_proposal(prior, gram, Σ̂, obs, θ_cur)
    else
        @assert size(proposal_scale) == (c, c) "proposal_scale must be $c×$c (one row/column per free parameter; got $(size(proposal_scale)))"
        scale = proposal_scale
    end
    scale_sym = Symmetric(Matrix(scale))
    @assert isposdef(scale_sym) "proposal_scale must be positive definite"
    # ξ folded into the Cholesky factor once, not per iteration
    Lξ = sqrt(ξ) .* cholesky(scale_sym).L
    A_cur, B_cur, d_cur, logw_cur = evaluate_candidate(prior, θ_cur, gram, Σ̂, obs, rng)
    lp_cur = marginal_log_prior(prior.A_prior, θ_cur)
    @assert isfinite(logw_cur + lp_cur) "the log posterior is not finite at the chain's starting point; supply a θ₀ inside every marginal prior's support that satisfies every restriction"
    total = burn_in + ndraws
    As = Vector{Matrix{T}}(undef, ndraws)
    Bs = Vector{Matrix{T}}(undef, ndraws)
    ds = Vector{Vector{T}}(undef, ndraws)
    θs = Matrix{Float64}(undef, ndraws, c)
    naccept = 0
    for iter in 1:total
        θ_prop =
            θ_cur .+ (Lξ * randn(rng, c)) ./ sqrt(rand(rng, Chisq(proposal_df)) / proposal_df)
        lp_prop = marginal_log_prior(prior.A_prior, θ_prop)
        # Only score proposals inside the marginals' support: outside it the
        # θ → A map itself can be undefined (e.g. B&H's -1/χ transform).
        if isfinite(lp_prop)
            A_prop, B_prop, d_prop, logw_prop =
                evaluate_candidate(prior, θ_prop, gram, Σ̂, obs, rng)
            # The marginal prior no longer cancels (the proposal is not the
            # prior, unlike sample_structural_mh); the symmetric t step
            # contributes nothing.
            if log(rand(rng)) < (logw_prop + lp_prop) - (logw_cur + lp_cur)
                θ_cur, A_cur, B_cur, d_cur = θ_prop, A_prop, B_prop, d_prop
                logw_cur, lp_cur = logw_prop, lp_prop
                naccept += 1
            end
        end
        if iter > burn_in
            As[iter - burn_in] = A_cur
            Bs[iter - burn_in] = B_cur
            ds[iter - burn_in] = d_cur
            θs[iter - burn_in, :] = θ_cur
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
    return draws, (acceptance_rate = naccept / total, θ = θs)
end

# Internal core of structural_log_posterior, taking the same precomputed
# blocks the samplers carry so sample_structural_rwmh's tuner can call it
# without a VARestimate in hand.
function structural_log_posterior(
    prior::HamiltonStructuralPrior{T},
    gram::NamedTuple,
    Σ̂::AbstractMatrix{T},
    obs::Real,
) where {T<:Real}
    rf = prior.reduced_form
    A_prior = prior.A_prior
    return function (θ::AbstractVector)
        lp = marginal_log_prior(A_prior, θ)
        # Out of the marginals' support: return before calling the map, whose
        # transforms (e.g. Baumeister & Hamilton's -1/χ) can be undefined there.
        isfinite(lp) || return -Inf
        A = theta_to_A(A_prior, θ)
        logw, posts = log_likelihood_weight(A, rf, prior.Ŝ, gram, Σ̂, obs)
        logw += extra_log_prior(A_prior, θ, A)
        # Zero density already — don't hand a degenerate A/B̄ to user
        # restriction closures (long_run_multiplier would invert singular A).
        isfinite(logw) || return -Inf
        if !isempty(A_prior.restrictions)
            B̄ = reduce(hcat, (posts[i].b̄ for i in 1:rf.vars))
            for r in A_prior.restrictions
                logw += r(A, B̄)
            end
        end
        out = lp + logw
        # NaN (e.g. from a user restriction) would wedge an optimizer or a
        # chain; zero density is the safe reading.
        return isnan(out) ? -Inf : out
    end
end

"""
    structural_log_posterior(prior::HamiltonStructuralPrior, est::VARestimate)

Returns a deterministic closure ``\\theta \\to \\ln p(\\theta\\mid Y_T)`` (up
to a constant): Baumeister & Hamilton (2019, AER)'s marginal posterior of the
structural parameters (their eq. 12) — the collapsed likelihood
(`log_likelihood_weight`, ``B,D`` integrated out) plus the marginal log prior
of ``\\theta`` plus any `extra_logprior` terms, with each joint restriction
``r(A,B)`` evaluated at the *conditional posterior mean* ``\\bar B(A)`` rather
than at a random ``B`` draw. It is therefore the exact `method = :rwmh`
target when no restriction depends on ``B`` (the common case — short-run,
sign, and determinant restrictions touch ``A`` alone), and a deterministic
approximation otherwise. Returns ``-\\infty`` outside the marginals' support
without evaluating the ``\\theta\\to A`` map.

The closure is `ForwardDiff`-differentiable, which is how
`sample_structural`'s default `:rwmh` tuning uses it (mode-finding plus
inverse-Hessian proposal scaling, the package's analogue of the paper's
`fminunc` step); it is exported so the same tuning can be done by hand with
any optimizer — maximize the closure, then pass the mode and (scaled) inverse
Hessian to `sample_structural` as `θ₀` and `proposal_scale`.
"""
function structural_log_posterior(
    prior::HamiltonStructuralPrior,
    est::VARestimate,
)
    @assert prior.reduced_form.lags == est.lags && prior.reduced_form.vars == est.vars "prior and est must come from the same model specification (matching lags/vars)"
    return structural_log_posterior(prior, gram_blocks(est), est.Σ, est.obs)
end

# Baumeister & Hamilton (2019)'s proposal-tuning step for the random-walk
# sampler: maximize the marginal log posterior from θ₀ (BFGS, ForwardDiff
# gradients) and return (mode, inverse Hessian there) as the proposal's
# location and scale. Mirrors their fminunc guard: whenever optimization
# fails, its optimum is not finite, or the inverse Hessian is not positive
# definite, fall back to (θ₀, an IQR-based diagonal) with a warning — IQR
# rather than var because the marginals may be fat-tailed (var(TDist(ν ≤ 2))
# is infinite) or Truncated (no var method).
function tune_rwmh_proposal(
    prior::HamiltonStructuralPrior{T},
    gram::NamedTuple,
    Σ̂::AbstractMatrix{T},
    obs::Real,
    θ₀::AbstractVector,
) where {T<:Real}
    logpost = structural_log_posterior(prior, gram, Σ̂, obs)
    @assert isfinite(logpost(θ₀)) "the log posterior is not finite at the tuner's starting point θ₀ = $θ₀; supply a θ₀ inside the marginal priors' support that satisfies every restriction"
    # normal-consistent IQR-based scale: (IQR/1.349)² per marginal
    fallback =
        Diagonal([((quantile(d, 0.75) - quantile(d, 0.25)) / 1.349)^2 for d in marginals(prior.A_prior)])
    neg(θ) = -logpost(θ)
    # explicit ForwardDiff gradient (Optim ≥ 2 takes ADTypes objects, not
    # :forward, for its autodiff kwarg; this sidesteps a third dependency)
    neg_grad!(G, θ) = copyto!(G, ForwardDiff.gradient(neg, θ))
    result = try
        # iterations is Optim's own default, passed explicitly so the bound on
        # this loop is visible here rather than inherited silently
        Optim.optimize(
            neg,
            neg_grad!,
            float.(collect(θ₀)),
            Optim.BFGS(),
            Optim.Options(iterations = 1000),
        )
    catch err
        err isa InterruptException && rethrow()
        @warn "proposal tuning failed to optimize the marginal log posterior — often benign (a marginal or extra_logprior term ForwardDiff cannot differentiate), but check the exception below in case it is a bug in a user-supplied map/extra_logprior; falling back to θ₀ and an IQR-based diagonal proposal scale" exception = err
        return collect(float.(θ₀)), fallback
    end
    mode = Optim.minimizer(result)
    if !isfinite(Optim.minimum(result))
        @warn "proposal tuning did not find a finite posterior optimum; falling back to θ₀ and an IQR-based diagonal proposal scale"
        return collect(float.(θ₀)), fallback
    end
    Optim.converged(result) ||
        @warn "proposal tuning's BFGS run did not converge; using its best point anyway"
    H = Symmetric(ForwardDiff.hessian(neg, mode))
    if !all(isfinite, H) || !isposdef(H)
        @warn "the Hessian of the negative log posterior at the mode is not positive definite; keeping the mode but falling back to an IQR-based diagonal proposal scale"
        return mode, fallback
    end
    return mode, inv(H)
end

"""
    discounted_gram_blocks(est1::VARestimate, est2::VARestimate, μ::Real)

Pools two subsamples' sufficient statistics under Baumeister & Hamilton (2019,
AER)'s two-subsample discount weight ``\\mu\\in[0,1]`` (`mu = 0.5` in their
`main_BH_AER.m`): the first subsample's likelihood enters raised to the power
``\\mu`` — equivalently, its stacked data rows are scaled by ``\\sqrt\\mu`` —
so each Gram block picks up exactly one factor of ``\\mu``,

``X'X = \\mu X_1'X_1+X_2'X_2, \\quad X'Y = \\mu X_1'Y_1+X_2'Y_2, \\quad
Y'Y = \\mu Y_1'Y_1+Y_2'Y_2,``

and the effective sample size ``T_{\\text{eff}} = \\mu T_1+T_2`` replaces
``T`` everywhere it appears downstream (``\\bar\\kappa_i =
\\kappa_i+T_{\\text{eff}}/2``, the ``(T_{\\text{eff}}/2)\\ln\\det(A\\tilde\\Omega
A')`` factor, and the ``2/T_{\\text{eff}}`` normalizer) — which is why the
returned `obs` is a `Real` rather than an `Int`. The pooled residual covariance
is the matching weighted average

``\\tilde\\Omega = \\frac{\\mu T_1\\hat\\Sigma_1+T_2\\hat\\Sigma_2}{\\mu T_1+T_2},``

where each `est.Σ` is already *that subsample's own* concentrated residual
matrix ``\\zeta_j/T_j`` — its own OLS projection, not a pooled one — so this
averages the two subsamples' own sums of squares. The prior blocks and
``\\hat S`` are never discounted: ``\\mu`` weights the likelihood alone.

The two subsamples may overlap, and typically should: B&H's overlap by `lags`
rows so the second subsample's first regressands have their initial conditions.
That is why this takes two `VARestimate`s rather than a split index. Requires
the two to describe the same specification (matching `lags`, `vars`, `names`
and `include_constant`) — it is the single place that check is made for the
two-subsample entry points. Returns `(gram, Σ̂, obs)`, exactly the triple the
structural samplers consume, computed once outside every sampling loop.
Internal; feeds `sample_structural(prior, est1, est2; μ)` and
`structural_log_posterior(prior, est1, est2; μ)`.
"""
function discounted_gram_blocks(
    est1::VARestimate{T},
    est2::VARestimate{T},
    μ::Real,
) where {T<:Real}
    @assert isfinite(μ) && 0 <= μ <= 1 "μ must be a finite discount weight in [0, 1] (got $μ); μ = 1 pools the two subsamples, μ = 0 discards the first"
    @assert est1.lags == est2.lags "est1 and est2 must be estimated at the same lag order (got $(est1.lags) and $(est2.lags))"
    @assert est1.vars == est2.vars "est1 and est2 must have the same number of variables (got $(est1.vars) and $(est2.vars))"
    @assert est1.names == est2.names "est1 and est2 must describe the same variables in the same order (got $(est1.names) and $(est2.names))"
    @assert est1.include_constant == est2.include_constant "est1 and est2 must agree on include_constant (got $(est1.include_constant) and $(est2.include_constant)); the two subsamples' regressor blocks would otherwise have different widths"
    μT = convert(T, μ)
    g1 = gram_blocks(est1)
    g2 = gram_blocks(est2)
    obs = μT * est1.obs + est2.obs
    # Normalized weights, not sum-then-divide: at μ = 0 this gives w2 = 1.0 and
    # hence Σ̂ == est2.Σ bit for bit (the two-subsample call then reduces exactly
    # to the single-est call on est2), and at μ = 1 with identical subsamples it
    # gives w1 = w2 = 0.5 exactly.
    w1 = μT * est1.obs / obs
    w2 = est2.obs / obs
    gram = (
        XᵀX = μT .* g1.XᵀX .+ g2.XᵀX,
        XᵀY = μT .* g1.XᵀY .+ g2.XᵀY,
        YᵀY = μT .* g1.YᵀY .+ g2.YᵀY,
    )
    Σ̂ = w1 .* est1.Σ .+ w2 .* est2.Σ
    return (gram = gram, Σ̂ = Σ̂, obs = obs)
end

"""
    sample_structural_blocks(
        prior::HamiltonStructuralPrior,
        gram::NamedTuple,
        Σ̂::AbstractMatrix,
        obs::Real,
        include_constant::Bool;
        ndraws::Int,
        rng::Random.AbstractRNG,
        method::Symbol,
        burn_in::Int,
        oversample::Int,
        θ₀::Union{Nothing,AbstractVector},
        proposal_scale::Union{Nothing,AbstractMatrix},
        ξ::Real,
        proposal_df::Real,
    )

The shared `method` dispatcher behind both public entry points — the
single-sample `sample_structural(prior, est; ...)` and the two-subsample
`sample_structural(prior, est1, est2; μ, ...)` — forwarding to
`sample_structural_sir`, `sample_structural_mh` or `sample_structural_rwmh`
whichever keywords that method consumes. Every keyword is required here: the
defaults live in the public entry points, so the two cannot drift apart.
Internal.
"""
function sample_structural_blocks(
    prior::HamiltonStructuralPrior,
    gram::NamedTuple,
    Σ̂::AbstractMatrix,
    obs::Real,
    include_constant::Bool;
    ndraws::Int,
    rng::Random.AbstractRNG,
    method::Symbol,
    burn_in::Int,
    oversample::Int,
    θ₀::Union{Nothing,AbstractVector},
    proposal_scale::Union{Nothing,AbstractMatrix},
    ξ::Real,
    proposal_df::Real,
)
    @assert method in (:sir, :mh, :rwmh) "method must be :sir, :mh, or :rwmh"
    if method == :sir
        return sample_structural_sir(
            prior,
            gram,
            Σ̂,
            obs,
            include_constant;
            ndraws = ndraws,
            rng = rng,
            oversample = oversample,
        )
    elseif method == :mh
        return sample_structural_mh(
            prior,
            gram,
            Σ̂,
            obs,
            include_constant;
            ndraws = ndraws,
            rng = rng,
            burn_in = burn_in,
        )
    else
        return sample_structural_rwmh(
            prior,
            gram,
            Σ̂,
            obs,
            include_constant;
            ndraws = ndraws,
            rng = rng,
            burn_in = burn_in,
            θ₀ = θ₀,
            proposal_scale = proposal_scale,
            ξ = ξ,
            proposal_df = proposal_df,
        )
    end
end

"""
    sample_structural(
        prior::HamiltonStructuralPrior,
        est::VARestimate;
        ndraws::Int = 1000,
        rng::Random.AbstractRNG = Random.default_rng(),
        method::Symbol = :mh,
        burn_in::Int = ndraws,
        oversample::Int = 10,
        θ₀::Union{Nothing,AbstractVector} = nothing,
        proposal_scale::Union{Nothing,AbstractMatrix} = nothing,
        ξ::Real = 1.0,
        proposal_df::Real = 2.0,
    )

Draws `ndraws` samples of the structural triple ``(A,B,D)`` from the
posterior implied by `prior` and the data summarized by `est`, following
Baumeister & Hamilton (2019, AER)'s marginal posterior of ``A`` (their eq.
12),

``p(A\\mid Y_T) \\propto p(A)\\,\\frac{\\big[\\det(A\\hat\\Omega_TA')\\big]^{T/2}}
{\\prod_i\\big[(2/T)\\tau_i^*(A)\\big]^{\\kappa_i^*}}\\,\\prod_i \\tau_i(A)^{\\kappa_i}.``

Under `method = :sir` and `method = :mh` (the default), candidate
``\\theta``'s (``A``'s free entries, or a `ParametricStructuralPrior`'s
economic parameters) are drawn from `prior.A_prior`'s marginal priors —
which are exactly the ``p(A)`` factor above — so that factor cancels out of
both the importance weight (`:sir`) and the Metropolis-Hastings acceptance
ratio (`:mh`), leaving only the likelihood ratio above
(`log_likelihood_weight`), any `extra_logprior` terms, and any nonlinear
`A_prior.restrictions`; ``B,D\\mid A`` are then drawn from the exact
conjugate posterior, which cancels the same way. This holds regardless of
whether a restriction depends on ``A`` alone (short-run/sign) or on
``(A,B)`` jointly (long-run) — one uniform mechanism handles all three.

`method = :mh` (default) is the more robust of those two: both share
essentially the same per-candidate cost, but `:sir`'s failure mode
(importance-weight collapse — a real risk here, since the posterior's
``T/2`` exponent is steep) can go unnoticed unless the returned `ess`
diagnostic is checked, whereas `:mh`'s failure mode (a low
`acceptance_rate`) is immediately visible. `:sir` is provided for direct
comparison against the paper's own (importance-sampling-based) algorithm.

`method = :rwmh` is the paper's own baseline algorithm
(`sample_structural_rwmh`): a random walk on the collapsed posterior of
``\\theta`` with a fat-tailed multivariate Student-``t`` step
(`proposal_df` degrees of freedom, the paper's 2 by default), so ``p(A)`` no
longer cancels and enters the acceptance ratio explicitly. By default the
proposal is tuned as the paper tunes it — the chain starts at the posterior
mode with scale ``\\xi\\cdot`` (inverse Hessian there), found by BFGS with
`ForwardDiff` gradients through `structural_log_posterior` — and both pieces
can be overridden via `θ₀` and `proposal_scale`. Follow the paper's advice
on `ξ`: adjust it until `acceptance_rate` lands near 30-35%. Prefer `:rwmh`
when `:mh`'s acceptance rate collapses (its whole-``A``-at-once prior draws
are accepted ever more rarely as free parameters multiply) and for full B&H
replication with a `ParametricStructuralPrior`.

Keywords by method — those a method does not consume are silently ignored:
`ndraws` and `rng` (all methods), `oversample` (`:sir`), `burn_in` (`:mh`,
`:rwmh`), `θ₀`/`proposal_scale`/`ξ`/`proposal_df` (`:rwmh`). Returns
`(draws::StructuralDraws, diagnostics::NamedTuple)` — `ess` for `:sir`,
`acceptance_rate` for `:mh`, `acceptance_rate` and the post-burn-in `θ`
draws for `:rwmh`.
"""
function sample_structural(
    prior::HamiltonStructuralPrior,
    est::VARestimate;
    ndraws::Int = 1000,
    rng::Random.AbstractRNG = Random.default_rng(),
    method::Symbol = :mh,
    burn_in::Int = ndraws,
    oversample::Int = 10,
    θ₀::Union{Nothing,AbstractVector} = nothing,
    proposal_scale::Union{Nothing,AbstractMatrix} = nothing,
    ξ::Real = 1.0,
    proposal_df::Real = 2.0,
)
    @assert prior.reduced_form.lags == est.lags && prior.reduced_form.vars == est.vars "prior and est must come from the same model specification (matching lags/vars)"
    @assert method in (:sir, :mh, :rwmh) "method must be :sir, :mh, or :rwmh"
    return sample_structural_blocks(
        prior,
        gram_blocks(est),
        est.Σ,
        est.obs,
        est.include_constant;
        ndraws = ndraws,
        rng = rng,
        method = method,
        burn_in = burn_in,
        oversample = oversample,
        θ₀ = θ₀,
        proposal_scale = proposal_scale,
        ξ = ξ,
        proposal_df = proposal_df,
    )
end

"""
    sample_structural(
        prior::HamiltonStructuralPrior,
        est1::VARestimate,
        est2::VARestimate;
        μ::Real,
        ndraws::Int = 1000,
        rng::Random.AbstractRNG = Random.default_rng(),
        method::Symbol = :mh,
        burn_in::Int = ndraws,
        oversample::Int = 10,
        θ₀::Union{Nothing,AbstractVector} = nothing,
        proposal_scale::Union{Nothing,AbstractMatrix} = nothing,
        ξ::Real = 1.0,
        proposal_df::Real = 2.0,
    )

Draws `ndraws` samples of the structural triple ``(A,B,D)`` from *two*
subsamples, the first one's likelihood discounted to the power ``\\mu`` —
Baumeister & Hamilton (2019, AER)'s two-subsample weight (`mu = 0.5` in their
`main_BH_AER.m`), the device by which an early, less trusted stretch of data
informs the posterior only fractionally. Everything else is identical to
`sample_structural(prior, est; ...)`, whose docstring documents the posterior,
the three `method`s and every other keyword: the discount enters purely through
the sufficient statistics `discounted_gram_blocks` builds, so the sampler and
the posterior math are untouched.

``\\mu`` scales `est1`'s Gram blocks once each (equivalently, its stacked data
rows by ``\\sqrt\\mu``) and the effective sample size becomes
``T_{\\text{eff}} = \\mu T_1+T_2``, which replaces ``T`` throughout the
posterior. `μ = 1` pools the two subsamples with equal weight; `μ = 0` discards
`est1` entirely and reduces *exactly* — bit for bit — to
`sample_structural(prior, est2; ...)`. It is a required keyword with no default
precisely so that pooling weight is always an explicit choice. The prior blocks
and ``\\hat S`` are never discounted; ``\\mu`` weights the likelihood alone.

The two subsamples may overlap, and typically should overlap by `lags` rows, so
that the second subsample's first regressands have their initial conditions —
as B&H's do.

**The one thing no assert can check:** for a faithful B&H replication the prior
itself — both `baumeister_hamilton_prior`'s ``\\hat\\sigma`` scales *and*
`hamilton_structural_prior`'s ``\\hat S`` — must be built from the **first**
subsample's ``Y``, **undiscounted**, as the paper does. `prior` arrives here
already constructed, so a prior built from the wrong data would pass silently.

Returns `(draws::StructuralDraws, diagnostics::NamedTuple)`, exactly as the
single-sample method does. See also `structural_log_posterior(prior, est1,
est2; μ)` for the corresponding marginal posterior closure.
"""
function sample_structural(
    prior::HamiltonStructuralPrior,
    est1::VARestimate,
    est2::VARestimate;
    μ::Real,
    ndraws::Int = 1000,
    rng::Random.AbstractRNG = Random.default_rng(),
    method::Symbol = :mh,
    burn_in::Int = ndraws,
    oversample::Int = 10,
    θ₀::Union{Nothing,AbstractVector} = nothing,
    proposal_scale::Union{Nothing,AbstractMatrix} = nothing,
    ξ::Real = 1.0,
    proposal_df::Real = 2.0,
)
    @assert prior.reduced_form.lags == est1.lags && prior.reduced_form.vars == est1.vars "prior and est1 must come from the same model specification (matching lags/vars)"
    @assert prior.reduced_form.lags == est2.lags && prior.reduced_form.vars == est2.vars "prior and est2 must come from the same model specification (matching lags/vars)"
    @assert method in (:sir, :mh, :rwmh) "method must be :sir, :mh, or :rwmh"
    blocks = discounted_gram_blocks(est1, est2, μ)
    return sample_structural_blocks(
        prior,
        blocks.gram,
        blocks.Σ̂,
        blocks.obs,
        est2.include_constant;
        ndraws = ndraws,
        rng = rng,
        method = method,
        burn_in = burn_in,
        oversample = oversample,
        θ₀ = θ₀,
        proposal_scale = proposal_scale,
        ξ = ξ,
        proposal_df = proposal_df,
    )
end

"""
    structural_log_posterior(
        prior::HamiltonStructuralPrior,
        est1::VARestimate,
        est2::VARestimate;
        μ::Real,
    )

Two-subsample version of `structural_log_posterior(prior, est)`: the same
deterministic closure ``\\theta\\to\\ln p(\\theta\\mid Y_T)`` (up to a
constant), with the first subsample's likelihood discounted to the power
``\\mu`` — Baumeister & Hamilton (2019, AER)'s two-subsample weight (`mu = 0.5`
in their `main_BH_AER.m`). ``\\mu`` scales `est1`'s Gram blocks once each
(equivalently, its stacked data rows by ``\\sqrt\\mu``, see
`discounted_gram_blocks`) and the effective sample size becomes
``T_{\\text{eff}} = \\mu T_1+T_2``, replacing ``T`` throughout the posterior.
`μ = 1` pools the two subsamples with equal weight; `μ = 0` discards `est1` and
reduces *exactly* to `structural_log_posterior(prior, est2)`. It is a required
keyword with no default so that pooling weight is always an explicit choice.
The prior blocks and ``\\hat S`` are never discounted; ``\\mu`` weights the
likelihood alone. The two subsamples may overlap, and typically should overlap
by `lags` rows so the second subsample's first regressands have their initial
conditions — as B&H's do.

**The one thing no assert can check:** for a faithful B&H replication the prior
itself — both `baumeister_hamilton_prior`'s ``\\hat\\sigma`` scales *and*
`hamilton_structural_prior`'s ``\\hat S`` — must be built from the **first**
subsample's ``Y``, **undiscounted**, as the paper does. `prior` arrives here
already constructed, so a prior built from the wrong data would pass silently.

The closure is `ForwardDiff`-differentiable exactly as the single-sample one
is, so it can be used for by-hand tuning of `sample_structural(prior, est1,
est2; μ, method = :rwmh)`.
"""
function structural_log_posterior(
    prior::HamiltonStructuralPrior,
    est1::VARestimate,
    est2::VARestimate;
    μ::Real,
)
    @assert prior.reduced_form.lags == est1.lags && prior.reduced_form.vars == est1.vars "prior and est1 must come from the same model specification (matching lags/vars)"
    @assert prior.reduced_form.lags == est2.lags && prior.reduced_form.vars == est2.vars "prior and est2 must come from the same model specification (matching lags/vars)"
    blocks = discounted_gram_blocks(est1, est2, μ)
    return structural_log_posterior(prior, blocks.gram, blocks.Σ̂, blocks.obs)
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
