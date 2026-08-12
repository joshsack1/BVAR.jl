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
    hamilton_structural_prior(
        reduced_form::BaumeisterHamiltonPrior,
        A_prior::StructuralPrior,
        Y::AbstractMatrix,
    )

Builds a `HamiltonStructuralPrior` pairing the reduced-form per-equation
``B,D\\mid A`` prior `reduced_form` (from `baumeister_hamilton_prior`) with
the structural prior `A_prior` on ``A`` (from `structural_prior`), computing
``\\hat S`` (`ar_residual_covariance`) from the same data `Y` and lag order
`reduced_form.lags` used to build `reduced_form`.
"""
function hamilton_structural_prior(
    reduced_form::BaumeisterHamiltonPrior{T},
    A_prior::StructuralPrior{T},
    Y::AbstractMatrix{T},
) where {T<:Real}
    @assert reduced_form.vars == A_prior.vars "reduced_form and A_prior must have the same number of variables"
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

function draw_A(prior::StructuralPrior{T}, rng::Random.AbstractRNG) where {T<:Real}
    values =
        Dict{Tuple{Int,Int},T}(idx => T(rand(rng, dist)) for (idx, dist) in prior.component)
    return assemble_A(prior, values)
end

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
draw reuses it directly rather than recomputing it. Internal; called by
`draw_candidate`.
"""
function log_likelihood_weight(
    A::AbstractMatrix{T},
    m::Vector{<:AbstractVector{T}},
    M::Vector{<:AbstractMatrix{T}},
    κ::Vector{T},
    Ŝ::AbstractMatrix{T},
    gram::NamedTuple,
    Σ̂::AbstractMatrix{T},
    obs::Int,
) where {T<:Real}
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
    draw_candidate(
        prior::HamiltonStructuralPrior,
        gram::NamedTuple,
        Σ̂::AbstractMatrix,
        obs::Int,
        rng::Random.AbstractRNG,
    )

Draws one candidate structural triple ``(A,B,D)``: ``A`` from
`prior.A_prior`'s component priors, then ``(B,D)\\mid A`` from the exact
conjugate posterior (`log_likelihood_weight`), then adds every joint
restriction in `prior.A_prior.restrictions`. Returns `(A, B, d, logw)`, `d`
the diagonal of `D`. Internal; the shared draw step of both
`sample_structural_sir` and `sample_structural_mh`.
"""
function draw_candidate(
    prior::HamiltonStructuralPrior{T},
    gram::NamedTuple,
    Σ̂::AbstractMatrix{T},
    obs::Int,
    rng::Random.AbstractRNG,
) where {T<:Real}
    rf = prior.reduced_form
    A = draw_A(prior.A_prior, rng)
    logw, posts = log_likelihood_weight(A, rf.m, rf.M, rf.κ, prior.Ŝ, gram, Σ̂, obs)
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
