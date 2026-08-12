# Per-family closed-form log marginal likelihoods, and a hand-rolled
# golden-section coordinate-ascent search used to maximize them over each
# family's (small) hyperparameter vector.

"""
    log_multivariate_gamma(n::Int, a::Real)

The log of the multivariate gamma function

``\\ln \\Gamma_n(a) = \\frac{n(n-1)}{4}\\ln\\pi + \\sum_{i=1}^n \\ln\\Gamma\\!\\left(a+\\frac{1-i}{2}\\right),``

used by the natural-conjugate Normal-Wishart marginal likelihood below.
"""
function log_multivariate_gamma(n::Int, a::Real)
    return (n * (n - 1) / 4) * log(π) + sum(loggamma(a + (1 - i) / 2) for i in 1:n)
end

"""
    log_marginal_likelihood(
        prior::NormalWishartPrior,
        XᵀX::AbstractMatrix,
        XᵀY::AbstractMatrix,
        YᵀY::AbstractMatrix,
        obs::Int,
    )

Closed-form log marginal likelihood of the data under a natural-conjugate
Normal-Wishart prior,

``\\ln p(Y) = -\\frac{Tn}{2}\\ln\\pi + \\frac{n}{2}\\left(\\ln|\\bar\\Omega|-\\ln|\\Omega_0|\\right)
+ \\frac{\\nu_0}{2}\\ln|S_0| - \\frac{\\bar\\nu}{2}\\ln|\\bar S|
+ \\ln\\Gamma_n\\!\\left(\\frac{\\bar\\nu}{2}\\right) - \\ln\\Gamma_n\\!\\left(\\frac{\\nu_0}{2}\\right),``

where ``\\bar\\Omega = (\\Omega_0^{-1}+X'X)^{-1}``,
``\\bar\\beta = \\bar\\Omega(\\Omega_0^{-1}\\beta_0+X'Y)``,
``\\bar S = S_0+Y'Y+\\beta_0'\\Omega_0^{-1}\\beta_0-\\bar\\beta'(\\Omega_0^{-1}+X'X)\\bar\\beta``, and
``\\bar\\nu = \\nu_0+T``. Used by `optimize_hyperparameters` to score
candidate hyperparameters for every family built on `NormalWishartPrior`.
"""
function log_marginal_likelihood(
    prior::NormalWishartPrior,
    XᵀX::AbstractMatrix,
    XᵀY::AbstractMatrix,
    YᵀY::AbstractMatrix,
    obs::Int,
)
    @unpack Ω0, S0, ν0 = prior
    n = size(YᵀY, 1)
    post = normal_wishart_posterior(prior, XᵀX, XᵀY, YᵀY, obs)
    return -(obs * n / 2) * log(π) +
           (n / 2) * (logdet(post.Ω̄) - logdet(Ω0)) +
           (ν0 / 2) * logdet(S0) - (post.ν̄ / 2) * logdet(post.S̄) +
           log_multivariate_gamma(n, post.ν̄ / 2) - log_multivariate_gamma(n, ν0 / 2)
end

"""
    log_marginal_likelihood(
        prior::MinnesotaPrior,
        XᵀX::AbstractMatrix,
        XᵀY::AbstractMatrix,
        YᵀY::AbstractMatrix,
        obs::Int,
    )

*Profile* log marginal likelihood for the original Minnesota prior, treating
each equation's residual variance as fixed/known at its univariate AR
estimate ``\\hat\\sigma_i^2`` rather than integrated out (the Minnesota prior
never places a proper prior on ``\\Sigma``, so there is no fully Bayesian
evidence to compute). Equations are scored independently, each via the
marginal distribution of a Bayesian linear regression with known variance,

``y_i \\sim N(X\\beta_{0,i},\\ \\hat\\sigma_i^2 I + X\\Omega_{0,i}X'),``

evaluated with the Woodbury/matrix-determinant identities so that only the
``k\\times k`` Gram blocks are needed. Do not treat the returned value as a
proper Bayesian model-evidence comparable across families with a genuine
``\\Sigma`` prior.
"""
function log_marginal_likelihood(
    prior::MinnesotaPrior,
    XᵀX::AbstractMatrix,
    XᵀY::AbstractMatrix,
    YᵀY::AbstractMatrix,
    obs::Int,
)
    @unpack β0, Ω0, σ_ar = prior
    n = size(β0, 2)
    total = zero(eltype(β0))
    for i in 1:n
        σᵢ² = σ_ar[i]^2
        β0ᵢ = β0[:, i]
        Ω0ᵢ_inv = Diagonal(1 ./ Ω0[:, i])
        P = Ω0ᵢ_inv + XᵀX ./ σᵢ²
        Xᵀeᵢ = XᵀY[:, i] - XᵀX * β0ᵢ
        eᵀeᵢ = YᵀY[i, i] - 2 * β0ᵢ' * XᵀY[:, i] + β0ᵢ' * XᵀX * β0ᵢ
        quad = eᵀeᵢ / σᵢ² - (Xᵀeᵢ' * (P \ Xᵀeᵢ)) / σᵢ²^2
        logdetΣᵢ = obs * log(σᵢ²) + logdet(Diagonal(Ω0[:, i])) + logdet(P)
        total += -(obs / 2) * log(2π) - logdetΣᵢ / 2 - quad / 2
    end
    return total
end

"""
    equation_log_marginal_likelihood(
        m::AbstractVector,
        M::AbstractMatrix,
        κ::Real,
        τ::Real,
        XᵀX::AbstractMatrix,
        Xᵀy::AbstractVector,
        yᵀy::Real,
        obs::Int,
    )

Closed-form log marginal likelihood of a single equation ``y=Xb+\\varepsilon``,
``\\varepsilon \\sim N(0,dI)``, ``b\\mid d \\sim N(m,dM)``, ``d^{-1}\\sim
\\text{Gamma}(\\kappa,\\tau)``:

``\\ln p(y) = -\\frac{T}{2}\\ln(2\\pi) + \\frac12\\left(\\ln|\\bar M|-\\ln|M|\\right)
+ \\kappa\\ln\\tau - \\bar\\kappa\\ln\\bar\\tau + \\ln\\Gamma(\\bar\\kappa) - \\ln\\Gamma(\\kappa),``

where ``\\bar M=(M^{-1}+X'X)^{-1}``, ``\\bar b=\\bar M(M^{-1}m+X'y)``,
``\\bar\\kappa=\\kappa+T/2``, and ``\\bar\\tau=\\tau+\\frac12\\left(y'y+m'M^{-1}m-\\bar
b'(M^{-1}+X'X)\\bar b\\right)``. Shared by `AsymmetricConjugatePrior` and
`BaumeisterHamiltonPrior`, which are both independent per-equation
Normal-Gamma priors.
"""
function equation_log_marginal_likelihood(
    m::AbstractVector,
    M::AbstractMatrix,
    κ::Real,
    τ::Real,
    XᵀX::AbstractMatrix,
    Xᵀy::AbstractVector,
    yᵀy::Real,
    obs::Int,
)
    post = equation_normal_gamma_posterior(m, M, κ, τ, XᵀX, Xᵀy, yᵀy, obs)
    return -(obs / 2) * log(2π) + (logdet(post.M̄) - logdet(M)) / 2 + κ * log(τ) -
           post.κ̄ * log(post.τ̄) + loggamma(post.κ̄) - loggamma(κ)
end

"""
    log_marginal_likelihood(
        prior::AsymmetricConjugatePrior,
        XᵀX::AbstractMatrix,
        XᵀY::AbstractMatrix,
        YᵀY::AbstractMatrix,
        obs::Int,
    )

Sum of the per-equation `equation_log_marginal_likelihood`, since the
asymmetric conjugate prior treats every equation as an independent
Normal-Gamma model.
"""
function log_marginal_likelihood(
    prior::AsymmetricConjugatePrior,
    XᵀX::AbstractMatrix,
    XᵀY::AbstractMatrix,
    YᵀY::AbstractMatrix,
    obs::Int,
)
    @unpack β0, Ω0, κ, τ = prior
    n = length(β0)
    return sum(
        equation_log_marginal_likelihood(
            β0[i],
            Ω0[i],
            κ[i],
            τ[i],
            XᵀX,
            XᵀY[:, i],
            YᵀY[i, i],
            obs,
        ) for i in 1:n
    )
end

"""
    log_marginal_likelihood(
        prior::BaumeisterHamiltonPrior,
        XᵀX::AbstractMatrix,
        XᵀY::AbstractMatrix,
        YᵀY::AbstractMatrix,
        obs::Int,
    )

Sum of the per-equation `equation_log_marginal_likelihood`, since the
Baumeister-Hamilton reduced-form prior also treats every equation as an
independent Normal-Gamma model.
"""
function log_marginal_likelihood(
    prior::BaumeisterHamiltonPrior,
    XᵀX::AbstractMatrix,
    XᵀY::AbstractMatrix,
    YᵀY::AbstractMatrix,
    obs::Int,
)
    @unpack m, M, κ, τ = prior
    n = length(m)
    return sum(
        equation_log_marginal_likelihood(
            m[i],
            M[i],
            κ[i],
            τ[i],
            XᵀX,
            XᵀY[:, i],
            YᵀY[i, i],
            obs,
        ) for i in 1:n
    )
end

"""
    golden_section_ascent(f, lo::Real, hi::Real; tol = 1e-4)

Hand-rolled golden-section search for the maximizer of a unimodal scalar
function `f` on `[lo, hi]` (Press et al., *Numerical Recipes*). Returns
`(x★, f(x★))`; used by `coordinate_ascent` to maximize the closed-form log
marginal likelihoods above without adding a general-purpose optimization
dependency.
"""
function golden_section_ascent(f, lo::Real, hi::Real; tol = 1e-4)
    φ = (sqrt(5) - 1) / 2
    a, b = float(lo), float(hi)
    c = b - φ * (b - a)
    d = a + φ * (b - a)
    fc, fd = f(c), f(d)
    while (b - a) > tol
        if fc > fd
            b, d, fd = d, c, fc
            c = b - φ * (b - a)
            fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + φ * (b - a)
            fd = f(d)
        end
    end
    x = (a + b) / 2
    return x, f(x)
end

"""
    coordinate_ascent(f, bounds::Vector{<:Tuple}; iters = 5)

Maximizes a scalar function `f` of `length(bounds)` hyperparameters by
repeatedly sweeping `golden_section_ascent` over one coordinate at a time,
holding the others fixed at their current best value, for `iters` full
sweeps. Suited to the low-dimensional (1-4 hyperparameter) marginal
likelihoods in this module; returns the optimizing vector.
"""
function coordinate_ascent(f, bounds::Vector{<:Tuple}; iters = 5)
    x = [(lo + hi) / 2 for (lo, hi) in bounds]
    for _ in 1:iters, j in eachindex(x)
        lo, hi = bounds[j]
        g(xj) = f([k == j ? xj : x[k] for k in eachindex(x)])
        x[j], _ = golden_section_ascent(g, lo, hi)
    end
    return x
end

"""
    optimize_hyperparameters(
        family::Symbol,
        Y::AbstractMatrix,
        lags::Int,
        names::Vector{Symbol},
        include_constant::Bool,
        XᵀX::AbstractMatrix,
        XᵀY::AbstractMatrix,
        YᵀY::AbstractMatrix,
        obs::Int;
        dummy_components::Vector{Symbol} = Symbol[],
        H = nothing,
    )

Maximizes the family-specific `log_marginal_likelihood` over that family's
own (small) hyperparameter vector via `coordinate_ascent`, then returns the
prior built at the optimizing hyperparameters. This is the routine
`build_prior` calls when `hyperparameter_method = :marginal_likelihood` (the
default); it is not implemented for `family = :independent_niw`, which
`build_prior` rejects before reaching here.
"""
function optimize_hyperparameters(
    family::Symbol,
    Y::AbstractMatrix,
    lags::Int,
    names::Vector{Symbol},
    include_constant::Bool,
    XᵀX::AbstractMatrix,
    XᵀY::AbstractMatrix,
    YᵀY::AbstractMatrix,
    obs::Int;
    dummy_components::Vector{Symbol} = Symbol[],
    H = nothing,
)
    if family == :minnesota
        bounds = [(0.01, 2.0), (0.01, 2.0), (0.1, 3.0)]
        build_minnesota(x) = minnesota_prior(
            Y,
            lags,
            names,
            include_constant;
            λ = (λ1 = x[1], λ2 = x[2], λ3 = x[3], λ4 = 1e5),
        )
        obj_minnesota(x) = log_marginal_likelihood(build_minnesota(x), XᵀX, XᵀY, YᵀY, obs)
        return build_minnesota(coordinate_ascent(obj_minnesota, bounds))
    elseif family == :normal_wishart && isempty(dummy_components)
        bounds = [(0.01, 2.0), (0.1, 3.0)]
        build_nw_direct(x) = normal_wishart_prior(
            Y,
            lags,
            names,
            include_constant;
            λ = (λ1 = x[1], λ3 = x[2], λ4 = 1e5, λ_soc = 1.0, λ_dio = 1.0, λ_lr = 1.0),
        )
        obj_nw_direct(x) = log_marginal_likelihood(build_nw_direct(x), XᵀX, XᵀY, YᵀY, obs)
        return build_nw_direct(coordinate_ascent(obj_nw_direct, bounds))
    elseif family == :normal_wishart
        component_key = Dict(
            :minnesota => :λ1,
            :sum_of_coefficients => :λ_soc,
            :dummy_initial_obs => :λ_dio,
            :long_run => :λ_lr,
        )
        bounds = fill((0.05, 5.0), length(dummy_components))
        function build_nw_dummy(x)
            weights =
                Dict(component_key[c] => x[idx] for (idx, c) in enumerate(dummy_components))
            λ = (
                λ1 = get(weights, :λ1, 0.2),
                λ3 = 1.0,
                λ4 = 1e5,
                λ_soc = get(weights, :λ_soc, 1.0),
                λ_dio = get(weights, :λ_dio, 1.0),
                λ_lr = get(weights, :λ_lr, 1.0),
            )
            return normal_wishart_prior(
                Y,
                lags,
                names,
                include_constant;
                dummy_components = dummy_components,
                λ = λ,
                H = H,
            )
        end
        obj_nw_dummy(x) = log_marginal_likelihood(build_nw_dummy(x), XᵀX, XᵀY, YᵀY, obs)
        return build_nw_dummy(coordinate_ascent(obj_nw_dummy, bounds))
    elseif family == :asymmetric_conjugate
        bounds = [(0.01, 2.0), (0.01, 2.0), (0.1, 3.0)]
        build_asymmetric(x) = asymmetric_conjugate_prior(
            Y,
            lags,
            names,
            include_constant;
            λ = (λ1 = x[1], λ2 = x[2], λ3 = x[3], λ4 = 1e5),
        )
        obj_asymmetric(x) = log_marginal_likelihood(build_asymmetric(x), XᵀX, XᵀY, YᵀY, obs)
        return build_asymmetric(coordinate_ascent(obj_asymmetric, bounds))
    elseif family == :hamilton_baumeister
        bounds = [(0.05, 3.0), (0.1, 3.0)]
        build_hamilton(x) = baumeister_hamilton_prior(
            Y,
            lags,
            names,
            include_constant;
            λ0 = x[1],
            λ1 = x[2],
        )
        obj_hamilton(x) = log_marginal_likelihood(build_hamilton(x), XᵀX, XᵀY, YᵀY, obs)
        return build_hamilton(coordinate_ascent(obj_hamilton, bounds))
    else
        error("marginal-likelihood tuning is not implemented for family = $family")
    end
end
