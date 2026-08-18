@testset "BaumeisterHamiltonPrior: reduced-form independent Normal-Gamma structure" begin
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    n = length(end_vec_p)
    k = n * lags_p + 1
    λ0, λ1, λ3, κ0 = 0.4, 0.8, 50.0, 3.0
    prior = BayesianVectorAutoregressions.baumeister_hamilton_prior(
        Y_endog,
        lags_p,
        end_vec_p,
        true;
        λ0 = λ0,
        λ1 = λ1,
        λ3 = λ3,
        κ0 = κ0,
    )
    σ_ar = sqrt.(BayesianVectorAutoregressions.ar_residual_variances(Y_endog, lags_p))

    @test prior isa BaumeisterHamiltonPrior
    @test !prior.structural
    @test length(prior.m) == n
    @test length(prior.M) == n
    @test all(length(mi) == k for mi in prior.m)

    # M is shared (equation-independent), per the paper's own assumption.
    for i in 2:n
        @test prior.M[i] == prior.M[1]
    end
    for j in 1:n
        @test prior.M[1][1 + j, 1 + j] ≈ λ0^2 / (1^(2λ1) * σ_ar[j]^2)
    end
    @test prior.M[1][1, 1] ≈ (λ0 * λ3)^2

    # τ_i = κ_i σ_i^2, so the prior mean of d_ii^{-1} is 1/σ_i^2.
    @test all(prior.κ .== κ0)
    for i in 1:n
        @test prior.τ[i] ≈ κ0 * σ_ar[i]^2
        @test prior.κ[i] / prior.τ[i] ≈ 1 / σ_ar[i]^2
    end

    # Random-walk mean: own first lag = 1.
    for i in 1:n
        @test prior.m[i][1 + i] == 1.0
    end
end

@testset "BaumeisterHamiltonPrior via build_prior matches the direct call" begin
    hp = (λ0 = 0.6, λ1 = 1.2, λ3 = 80.0, κ0 = 2.5, random_walk = true)
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    direct = BayesianVectorAutoregressions.baumeister_hamilton_prior(
        Y_endog,
        lags_p,
        end_vec_p,
        true;
        hp...,
    )
    via_build = build_prior(
        df_p,
        end_vec_p,
        est_p,
        :hamilton_baumeister;
        hyperparameter_method = :fixed,
        hyperparameters = hp,
    )
    @test via_build.m == direct.m
    @test via_build.M == direct.M
    @test via_build.κ == direct.κ
    @test via_build.τ == direct.τ

    # The prior-mean keywords thread through build_prior's splat the same way.
    n = length(end_vec_p)
    k = n * lags_p + 1
    m_custom = [zeros(k) for _ in 1:n]
    m_custom[1][2] = 0.1
    hp_m = (λ0 = 0.6, λ1 = 1.2, λ3 = 80.0, κ0 = 2.5, random_walk = false, m = m_custom)
    direct_m = BayesianVectorAutoregressions.baumeister_hamilton_prior(
        Y_endog,
        lags_p,
        end_vec_p,
        true;
        hp_m...,
    )
    via_build_m = build_prior(
        df_p,
        end_vec_p,
        est_p,
        :hamilton_baumeister;
        hyperparameter_method = :fixed,
        hyperparameters = hp_m,
    )
    @test via_build_m.m == direct_m.m == m_custom

    η_rw = zeros(n, k)
    for i in 1:n
        η_rw[i, 1 + i] = 1.0
    end
    hp_η = (λ0 = 0.6, λ1 = 1.2, λ3 = 80.0, κ0 = 2.5, random_walk = false, η = η_rw)
    direct_η = BayesianVectorAutoregressions.baumeister_hamilton_prior(
        Y_endog,
        lags_p,
        end_vec_p,
        true;
        hp_η...,
    )
    via_build_η = build_prior(
        df_p,
        end_vec_p,
        est_p,
        :hamilton_baumeister;
        hyperparameter_method = :fixed,
        hyperparameters = hp_η,
    )
    @test via_build_η.η == direct_η.η == η_rw
    @test via_build_η.m == direct_η.m
end

@testset "Custom constant prior mean m" begin
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    n = length(end_vec_p)
    k = n * lags_p + 1

    # BH-baseline-style mean: zeros except two entries on a first-lag
    # coefficient (their main_BH_AER.m sets +0.1 and -0.1 on the first lag of
    # the real oil price in the supply and demand equations).
    m_bh = [zeros(k) for _ in 1:n]
    m_bh[1][2] = 0.1
    m_bh[2][2] = -0.1
    prior_m = BayesianVectorAutoregressions.baumeister_hamilton_prior(
        Y_endog,
        lags_p,
        end_vec_p,
        true;
        random_walk = false,
        m = m_bh,
    )
    @test prior_m.m == m_bh
    @test prior_m.η === nothing
    @test !prior_m.structural
    # The builder copies: mutating the caller's array must not reach the prior.
    m_bh[1][2] = 99.0
    @test prior_m.m[1][2] == 0.1
    m_bh[1][2] = 0.1

    # An explicit zero mean is field-identical to the built-in zero mean.
    prior_zero_kw = BayesianVectorAutoregressions.baumeister_hamilton_prior(
        Y_endog,
        lags_p,
        end_vec_p,
        true;
        random_walk = false,
        m = [zeros(k) for _ in 1:n],
    )
    prior_zero = BayesianVectorAutoregressions.baumeister_hamilton_prior(
        Y_endog,
        lags_p,
        end_vec_p,
        true;
        random_walk = false,
    )
    @test prior_zero_kw.m == prior_zero.m
    @test prior_zero_kw.M == prior_zero.M
    @test prior_zero_kw.κ == prior_zero.κ
    @test prior_zero_kw.τ == prior_zero.τ

    # A nonzero m shifts the equation posterior mean by exactly M̄·M⁻¹·m and
    # leaves M̄ and κ̄ untouched (linearity of the Normal-Gamma update in m).
    gram = BayesianVectorAutoregressions.gram_blocks(est_p)
    i = 1
    post_m = BayesianVectorAutoregressions.equation_normal_gamma_posterior(
        prior_m.m[i],
        prior_m.M[i],
        prior_m.κ[i],
        prior_m.τ[i],
        gram.XᵀX,
        gram.XᵀY[:, i],
        gram.YᵀY[i, i],
        est_p.obs,
    )
    post_0 = BayesianVectorAutoregressions.equation_normal_gamma_posterior(
        zeros(k),
        prior_m.M[i],
        prior_m.κ[i],
        prior_m.τ[i],
        gram.XᵀX,
        gram.XᵀY[:, i],
        gram.YᵀY[i, i],
        est_p.obs,
    )
    @test post_m.b̄ - post_0.b̄ ≈ post_m.M̄ * (inv(prior_m.M[i]) * prior_m.m[i])
    @test post_m.M̄ == post_0.M̄
    @test post_m.κ̄ == post_0.κ̄
end

@testset "A-dependent prior mean η" begin
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    n = length(end_vec_p)
    k = n * lags_p + 1

    # η with unit rows on each equation's own first lag is the structural
    # random-walk prior; at A = I it must reproduce random_walk = true's m
    # exactly (η'eᵢ = row i of η).
    η_rw = zeros(n, k)
    for i in 1:n
        η_rw[i, 1 + i] = 1.0
    end
    prior_η = BayesianVectorAutoregressions.baumeister_hamilton_prior(
        Y_endog,
        lags_p,
        end_vec_p,
        true;
        random_walk = false,
        η = η_rw,
    )
    prior_rw = BayesianVectorAutoregressions.baumeister_hamilton_prior(
        Y_endog,
        lags_p,
        end_vec_p,
        true,
    )
    @test prior_η.m == prior_rw.m
    @test prior_η.structural
    @test !prior_rw.structural
    @test prior_η.η == η_rw
    # The builder copies η: caller-side mutation must not reach the prior.
    η_rw[1, 2] = 99.0
    @test prior_η.η[1, 2] == 1.0
    η_rw[1, 2] = 1.0
end

@testset "Prior-mean guardrails" begin
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    n = length(end_vec_p)
    k = n * lags_p + 1
    m_ok = [zeros(k) for _ in 1:n]
    η_ok = zeros(n, k)

    build(; kwargs...) = BayesianVectorAutoregressions.baumeister_hamilton_prior(
        Y_endog,
        lags_p,
        end_vec_p,
        true;
        kwargs...,
    )

    # random_walk defaults to true, so a bare custom mean must be rejected.
    @test_throws AssertionError build(m = m_ok)
    @test_throws AssertionError build(η = η_ok)
    @test_throws AssertionError build(random_walk = false, m = m_ok, η = η_ok)
    @test_throws AssertionError build(random_walk = false, m = m_ok[1:(n - 1)])
    @test_throws AssertionError build(
        random_walk = false,
        m = [zeros(k - 1) for _ in 1:n],
    )
    m_nan = [zeros(k) for _ in 1:n]
    m_nan[1][1] = NaN
    @test_throws AssertionError build(random_walk = false, m = m_nan)
    @test_throws AssertionError build(random_walk = false, η = zeros(n, k - 1))
    @test_throws AssertionError build(random_walk = false, η = zeros(n + 1, k))
    η_inf = zeros(n, k)
    η_inf[1, 1] = Inf
    @test_throws AssertionError build(random_walk = false, η = η_inf)
end
