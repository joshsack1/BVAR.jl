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
end
