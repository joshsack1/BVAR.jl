@testset "dummy_minnesota: shapes and weights" begin
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    n = length(end_vec_p)
    λ1, λ3 = 0.3, 1.2
    Yd, Xd = BayesianVectorAutoregressions.dummy_minnesota(
        Y_endog,
        lags_p,
        true;
        λ1 = λ1,
        λ3 = λ3,
    )
    σ_ar = sqrt.(BayesianVectorAutoregressions.ar_residual_variances(Y_endog, lags_p))
    @test size(Yd) == (n * lags_p, n)
    @test size(Xd) == (n * lags_p, n * lags_p + 1)
    for ℓ in 1:lags_p, j in 1:n
        row = (ℓ - 1) * n + j
        weight = σ_ar[j] * ℓ^λ3 / λ1
        @test Yd[row, j] ≈ weight
        @test Xd[row, 1 + (ℓ - 1) * n + j] ≈ weight
        # Off-target entries are exactly zero.
        others = trues(n)
        others[j] = false
        @test all(Yd[row, others] .== 0.0)
    end
end

@testset "dummy_sum_of_coefficients: one row per variable, own lags equal" begin
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    n = length(end_vec_p)
    λ_soc = 0.7
    Yd, Xd = BayesianVectorAutoregressions.dummy_sum_of_coefficients(
        Y_endog,
        lags_p,
        true;
        λ_soc = λ_soc,
    )
    ȳ = vec(sum(Y_endog[1:lags_p, :]; dims = 1) ./ lags_p)
    @test size(Yd) == (n, n)
    @test size(Xd) == (n, n * lags_p + 1)
    for i in 1:n
        weight = ȳ[i] / λ_soc
        @test Yd[i, i] ≈ weight
        @test Xd[i, 1] == 0.0 # never touches the constant
        for ℓ in 1:lags_p
            @test Xd[i, 1 + (ℓ - 1) * n + i] ≈ weight
        end
    end
end

@testset "dummy_initial_observation: single row, constant identified" begin
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    n = length(end_vec_p)
    λ_dio = 0.9
    Yd, Xd = BayesianVectorAutoregressions.dummy_initial_observation(
        Y_endog,
        lags_p,
        true;
        λ_dio = λ_dio,
    )
    ȳ = vec(sum(Y_endog[1:lags_p, :]; dims = 1) ./ lags_p)
    @test size(Yd) == (1, n)
    @test size(Xd) == (1, n * lags_p + 1)
    @test vec(Yd) ≈ ȳ ./ λ_dio
    @test Xd[1, 1] ≈ 1 / λ_dio
    for ℓ in 1:lags_p, j in 1:n
        @test Xd[1, 1 + (ℓ - 1) * n + j] ≈ ȳ[j] / λ_dio
    end
end

@testset "dummy_long_run with H = I reproduces dummy_sum_of_coefficients" begin
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    n = length(end_vec_p)
    λ = 0.7
    Yd_soc, Xd_soc = BayesianVectorAutoregressions.dummy_sum_of_coefficients(
        Y_endog,
        lags_p,
        true;
        λ_soc = λ,
    )
    Yd_lr, Xd_lr = BayesianVectorAutoregressions.dummy_long_run(
        Y_endog,
        lags_p,
        true,
        Matrix{Float64}(I, n, n);
        λ_lr = λ,
    )
    @test Yd_lr ≈ Yd_soc
    @test Xd_lr ≈ Xd_soc
end

@testset "Composed NormalWishartPrior via dummy_components agrees with a manual stack" begin
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    λ = (λ1 = 0.2, λ3 = 1.0, λ4 = 1e5, λ_soc = 1.0, λ_dio = 1.0, λ_lr = 1.0)
    prior = BayesianVectorAutoregressions.normal_wishart_prior(
        Y_endog,
        lags_p,
        end_vec_p,
        true;
        dummy_components = [:minnesota, :sum_of_coefficients, :dummy_initial_obs],
        λ = λ,
    )
    Yd1, Xd1 = BayesianVectorAutoregressions.dummy_minnesota(
        Y_endog,
        lags_p,
        true;
        λ1 = λ.λ1,
        λ3 = λ.λ3,
    )
    Yd2, Xd2 = BayesianVectorAutoregressions.dummy_sum_of_coefficients(
        Y_endog,
        lags_p,
        true;
        λ_soc = λ.λ_soc,
    )
    Yd3, Xd3 = BayesianVectorAutoregressions.dummy_initial_observation(
        Y_endog,
        lags_p,
        true;
        λ_dio = λ.λ_dio,
    )
    moments =
        BayesianVectorAutoregressions.dummy_gram(vcat(Yd1, Yd2, Yd3), vcat(Xd1, Xd2, Xd3))
    @test prior.β0 ≈ moments.β0
    @test prior.Ω0 ≈ moments.Ω0
    @test prior.S0 ≈ moments.S0
    @test prior.ν0 == moments.ν0
    @test isposdef(Symmetric(prior.S0))
    @test isposdef(Symmetric(prior.Ω0))
end
