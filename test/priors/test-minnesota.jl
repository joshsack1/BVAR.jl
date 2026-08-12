@testset "MinnesotaPrior: shapes and random-walk mean" begin
    Y_endog = BVAR.get_endogenous(df_p, end_vec_p)
    λ = (λ1 = 0.3, λ2 = 0.4, λ3 = 1.5, λ4 = 1e4)
    prior = BVAR.minnesota_prior(Y_endog, lags_p, end_vec_p, true; λ = λ)
    n = length(end_vec_p)
    k = n * lags_p + 1

    @test prior isa MinnesotaPrior
    @test size(prior.β0) == (k, n)
    @test size(prior.Ω0) == (k, n)
    @test length(prior.σ_ar) == n
    @test all(isfinite, prior.σ_ar) && all(>(0), prior.σ_ar)

    # Random-walk prior mean: own first lag = 1, everything else 0.
    for i in 1:n
        @test prior.β0[1 + i, i] == 1.0
        others = trues(k)
        others[1 + i] = false
        @test all(prior.β0[others, i] .== 0.0)
    end

    # Constant-term prior variance matches (σ_i λ4)^2.
    for i in 1:n
        @test prior.Ω0[1, i] ≈ (prior.σ_ar[i] * λ.λ4)^2
    end

    # Own-lag variance matches (λ1/ℓ^λ3)^2 and decays with the lag.
    for j in 1:n
        own_lag1 = (λ.λ1 / 1^λ.λ3)^2
        own_lag2 = (λ.λ1 / 2^λ.λ3)^2
        @test prior.Ω0[1 + j, j] ≈ own_lag1
        @test prior.Ω0[1 + n + j, j] ≈ own_lag2
        @test own_lag2 < own_lag1
    end

    # Cross-lag variance matches (λ1λ2/ℓ^λ3)^2 (σ_i/σ_j)^2 and is tighter than
    # own-lag variance for λ2 < 1.
    for i in 1:n, j in 1:n
        i == j && continue
        cross_lag1 = (λ.λ1 * λ.λ2 / 1^λ.λ3)^2 * (prior.σ_ar[i]^2 / prior.σ_ar[j]^2)
        @test prior.Ω0[1 + j, i] ≈ cross_lag1
    end
end

@testset "MinnesotaPrior: random_walk = false gives a zero prior mean" begin
    Y_endog = BVAR.get_endogenous(df_p, end_vec_p)
    prior = BVAR.minnesota_prior(Y_endog, lags_p, end_vec_p, true; random_walk = false)
    @test all(prior.β0 .== 0.0)
end

@testset "MinnesotaPrior via build_prior matches the direct call" begin
    hp = (λ1 = 0.25, λ2 = 0.6, λ3 = 1.2, λ4 = 1e5)
    Y_endog = BVAR.get_endogenous(df_p, end_vec_p)
    direct = BVAR.minnesota_prior(Y_endog, lags_p, end_vec_p, true; λ = hp)
    via_build = build_prior(
        df_p,
        end_vec_p,
        est_p,
        :minnesota;
        hyperparameter_method = :fixed,
        hyperparameters = hp,
    )
    @test via_build.β0 == direct.β0
    @test via_build.Ω0 == direct.Ω0
    @test via_build.σ_ar == direct.σ_ar
end
