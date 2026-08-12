@testset "normal_wishart_prior (direct): shapes and Kadiyala-Karlsson moments" begin
    Y_endog = BVAR.get_endogenous(df_p, end_vec_p)
    n = length(end_vec_p)
    k = n * lags_p + 1
    λ = (λ1 = 0.3, λ3 = 1.1, λ4 = 1e4, λ_soc = 1.0, λ_dio = 1.0, λ_lr = 1.0)
    prior = BVAR.normal_wishart_prior(Y_endog, lags_p, end_vec_p, true; λ = λ)
    σ_ar = sqrt.(BVAR.ar_residual_variances(Y_endog, lags_p))

    @test prior isa NormalWishartPrior
    @test size(prior.β0) == (k, n)
    @test size(prior.Ω0) == (k, k)
    @test size(prior.S0) == (n, n)
    @test isposdef(Symmetric(prior.Ω0))
    @test isposdef(Symmetric(prior.S0))
    @test prior.ν0 == n + 2

    # Ω0 is shared across equations (a single k×k matrix), diagonal, with the
    # own-lag entries matching the direct formula.
    @test isdiag(prior.Ω0)
    for j in 1:n
        @test prior.Ω0[1 + j, 1 + j] ≈ (λ.λ1 / 1^λ.λ3)^2 / σ_ar[j]^2
    end
    @test prior.Ω0[1, 1] ≈ λ.λ4^2

    # E[Σ] = S0 / (ν0 - n - 1) recovers the univariate AR variances.
    @test diag(prior.S0) ./ (prior.ν0 - n - 1) ≈ σ_ar .^ 2
end

@testset "normal_wishart_prior (dummy route) via build_prior matches the direct call" begin
    hp = (λ1 = 0.2, λ3 = 1.0, λ4 = 1e5, λ_soc = 1.0, λ_dio = 1.0, λ_lr = 1.0)
    Y_endog = BVAR.get_endogenous(df_p, end_vec_p)
    dummy_components = [:minnesota, :sum_of_coefficients, :dummy_initial_obs]
    direct = BVAR.normal_wishart_prior(
        Y_endog,
        lags_p,
        end_vec_p,
        true;
        dummy_components = dummy_components,
        λ = hp,
    )
    via_build = build_prior(
        df_p,
        end_vec_p,
        est_p,
        :normal_wishart;
        dummy_components = dummy_components,
        hyperparameter_method = :fixed,
        hyperparameters = hp,
    )
    @test via_build.β0 == direct.β0
    @test via_build.S0 == direct.S0
    @test via_build.ν0 == direct.ν0
end

@testset "IndependentNIWPrior reuses the per-equation Minnesota moments" begin
    Y_endog = BVAR.get_endogenous(df_p, end_vec_p)
    n = length(end_vec_p)
    λ = (λ1 = 0.25, λ2 = 0.4, λ3 = 1.3, λ4 = 1e4)
    prior = BVAR.independent_niw_prior(Y_endog, lags_p, end_vec_p, true; λ = λ)
    minn = BVAR.minnesota_prior(Y_endog, lags_p, end_vec_p, true; λ = λ)
    k = n * lags_p + 1

    @test length(prior.β0) == k * n
    @test size(prior.Ω0) == (k * n, k * n)
    @test isdiag(prior.Ω0)
    @test prior.β0 ≈ vec(minn.β0)
    @test diag(prior.Ω0) ≈ vec(minn.Ω0)
    @test isposdef(Symmetric(prior.S0))
end

@testset "AsymmetricConjugatePrior: per-equation Normal-Gamma structure" begin
    Y_endog = BVAR.get_endogenous(df_p, end_vec_p)
    n = length(end_vec_p)
    κ0 = 4.0
    λ = (λ1 = 0.25, λ2 = 0.4, λ3 = 1.3, λ4 = 1e4)
    prior =
        BVAR.asymmetric_conjugate_prior(Y_endog, lags_p, end_vec_p, true; λ = λ, κ0 = κ0)
    minn = BVAR.minnesota_prior(Y_endog, lags_p, end_vec_p, true; λ = λ)

    @test length(prior.β0) == n
    @test length(prior.Ω0) == n
    @test all(prior.κ .== κ0)
    for i in 1:n
        @test prior.β0[i] == minn.β0[:, i]
        @test diag(prior.Ω0[i]) == minn.Ω0[:, i]
        @test prior.τ[i] ≈ κ0 * minn.σ_ar[i]^2
        @test prior.κ[i] / prior.τ[i] ≈ 1 / minn.σ_ar[i]^2 # prior mean of d_ii^{-1}
    end
end
