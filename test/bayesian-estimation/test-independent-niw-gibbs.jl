niw_hp = (λ1 = 0.2, λ2 = 0.5, λ3 = 1.0, λ4 = 1e5)

@testset "IndependentNIWPrior: shapes and metadata" begin
    n = length(end_vec_p)
    k = n * lags_p + 1
    prior = build_prior(
        df_p,
        end_vec_p,
        est_p,
        :independent_niw;
        hyperparameter_method = :fixed,
        hyperparameters = niw_hp,
    )
    draws = sample_posterior(prior, est_p; ndraws = 15, burn_in = 15, rng = Xoshiro(1))
    @test draws isa BayesianVectorAutoregressions.BVARdraws
    @test draws.family == :independent_niw
    @test length(draws.β) == 15
    @test length(draws.Σ) == 15
    @test all(size(βd) == (k, n) for βd in draws.β)
    @test all(size(Σd) == (n, n) for Σd in draws.Σ)
    @test all(isposdef(Symmetric(Σd)) for Σd in draws.Σ)
    @test draws.lags == lags_p
    @test draws.vars == n
    @test draws.names == end_vec_p
end

@testset "IndependentNIWPrior: burn_in defaults to ndraws" begin
    prior = build_prior(
        df_p,
        end_vec_p,
        est_p,
        :independent_niw;
        hyperparameter_method = :fixed,
        hyperparameters = niw_hp,
    )
    draws = sample_posterior(prior, est_p; ndraws = 10, rng = Xoshiro(1))
    @test length(draws.β) == 10
end

@testset "Coefficient recovery via Gibbs sampling" begin
    # Reuses the small-magnitude Φ₁/c₁ fixture from the frequentist
    # "Coefficient recovery" testset (test-var-estimation.jl), whose atol=0.1
    # is already calibrated to this scale — a larger-magnitude intercept
    # (e.g. c=[2,3]) needs proportionally more data for the same absolute
    # tolerance to hold, since OLS/posterior intercept noise scales with
    # (I-Φ)⁻¹'s amplification of the innovation variance.
    df_niw = DataFrame(simulate_var([Φ₁], c₁, 2000), [:y1, :y2])
    end_vec_niw = [:y1, :y2]
    est_niw = estimate_var(df_niw, end_vec_niw, 1)
    prior = build_prior(
        df_niw,
        end_vec_niw,
        est_niw,
        :independent_niw;
        hyperparameter_method = :fixed,
        hyperparameters = niw_hp,
    )
    draws = sample_posterior(prior, est_niw; ndraws = 1000, rng = Xoshiro(1))
    mean_β = sum(draws.β) / length(draws.β)
    @test mean_β[1, :] ≈ c₁ atol = 0.1
    @test mean_β[2:3, :]' ≈ Φ₁ atol = 0.1
end

@testset "Errors and contract checks" begin
    prior = build_prior(
        df_p,
        end_vec_p,
        est_p,
        :independent_niw;
        hyperparameter_method = :fixed,
        hyperparameters = niw_hp,
    )
    @test_throws AssertionError sample_posterior(prior, est_p; ndraws = 0)
    @test_throws AssertionError sample_posterior(prior, est_p; ndraws = 5, burn_in = -1)
    est_wrong_vars = estimate_var(df_p[:, [:y1]], [:y1], lags_p)
    prior_wrong_vars = build_prior(
        df_p[:, [:y1]],
        [:y1],
        est_wrong_vars,
        :independent_niw;
        hyperparameter_method = :fixed,
        hyperparameters = niw_hp,
    )
    @test_throws AssertionError sample_posterior(prior_wrong_vars, est_p)
end
