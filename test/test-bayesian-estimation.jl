using LinearAlgebra

include("bayesian-estimation/test-conjugate-sampling.jl")
include("bayesian-estimation/test-independent-niw-gibbs.jl")

@testset "sample_posterior dispatch across all five families" begin
    for (family, hyperparameter_method, hyperparameters) in (
        (:minnesota, :marginal_likelihood, nothing),
        (:normal_wishart, :marginal_likelihood, nothing),
        (:asymmetric_conjugate, :marginal_likelihood, nothing),
        (:hamilton_baumeister, :marginal_likelihood, nothing),
        (:independent_niw, :fixed, (λ1 = 0.2, λ2 = 0.5, λ3 = 1.0, λ4 = 1e5)),
    )
        prior = build_prior(
            df_p,
            end_vec_p,
            est_p,
            family;
            hyperparameter_method = hyperparameter_method,
            hyperparameters = hyperparameters,
        )
        draws = sample_posterior(prior, est_p; ndraws = 10, rng = Xoshiro(1))
        @test draws isa BVAR.BVARdraws
        @test draws.family == family
        @test length(draws.β) == 10
        @test length(draws.Σ) == 10
        @test draws.lags == lags_p
        @test draws.vars == length(end_vec_p)
        @test draws.names == end_vec_p
    end
end
