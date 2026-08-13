using LinearAlgebra

# Shared fixture for every priors test file below: a stable VAR(1) with a
# clearly nonzero unconditional mean (the sum-of-coefficients and
# dummy-initial-observation dummy priors are designed for variables in
# meaningful levels, e.g. log GDP, and are uninformative for demeaned data).
Φ_p = [0.5 0.1; 0.1 0.4]
c_p = [2.0, 3.0]
@assert companion_spectral_radius([Φ_p]) < 1 "Priors test fixture VAR must be stable"
Y_p = simulate_var([Φ_p], c_p, 300)
df_p = DataFrame(Y_p, [:y1, :y2])
end_vec_p = [:y1, :y2]
lags_p = 2
est_p = estimate_var(df_p, end_vec_p, lags_p)

include("priors/test-minnesota.jl")
include("priors/test-dummy-observations.jl")
include("priors/test-conjugate.jl")
include("priors/test-hamilton.jl")

@testset "build_prior dispatch across all five families" begin
    for (family, dummy_components, H) in (
        (:minnesota, Symbol[], nothing),
        (:normal_wishart, Symbol[], nothing),
        (:normal_wishart, [:minnesota, :sum_of_coefficients, :dummy_initial_obs], nothing),
        (:asymmetric_conjugate, Symbol[], nothing),
        (:hamilton_baumeister, Symbol[], nothing),
    )
        prior = build_prior(
            df_p,
            end_vec_p,
            est_p,
            family;
            dummy_components = dummy_components,
            H = H,
        )
        @test prior isa BayesianVectorAutoregressions.AbstractVARPrior
        @test prior.lags == lags_p
        @test prior.vars == length(end_vec_p)
        @test prior.names == end_vec_p
    end
end

@testset "independent_niw requires hyperparameter_method = :fixed" begin
    hp = (λ1 = 0.2, λ2 = 0.5, λ3 = 1.0, λ4 = 1e5)
    prior = build_prior(
        df_p,
        end_vec_p,
        est_p,
        :independent_niw;
        hyperparameter_method = :fixed,
        hyperparameters = hp,
    )
    @test prior isa IndependentNIWPrior
    @test_throws AssertionError build_prior(df_p, end_vec_p, est_p, :independent_niw)
end

@testset "Errors and contract checks" begin
    @test_throws AssertionError build_prior(df_p, end_vec_p, est_p, :bogus)
    @test_throws AssertionError build_prior(
        df_p,
        end_vec_p,
        est_p,
        :normal_wishart;
        dummy_components = [:bogus],
    )
    @test_throws AssertionError build_prior(
        df_p,
        end_vec_p,
        est_p,
        :minnesota;
        dummy_components = [:minnesota],
    )
    @test_throws AssertionError build_prior(
        df_p,
        end_vec_p,
        est_p,
        :minnesota;
        hyperparameter_method = :fixed,
    )
    @test_throws AssertionError build_prior(
        df_p,
        end_vec_p,
        est_p,
        :normal_wishart;
        dummy_components = [:long_run],
    )
    # Too few dummy rows for a proper Inverse-Wishart (ν0 = T_d - k = 5 - 5 = 0, needs > n - 1).
    @test_throws AssertionError build_prior(
        df_p,
        end_vec_p,
        est_p,
        :normal_wishart;
        dummy_components = [:minnesota, :dummy_initial_obs],
        hyperparameter_method = :fixed,
        hyperparameters = (
            λ1 = 0.2,
            λ3 = 1.0,
            λ4 = 1e5,
            λ_soc = 1.0,
            λ_dio = 1.0,
            λ_lr = 1.0,
        ),
    )
end

@testset "Marginal-likelihood optimum is not dominated by a grid neighbor" begin
    gram = BayesianVectorAutoregressions.gram_blocks(est_p)
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    prior = build_prior(df_p, end_vec_p, est_p, :minnesota)
    obj(λ1, λ2, λ3) = BayesianVectorAutoregressions.log_marginal_likelihood(
        BayesianVectorAutoregressions.minnesota_prior(
            Y_endog,
            lags_p,
            end_vec_p,
            true;
            λ = (λ1 = λ1, λ2 = λ2, λ3 = λ3, λ4 = 1e5),
        ),
        gram.XᵀX,
        gram.XᵀY,
        gram.YᵀY,
        est_p.obs,
    )
    best = obj(prior.λ.λ1, prior.λ.λ2, prior.λ.λ3)
    for δ1 in (-0.05, 0.05), δ2 in (-0.05, 0.05), δ3 in (-0.1, 0.1)
        λ1n = clamp(prior.λ.λ1 + δ1, 0.01, 2.0)
        λ2n = clamp(prior.λ.λ2 + δ2, 0.01, 2.0)
        λ3n = clamp(prior.λ.λ3 + δ3, 0.1, 3.0)
        @test obj(λ1n, λ2n, λ3n) <= best + 1e-6
    end
end
