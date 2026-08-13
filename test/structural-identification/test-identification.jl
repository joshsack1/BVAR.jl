@testset "random_orthogonal returns an orthogonal matrix" begin
    Q = BayesianVectorAutoregressions.random_orthogonal(3, Xoshiro(1), Float64)
    @test Q' * Q ≈ I(3) atol = 1e-10
end

@testset "identify_short_run: lower-triangular impact matrix" begin
    prior = build_prior(df_p, end_vec_p, est_p, :normal_wishart)
    draws = sample_posterior(prior, est_p; ndraws = 20, rng = Xoshiro(2))
    irf = identify_short_run(draws; horizon = 5)
    @test irf isa BayesianVectorAutoregressions.IRFdraws
    @test irf.method == :cholesky
    @test length(irf.H) == 20
    for d in 1:20
        impact = irf.H[d][1]
        @test impact[1, 2] ≈ 0.0 atol = 1e-10
        L = cholesky(Symmetric(draws.Σ[d])).L
        @test impact ≈ Matrix(L)
    end
end

@testset "identify_sign_restrictions: every draw satisfies the requested pattern" begin
    prior = build_prior(df_p, end_vec_p, est_p, :normal_wishart)
    draws = sample_posterior(prior, est_p; ndraws = 20, rng = Xoshiro(2))
    sign_pattern = [1 0; -1 1]
    irf = identify_sign_restrictions(draws, sign_pattern; horizon = 3, rng = Xoshiro(4))
    @test irf.method == :sign_restriction
    for d in 1:20
        impact = irf.H[d][1]
        @test impact[1, 1] > 0
        @test impact[2, 1] < 0
    end
end

@testset "identify_sign_restrictions fails loudly when max_attempts is exhausted" begin
    prior = build_prior(df_p, end_vec_p, est_p, :normal_wishart)
    draws = sample_posterior(prior, est_p; ndraws = 1, rng = Xoshiro(2))
    sign_pattern = [1 0; -1 1]
    @test_throws AssertionError identify_sign_restrictions(
        draws,
        sign_pattern;
        max_attempts = 0,
        rng = Xoshiro(9),
    )
end
