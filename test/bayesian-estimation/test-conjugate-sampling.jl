@testset "sample_posterior (closed-form families): shapes and metadata" begin
    n = length(end_vec_p)
    k = n * lags_p + 1
    for family in (:minnesota, :normal_wishart, :asymmetric_conjugate, :hamilton_baumeister)
        prior = build_prior(df_p, end_vec_p, est_p, family)
        draws = sample_posterior(prior, est_p; ndraws = 20, rng = Xoshiro(1))
        @test draws isa BVAR.BVARdraws
        @test draws.family == family
        @test length(draws.β) == 20
        @test length(draws.Σ) == 20
        @test all(size(βd) == (k, n) for βd in draws.β)
        @test all(size(Σd) == (n, n) for Σd in draws.Σ)
        @test all(issymmetric(round.(Σd; digits = 10)) for Σd in draws.Σ)
        @test all(isposdef(Symmetric(Σd)) for Σd in draws.Σ)
        @test draws.lags == lags_p
        @test draws.vars == n
        @test draws.names == end_vec_p
    end
end

@testset "MinnesotaPrior: Σ draws are the fixed AR variance, not random" begin
    prior = build_prior(df_p, end_vec_p, est_p, :minnesota)
    draws = sample_posterior(prior, est_p; ndraws = 10, rng = Xoshiro(1))
    Σ_fixed = Diagonal(prior.σ_ar .^ 2)
    @test all(Σd == Σ_fixed for Σd in draws.Σ)
end

@testset "Posterior mean recovery against the analytic *_posterior helpers" begin
    gram = BVAR.gram_blocks(est_p)
    n = length(end_vec_p)

    prior_mn = build_prior(df_p, end_vec_p, est_p, :minnesota)
    draws_mn = sample_posterior(prior_mn, est_p; ndraws = 5000, rng = Xoshiro(7))
    post_mn = BVAR.minnesota_posterior(prior_mn, gram.XᵀX, gram.XᵀY)
    mean_β_mn = sum(draws_mn.β) / length(draws_mn.β)
    @test mean_β_mn ≈ post_mn.β̄ atol = 0.05

    prior_nw = build_prior(df_p, end_vec_p, est_p, :normal_wishart)
    draws_nw = sample_posterior(prior_nw, est_p; ndraws = 5000, rng = Xoshiro(7))
    post_nw =
        BVAR.normal_wishart_posterior(prior_nw, gram.XᵀX, gram.XᵀY, gram.YᵀY, est_p.obs)
    mean_β_nw = sum(draws_nw.β) / length(draws_nw.β)
    mean_Σ_nw = sum(draws_nw.Σ) / length(draws_nw.Σ)
    @test mean_β_nw ≈ post_nw.β̄ atol = 0.05
    @test mean_Σ_nw ≈ post_nw.S̄ / (post_nw.ν̄ - n - 1) atol = 0.05

    for (family, means, scales) in
        ((:asymmetric_conjugate, :β0, :Ω0), (:hamilton_baumeister, :m, :M))
        prior = build_prior(df_p, end_vec_p, est_p, family)
        draws = sample_posterior(prior, est_p; ndraws = 5000, rng = Xoshiro(7))
        mean_β = sum(draws.β) / length(draws.β)
        m = getproperty(prior, means)
        M = getproperty(prior, scales)
        for i in 1:n
            post = BVAR.equation_normal_gamma_posterior(
                m[i],
                M[i],
                prior.κ[i],
                prior.τ[i],
                gram.XᵀX,
                gram.XᵀY[:, i],
                gram.YᵀY[i, i],
                est_p.obs,
            )
            @test mean_β[:, i] ≈ post.b̄ atol = 0.05
        end
    end
end

@testset "Errors and contract checks" begin
    prior = build_prior(df_p, end_vec_p, est_p, :minnesota)
    est_wrong_lags = estimate_var(df_p, end_vec_p, lags_p + 1)
    @test_throws AssertionError sample_posterior(prior, est_wrong_lags)
end
