@testset "HamiltonStructuralPrior with A fully fixed at I reduces to the reduced-form path" begin
    n = length(end_vec_p)
    reduced_form = build_prior(df_p, end_vec_p, est_p, :hamilton_baumeister)
    template = Matrix(1.0I, n, n)
    free = falses(n, n)
    component = Dict{Tuple{Int,Int},UnivariateDistribution}()
    A_prior = structural_prior(template, free, component; names = end_vec_p)
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    prior = hamilton_structural_prior(reduced_form, A_prior, Y_endog)
    @test prior isa HamiltonStructuralPrior

    draws, diagnostics =
        sample_structural(prior, est_p; ndraws = 5000, rng = Xoshiro(11), method = :mh)
    @test draws isa StructuralDraws
    @test all(A == template for A in draws.A)
    @test diagnostics.acceptance_rate == 1.0  # nothing free to reject: every candidate A is the same template

    gram = BayesianVectorAutoregressions.gram_blocks(est_p)
    mean_B = sum(draws.B) / length(draws.B)
    for i in 1:n
        post = BayesianVectorAutoregressions.equation_normal_gamma_posterior(
            reduced_form.m[i],
            reduced_form.M[i],
            reduced_form.κ[i],
            reduced_form.τ[i],
            gram.XᵀX,
            gram.XᵀY[:, i],
            gram.YᵀY[i, i],
            est_p.obs,
        )
        @test mean_B[:, i] ≈ post.b̄ atol = 0.05
    end
end

@testset "sample_structural: :sir and :mh agree on a genuinely free case" begin
    n = length(end_vec_p)
    reduced_form = build_prior(df_p, end_vec_p, est_p, :hamilton_baumeister)
    template = Matrix(1.0I, n, n)
    free = falses(n, n)
    free[2, 1] = true
    component = Dict{Tuple{Int,Int},UnivariateDistribution}((2, 1) => Normal(0.0, 1.0))
    A_prior = structural_prior(template, free, component; names = end_vec_p)
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    prior = hamilton_structural_prior(reduced_form, A_prior, Y_endog)

    draws_mh, diag_mh =
        sample_structural(prior, est_p; ndraws = 4000, rng = Xoshiro(3), method = :mh)
    draws_sir, diag_sir = sample_structural(
        prior,
        est_p;
        ndraws = 4000,
        rng = Xoshiro(3),
        method = :sir,
        oversample = 20,
    )

    @test diag_mh.acceptance_rate > 0
    @test diag_sir.ess > 0

    mean_a21_mh = sum(A[2, 1] for A in draws_mh.A) / length(draws_mh.A)
    mean_a21_sir = sum(A[2, 1] for A in draws_sir.A) / length(draws_sir.A)
    @test mean_a21_mh ≈ mean_a21_sir atol = 0.2
end

@testset "restrictions contribute to draw_candidate's weight" begin
    n = length(end_vec_p)
    reduced_form = build_prior(df_p, end_vec_p, est_p, :hamilton_baumeister)
    template = Matrix(1.0I, n, n)
    free = falses(n, n)
    free[2, 1] = true
    component = Dict{Tuple{Int,Int},UnivariateDistribution}((2, 1) => Normal(0.0, 1.0))
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)

    A_prior_unrestricted = structural_prior(template, free, component; names = end_vec_p)
    prior_unrestricted =
        hamilton_structural_prior(reduced_form, A_prior_unrestricted, Y_endog)

    A_prior_impossible = structural_prior(
        template,
        free,
        component;
        restrictions = Function[(A, B) -> -Inf],
        names = end_vec_p,
    )
    prior_impossible = hamilton_structural_prior(reduced_form, A_prior_impossible, Y_endog)

    gram = BayesianVectorAutoregressions.gram_blocks(est_p)
    _, _, _, logw_unrestricted = BayesianVectorAutoregressions.draw_candidate(
        prior_unrestricted,
        gram,
        est_p.Σ,
        est_p.obs,
        Xoshiro(5),
    )
    _, _, _, logw_impossible = BayesianVectorAutoregressions.draw_candidate(
        prior_impossible,
        gram,
        est_p.Σ,
        est_p.obs,
        Xoshiro(5),
    )
    @test isfinite(logw_unrestricted)
    @test logw_impossible == -Inf
end

@testset "det_sign_restriction and long_run_sign_restriction" begin
    A_pos = [1.0 0.0; 0.0 1.0]  # det = 1 > 0
    A_neg = [0.0 1.0; 1.0 0.0]  # det = -1 < 0
    r_pos = det_sign_restriction(1)
    @test r_pos(A_pos, zeros(2, 2)) == 0.0
    @test r_pos(A_neg, zeros(2, 2)) == -Inf
    @test_throws AssertionError det_sign_restriction(0)

    A = Matrix(1.0I, 2, 2)
    B = Φ_p'  # single lag, no constant
    Ξ = long_run_multiplier(A, B, 1, false)
    correct_sgn = Ξ[1, 1] > 0 ? 1 : -1
    r = long_run_sign_restriction(1, 1, correct_sgn, 1, false)
    @test r(A, B) == 0.0
    r_wrong = long_run_sign_restriction(1, 1, -correct_sgn, 1, false)
    @test r_wrong(A, B) == -Inf
    @test_throws AssertionError long_run_sign_restriction(1, 1, 2, 1, false)
end
