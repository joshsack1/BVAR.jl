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

@testset "parametric prior agrees with the equivalent per-entry prior" begin
    n = length(end_vec_p)
    reduced_form = build_prior(df_p, end_vec_p, est_p, :hamilton_baumeister)
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)

    template = Matrix(1.0I, n, n)
    free = falses(n, n)
    free[2, 1] = true
    component = Dict{Tuple{Int,Int},UnivariateDistribution}((2, 1) => Normal(0.0, 1.0))
    A_prior_entry = structural_prior(template, free, component; names = end_vec_p)
    prior_entry = hamilton_structural_prior(reduced_form, A_prior_entry, Y_endog)

    A_prior_param = parametric_structural_prior(
        UnivariateDistribution[Normal(0.0, 1.0)],
        θ -> [1.0 0.0; θ[1] 1.0],
        n;
        names = end_vec_p,
    )
    @test A_prior_param isa ParametricStructuralPrior
    @test A_prior_param isa AbstractStructuralPrior
    prior_param = hamilton_structural_prior(reduced_form, A_prior_param, Y_endog)
    @test prior_param isa HamiltonStructuralPrior

    draws_entry, _ =
        sample_structural(prior_entry, est_p; ndraws = 4000, rng = Xoshiro(3), method = :mh)
    draws_param, _ =
        sample_structural(prior_param, est_p; ndraws = 4000, rng = Xoshiro(3), method = :mh)
    mean_entry = sum(A[2, 1] for A in draws_entry.A) / length(draws_entry.A)
    mean_param = sum(A[2, 1] for A in draws_param.A) / length(draws_param.A)
    @test mean_entry ≈ mean_param atol = 0.2
end

@testset "extra_logprior enters the candidate weight" begin
    n = length(end_vec_p)
    reduced_form = build_prior(df_p, end_vec_p, est_p, :hamilton_baumeister)
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    gram = BayesianVectorAutoregressions.gram_blocks(est_p)
    θ_prior = UnivariateDistribution[Normal(0.0, 1.0)]
    A_map = θ -> [1.0 0.0; θ[1] 1.0]

    plain = hamilton_structural_prior(
        reduced_form,
        parametric_structural_prior(θ_prior, A_map, n; names = end_vec_p),
        Y_endog,
    )
    shifted = hamilton_structural_prior(
        reduced_form,
        parametric_structural_prior(
            θ_prior,
            A_map,
            n;
            extra_logprior = Function[(θ, A) -> -1.25],
            names = end_vec_p,
        ),
        Y_endog,
    )
    impossible = hamilton_structural_prior(
        reduced_form,
        parametric_structural_prior(
            θ_prior,
            A_map,
            n;
            extra_logprior = Function[(θ, A) -> -Inf],
            names = end_vec_p,
        ),
        Y_endog,
    )

    # identical θ and identically seeded rng: the weights differ by exactly
    # the extra term, because the (B, d) | A draw consumes the same stream
    θ = [0.3]
    _, _, _, logw_plain = BayesianVectorAutoregressions.evaluate_candidate(
        plain, θ, gram, est_p.Σ, est_p.obs, Xoshiro(5),
    )
    _, _, _, logw_shifted = BayesianVectorAutoregressions.evaluate_candidate(
        shifted, θ, gram, est_p.Σ, est_p.obs, Xoshiro(5),
    )
    _, _, _, logw_impossible = BayesianVectorAutoregressions.evaluate_candidate(
        impossible, θ, gram, est_p.Σ, est_p.obs, Xoshiro(5),
    )
    @test isfinite(logw_plain)
    @test logw_shifted ≈ logw_plain - 1.25
    @test logw_impossible == -Inf
end

@testset "sample_structural: :rwmh agrees with :mh and reports θ draws" begin
    n = length(end_vec_p)
    reduced_form = build_prior(df_p, end_vec_p, est_p, :hamilton_baumeister)
    template = Matrix(1.0I, n, n)
    free = falses(n, n)
    free[2, 1] = true
    component = Dict{Tuple{Int,Int},UnivariateDistribution}((2, 1) => Normal(0.0, 1.0))
    A_prior = structural_prior(template, free, component; names = end_vec_p)
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    prior = hamilton_structural_prior(reduced_form, A_prior, Y_endog)

    draws_mh, _ =
        sample_structural(prior, est_p; ndraws = 4000, rng = Xoshiro(3), method = :mh)
    draws_rw, diag_rw =
        sample_structural(prior, est_p; ndraws = 4000, rng = Xoshiro(3), method = :rwmh)

    @test 0 < diag_rw.acceptance_rate < 1
    @test size(diag_rw.θ) == (4000, 1)
    # the θ diagnostic is the same object the stored A entries came from
    @test all(diag_rw.θ[i, 1] == draws_rw.A[i][2, 1] for i in 1:4000)

    mean_mh = sum(A[2, 1] for A in draws_mh.A) / length(draws_mh.A)
    mean_rw = sum(A[2, 1] for A in draws_rw.A) / length(draws_rw.A)
    @test mean_mh ≈ mean_rw atol = 0.2

    # a user-supplied proposal_scale (with θ₀) skips tuning and still works
    draws_us, diag_us = sample_structural(
        prior,
        est_p;
        ndraws = 2000,
        rng = Xoshiro(9),
        method = :rwmh,
        θ₀ = [0.0],
        proposal_scale = fill(0.01, 1, 1),
        ξ = 0.5,
        proposal_df = 3.0,
    )
    @test 0 < diag_us.acceptance_rate < 1
    mean_us = sum(A[2, 1] for A in draws_us.A) / length(draws_us.A)
    @test mean_us ≈ mean_mh atol = 0.2
end

@testset ":rwmh respects a truncated component's support" begin
    n = length(end_vec_p)
    reduced_form = build_prior(df_p, end_vec_p, est_p, :hamilton_baumeister)
    template = Matrix(1.0I, n, n)
    free = falses(n, n)
    free[2, 1] = true
    component = Dict{Tuple{Int,Int},UnivariateDistribution}(
        (2, 1) => truncated(Normal(0.0, 1.0), -2.0, 0.0),
    )
    A_prior = structural_prior(template, free, component; names = end_vec_p)
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    prior = hamilton_structural_prior(reduced_form, A_prior, Y_endog)

    draws, diag =
        sample_structural(prior, est_p; ndraws = 2000, rng = Xoshiro(7), method = :rwmh)
    @test 0 < diag.acceptance_rate < 1
    @test all(-2.0 <= A[2, 1] <= 0.0 for A in draws.A)
end

@testset "structural_log_posterior is finite, differentiable, and peaked" begin
    n = length(end_vec_p)
    reduced_form = build_prior(df_p, end_vec_p, est_p, :hamilton_baumeister)
    template = Matrix(1.0I, n, n)
    free = falses(n, n)
    free[2, 1] = true
    component = Dict{Tuple{Int,Int},UnivariateDistribution}((2, 1) => Normal(0.0, 1.0))
    A_prior = structural_prior(template, free, component; names = end_vec_p)
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    prior = hamilton_structural_prior(reduced_form, A_prior, Y_endog)

    logpost = structural_log_posterior(prior, est_p)
    θ_med = [median(Normal(0.0, 1.0))]
    @test isfinite(logpost(θ_med))
    grad = BayesianVectorAutoregressions.ForwardDiff.gradient(logpost, θ_med)
    @test all(isfinite, grad)

    # the tuner's mode beats both the starting point and a far tail point,
    # and lands near the :sir posterior mean
    gram = BayesianVectorAutoregressions.gram_blocks(est_p)
    mode, scale = BayesianVectorAutoregressions.tune_rwmh_proposal(
        prior, gram, est_p.Σ, est_p.obs, θ_med,
    )
    @test logpost(mode) >= logpost(θ_med)
    @test logpost(mode) > logpost([3.0])
    @test isposdef(Symmetric(Matrix(scale)))
    draws_sir, _ = sample_structural(
        prior, est_p; ndraws = 4000, rng = Xoshiro(3), method = :sir, oversample = 20,
    )
    mean_sir = sum(A[2, 1] for A in draws_sir.A) / length(draws_sir.A)
    @test mode[1] ≈ mean_sir atol = 0.2
end

@testset ":rwmh tuner falls back gracefully when optimization fails" begin
    n = length(end_vec_p)
    reduced_form = build_prior(df_p, end_vec_p, est_p, :hamilton_baumeister)
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    # Float64(θ[1]) throws on the ForwardDiff.Dual input the tuner's BFGS
    # gradient uses, but is harmless on the chain's Float64 draws — so the
    # tuner must warn and fall back while sampling still succeeds.
    A_prior = parametric_structural_prior(
        UnivariateDistribution[Normal(0.0, 1.0)],
        θ -> [1.0 0.0; θ[1] 1.0],
        n;
        extra_logprior = Function[(θ, A) -> -abs2(Float64(θ[1])) / 100],
        names = end_vec_p,
    )
    prior = hamilton_structural_prior(reduced_form, A_prior, Y_endog)
    draws, diag = @test_logs (:warn, r"falling back") match_mode = :any sample_structural(
        prior, est_p; ndraws = 1000, rng = Xoshiro(13), method = :rwmh,
    )
    @test 0 < diag.acceptance_rate < 1
    @test length(draws.A) == 1000
end

@testset "a singular A is rejected with -Inf, not a crash" begin
    n = length(end_vec_p)
    reduced_form = build_prior(df_p, end_vec_p, est_p, :hamilton_baumeister)
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    gram = BayesianVectorAutoregressions.gram_blocks(est_p)
    # this map always produces a rank-1 A: A Σ̂ A′ is singular, τ degenerates,
    # and logdet would throw a DomainError were it not guarded — the sampler
    # must see a zero-density (-Inf) candidate instead of dying mid-chain
    A_prior = parametric_structural_prior(
        UnivariateDistribution[Normal(0.0, 1.0)],
        θ -> [1.0 1.0; 1.0 1.0],
        n;
        names = end_vec_p,
    )
    prior = hamilton_structural_prior(reduced_form, A_prior, Y_endog)
    A, B, d, logw = BayesianVectorAutoregressions.evaluate_candidate(
        prior, [0.3], gram, est_p.Σ, est_p.obs, Xoshiro(5),
    )
    @test logw == -Inf
    @test size(B) == (n * est_p.lags + 1, n)
    logpost = structural_log_posterior(prior, est_p)
    @test logpost([0.3]) == -Inf
end

@testset ":rwmh and parametric-prior guardrails" begin
    n = length(end_vec_p)
    reduced_form = build_prior(df_p, end_vec_p, est_p, :hamilton_baumeister)
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    template = Matrix(1.0I, n, n)
    free = falses(n, n)
    free[2, 1] = true
    component = Dict{Tuple{Int,Int},UnivariateDistribution}(
        (2, 1) => truncated(Normal(0.0, 1.0), -2.0, 0.0),
    )
    A_prior = structural_prior(template, free, component; names = end_vec_p)
    prior = hamilton_structural_prior(reduced_form, A_prior, Y_endog)

    # unknown method
    @test_throws AssertionError sample_structural(prior, est_p; method = :bogus)
    # θ₀ outside the marginal's support
    @test_throws AssertionError sample_structural(
        prior, est_p;
        method = :rwmh, θ₀ = [5.0], proposal_scale = fill(0.01, 1, 1),
    )
    # proposal_scale of the wrong size
    @test_throws AssertionError sample_structural(
        prior, est_p;
        method = :rwmh, proposal_scale = fill(0.01, 2, 2),
    )
    # a fully fixed A leaves a random walk nothing to move
    A_fixed = structural_prior(
        template,
        falses(n, n),
        Dict{Tuple{Int,Int},UnivariateDistribution}();
        names = end_vec_p,
    )
    prior_fixed = hamilton_structural_prior(reduced_form, A_fixed, Y_endog)
    @test_throws AssertionError sample_structural(prior_fixed, est_p; method = :rwmh)
    # the parametric builder rejects a map with the wrong output shape
    @test_throws AssertionError parametric_structural_prior(
        UnivariateDistribution[Normal(0.0, 1.0)],
        θ -> [1.0 0.0 θ[1]],
        n,
    )
end

@testset "BH-style supply/demand block runs end-to-end under :rwmh" begin
    n = length(end_vec_p)
    reduced_form = build_prior(df_p, end_vec_p, est_p, :hamilton_baumeister)
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    # θ = (supply elasticity > 0, demand elasticity < 0), fat-tailed t priors
    # as in Baumeister & Hamilton (2019), plus a soft prior on det(A)'s sign
    θ_prior = UnivariateDistribution[
        truncated(0.2 * TDist(3) + 0.1, 0.0, Inf),
        truncated(0.2 * TDist(3) - 0.1, -Inf, 0.0),
    ]
    A_map = θ -> [1.0 -θ[1]; 1.0 -θ[2]]
    A_prior = parametric_structural_prior(
        θ_prior,
        A_map,
        n;
        extra_logprior = Function[(θ, A) -> logpdf(Normal(1.0, 2.0), det(A))],
        names = end_vec_p,
    )
    prior = hamilton_structural_prior(reduced_form, A_prior, Y_endog)
    draws, diag =
        sample_structural(prior, est_p; ndraws = 2000, rng = Xoshiro(21), method = :rwmh)
    @test 0 < diag.acceptance_rate < 1
    @test size(diag.θ) == (2000, 2)
    @test all(diag.θ[:, 1] .>= 0.0)
    @test all(diag.θ[:, 2] .<= 0.0)
    @test all(det(A) > 0 for A in draws.A)  # supply slope above demand slope
end

@testset "η prior mean with A fixed at I is bit-identical to the random-walk prior" begin
    n = length(end_vec_p)
    k = n * lags_p + 1
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    η_rw = zeros(n, k)
    for i in 1:n
        η_rw[i, 1 + i] = 1.0
    end
    rf_η = BayesianVectorAutoregressions.baumeister_hamilton_prior(
        Y_endog, lags_p, end_vec_p, true; random_walk = false, η = η_rw,
    )
    rf_rw = BayesianVectorAutoregressions.baumeister_hamilton_prior(
        Y_endog, lags_p, end_vec_p, true,
    )
    template = Matrix(1.0I, n, n)
    free = falses(n, n)
    component = Dict{Tuple{Int,Int},UnivariateDistribution}()
    A_prior = structural_prior(template, free, component; names = end_vec_p)
    prior_η = hamilton_structural_prior(rf_η, A_prior, Y_endog)
    prior_rw = hamilton_structural_prior(rf_rw, A_prior, Y_endog)

    # With every A ≡ I, mᵢ(I) = η'eᵢ = the random-walk mean exactly, and the
    # RNG stream is identical — so the (B, D) | A draws must match bit-for-bit.
    draws_η, _ =
        sample_structural(prior_η, est_p; ndraws = 500, rng = Xoshiro(11), method = :mh)
    draws_rw, _ =
        sample_structural(prior_rw, est_p; ndraws = 500, rng = Xoshiro(11), method = :mh)
    @test draws_η.A == draws_rw.A
    @test draws_η.B == draws_rw.B
    @test draws_η.D == draws_rw.D
end

@testset "η rotates the prior mean with A: exact identity against constant m" begin
    n = length(end_vec_p)
    k = n * lags_p + 1
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    η_rw = zeros(n, k)
    for i in 1:n
        η_rw[i, 1 + i] = 1.0
    end
    template = Matrix(1.0I, n, n)
    free = falses(n, n)
    free[2, 1] = true
    component = Dict{Tuple{Int,Int},UnivariateDistribution}((2, 1) => Normal(0.0, 1.0))
    A_prior = structural_prior(template, free, component; names = end_vec_p)

    rf_η = BayesianVectorAutoregressions.baumeister_hamilton_prior(
        Y_endog, lags_p, end_vec_p, true; random_walk = false, η = η_rw,
    )
    prior_η = hamilton_structural_prior(rf_η, A_prior, Y_endog)
    logpost_η = structural_log_posterior(prior_η, est_p)

    # At a fixed θ, a constant-m prior whose m is η'aᵢ evaluated at that θ's A
    # must give the identical log posterior — this pins the mᵢ(A) seam exactly.
    θ_pin = [0.5]
    A_pin = [1.0 0.0; 0.5 1.0]
    m_pin = [Vector(η_rw' * A_pin[i, :]) for i in 1:n]
    rf_const = BayesianVectorAutoregressions.baumeister_hamilton_prior(
        Y_endog, lags_p, end_vec_p, true; random_walk = false, m = m_pin,
    )
    prior_const = hamilton_structural_prior(rf_const, A_prior, Y_endog)
    @test logpost_η(θ_pin) == structural_log_posterior(prior_const, est_p)(θ_pin)

    # At A = I the η prior coincides with the random-walk prior; away from I
    # the A-dependence must actually bite.
    rf_rw = BayesianVectorAutoregressions.baumeister_hamilton_prior(
        Y_endog, lags_p, end_vec_p, true,
    )
    prior_rw = hamilton_structural_prior(rf_rw, A_prior, Y_endog)
    logpost_rw = structural_log_posterior(prior_rw, est_p)
    @test logpost_η([0.0]) == logpost_rw([0.0])
    @test logpost_η(θ_pin) != logpost_rw(θ_pin)
end

@testset "η path is differentiable and samples under :rwmh" begin
    n = length(end_vec_p)
    k = n * lags_p + 1
    Y_endog = BayesianVectorAutoregressions.get_endogenous(df_p, end_vec_p)
    η_rw = zeros(n, k)
    for i in 1:n
        η_rw[i, 1 + i] = 1.0
    end
    rf_η = BayesianVectorAutoregressions.baumeister_hamilton_prior(
        Y_endog, lags_p, end_vec_p, true; random_walk = false, η = η_rw,
    )
    template = Matrix(1.0I, n, n)
    free = falses(n, n)
    free[2, 1] = true
    component = Dict{Tuple{Int,Int},UnivariateDistribution}((2, 1) => Normal(0.0, 1.0))
    A_prior = structural_prior(template, free, component; names = end_vec_p)
    prior_η = hamilton_structural_prior(rf_η, A_prior, Y_endog)
    logpost = structural_log_posterior(prior_η, est_p)

    # The mean's A-dependence must enter the gradient: check autodiff against
    # a central finite difference (it would silently vanish if mᵢ were built
    # from a Float-truncated aᵢ).
    θ0 = [0.3]
    grad = BayesianVectorAutoregressions.ForwardDiff.gradient(logpost, θ0)
    @test all(isfinite, grad)
    h = 1e-6
    fd = (logpost([θ0[1] + h]) - logpost([θ0[1] - h])) / (2h)
    @test grad[1] ≈ fd rtol = 1e-6

    draws, diag =
        sample_structural(prior_η, est_p; ndraws = 1000, rng = Xoshiro(7), method = :rwmh)
    @test 0 < diag.acceptance_rate < 1
    @test length(draws.A) == 1000
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
