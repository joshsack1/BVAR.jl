@testset "ar_residual_covariance matches ar_residual_variances on the diagonal" begin
    Y_endog = BVAR.get_endogenous(df_p, end_vec_p)
    Ŝ = BVAR.ar_residual_covariance(Y_endog, lags_p)
    σ² = BVAR.ar_residual_variances(Y_endog, lags_p)
    @test diag(Ŝ) ≈ σ²
    @test issymmetric(round.(Ŝ; digits = 12))
end

@testset "ar_residual_covariance hand-computed 2-variable case" begin
    Y = [1.0 2.0; 2.0 1.0; 4.0 3.0; 3.0 5.0; 5.0 4.0; 6.0 7.0]
    Ŝ = BVAR.ar_residual_covariance(Y, 1)
    e1 = BVAR.ols_var(reshape(Y[:, 1], :, 1), 1, true).ε
    e2 = BVAR.ols_var(reshape(Y[:, 2], :, 1), 1, true).ε
    @test Ŝ[1, 2] ≈ (e1' * e2)[1] / length(e1)
    @test Ŝ[1, 1] ≈ (e1' * e1)[1] / length(e1)
end

@testset "structural_prior construction and validation" begin
    n = 2
    template = Matrix(1.0I, n, n)
    free = falses(n, n)
    free[2, 1] = true
    component = Dict{Tuple{Int,Int},UnivariateDistribution}((2, 1) => Normal(0.0, 1.0))
    prior = structural_prior(template, free, component)
    @test prior isa StructuralPrior
    @test prior.vars == n
    @test prior.template == template
    @test prior.free == free
    @test prior.component[(2, 1)] isa Normal
    @test isempty(prior.restrictions)

    @test_throws AssertionError structural_prior(zeros(3, 2), free, component)
    @test_throws AssertionError structural_prior(template, falses(3, 3), component)
    @test_throws AssertionError structural_prior(
        template,
        free,
        Dict{Tuple{Int,Int},UnivariateDistribution}(),
    )
    @test_throws AssertionError structural_prior(
        template,
        free,
        component;
        names = [:only_one],
    )
end
