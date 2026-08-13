using Aqua
using DataFrames
using LinearAlgebra
using Random

"""
    simulate_var(Φ, c, T_obs; burn = 200, rng = Xoshiro(42))

Simulates a stable VAR(p) with standard normal innovations

``Y_t = c + \\Phi_1 Y_{t-1} + \\cdots + \\Phi_p Y_{t-p} + \\varepsilon_t``

where `Φ` is a vector of the ``n \\times n`` lag coefficient matrices
``\\Phi_1, \\ldots, \\Phi_p``. The first `burn` draws are discarded so the
returned ``T_{obs} \\times n`` matrix starts near the stationary distribution.
"""
function simulate_var(Φ, c, T_obs; burn = 200, rng = Xoshiro(42))
    p = length(Φ)
    n = length(c)
    Y = zeros(T_obs + burn, n)
    for t in (p + 1):(T_obs + burn)
        Y[t, :] = c + sum(Φ[i] * Y[t - i, :] for i in 1:p) + randn(rng, n)
    end
    return Y[(burn + 1):end, :]
end

"""
    companion_spectral_radius(Φ)

Computes the spectral radius of the companion matrix of the VAR(p) with lag
coefficient matrices `Φ`; the process is stable when it is below one.
"""
function companion_spectral_radius(Φ)
    p = length(Φ)
    n = size(Φ[1], 1)
    F = zeros(n * p, n * p)
    for i in 1:p
        F[1:n, ((i - 1) * n + 1):(i * n)] = Φ[i]
    end
    F[(n + 1):end, 1:(n * (p - 1))] = I(n * (p - 1))
    return maximum(abs.(eigvals(F)))
end

Φ₁ = [0.5 0.1; 0.2 0.3]
c₁ = [0.1, -0.2]
@assert companion_spectral_radius([Φ₁]) < 1 "Test VAR(1) must be stable"
Y = simulate_var([Φ₁], c₁, 300)
df = DataFrame(Y, [:y1, :y2])
end_vec = [:y1, :y2]

@testset "Agreement between the :ols and :fem methods" begin
    for lags in (1, 2, 3), include_constant in (true, false)
        ols = estimate_var(
            df,
            end_vec,
            lags;
            include_constant = include_constant,
            method = :ols,
        )
        fem = estimate_var(
            df,
            end_vec,
            lags;
            include_constant = include_constant,
            method = :fem,
        )
        ref = BayesianVectorAutoregressions.ols_var(Y, lags, include_constant)
        @test fem.β_hat ≈ ols.β_hat rtol = 1e-8
        @test fem.Σ ≈ ols.Σ rtol = 1e-8
        @test fem.se ≈ ols.se rtol = 1e-8
        @test fem.XᵀX ≈ ols.XᵀX rtol = 1e-10
        @test ols.β_hat ≈ ref.β_hat rtol = 1e-12
    end
end

@testset "Default method is :ols" begin
    default = estimate_var(df, end_vec, 2)
    ols = estimate_var(df, end_vec, 2; method = :ols)
    @test default.β_hat == ols.β_hat
    @test default.Σ == ols.Σ
    @test default.se == ols.se
    @test default.XᵀX == ols.XᵀX
end

@testset "Dimensions and metadata" begin
    lags = 2
    est = estimate_var(df, end_vec, lags)
    n = length(end_vec)
    @test size(est.β_hat) == (n * lags + 1, n)
    @test size(est.se) == (n * lags + 1, n)
    @test size(est.Σ) == (n, n)
    @test size(est.XᵀX) == (n * lags + 1, n * lags + 1)
    @test issymmetric(round.(est.Σ; digits = 12))
    @test est.obs == size(Y, 1) - lags
    @test est.lags == lags
    @test est.vars == n
    @test est.names == end_vec
    @test est.include_constant
    @test all(isfinite, est.se)
end

@testset "Known simple case" begin
    # A tiny dataset where (X'X) \ (X'Y) is computed inline as the truth
    Y_small = [
        1.0 2.0
        2.0 1.0
        4.0 3.0
        3.0 5.0
        5.0 4.0
        6.0 7.0
    ]
    df_small = DataFrame(Y_small, [:y1, :y2])
    X = hcat(ones(5), Y_small[1:5, :])
    β_truth = (X' * X) \ (X' * Y_small[2:6, :])
    est = estimate_var(df_small, [:y1, :y2], 1)
    @test est.β_hat ≈ β_truth rtol = 1e-8
end

@testset "Coefficient recovery" begin
    est₁ = estimate_var(DataFrame(simulate_var([Φ₁], c₁, 2000), [:y1, :y2]), end_vec, 1)
    @test est₁.β_hat[1, :] ≈ c₁ atol = 0.1
    @test est₁.β_hat[2:3, :]' ≈ Φ₁ atol = 0.1

    Φ₂ = [[0.4 0.1; 0.0 0.3], [0.2 0.0; 0.05 0.1]]
    c₂ = [0.3, -0.1]
    @assert companion_spectral_radius(Φ₂) < 1 "Test VAR(2) must be stable"
    est₂ = estimate_var(DataFrame(simulate_var(Φ₂, c₂, 2000), [:y1, :y2]), end_vec, 2)
    @test est₂.β_hat[1, :] ≈ c₂ atol = 0.1
    @test est₂.β_hat[2:3, :]' ≈ Φ₂[1] atol = 0.1
    @test est₂.β_hat[4:5, :]' ≈ Φ₂[2] atol = 0.1
end

@testset "Errors and contract checks" begin
    @test_throws AssertionError estimate_var(df, end_vec, 1; method = :bogus)
    @test_throws AssertionError estimate_var(df, end_vec, 0)
    @test_throws AssertionError estimate_var(df, end_vec, size(Y, 1))
    @test_throws AssertionError estimate_var(df, [:y1, :not_a_column], 1)
    df_nan = DataFrame(y1 = [1.0, NaN, 2.0, 3.0], y2 = [1.0, 2.0, 3.0, 4.0])
    @test_throws AssertionError estimate_var(df_nan, [:y1, :y2], 1)
    df_missing = DataFrame(y1 = [1.0, missing, 2.0, 3.0], y2 = [1.0, 2.0, 3.0, 4.0])
    @test_throws AssertionError estimate_var(df_missing, [:y1, :y2], 1)
end

@testset "Package QA (Aqua)" begin
    # Documenter is staged for docs; Turing and Random are used by the
    # Bayesian estimation stage (src/bayesian-estimation/)
    Aqua.test_all(BayesianVectorAutoregressions; stale_deps = (ignore = [:Documenter],))
end
