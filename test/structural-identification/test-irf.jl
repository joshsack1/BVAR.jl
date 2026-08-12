@testset "lag_blocks known simple case" begin
    S = Φ_p'  # k×n, single lag, no constant
    Φ = BVAR.lag_blocks(S, 1, false)
    @test length(Φ) == 1
    @test Φ[1] ≈ Φ_p
end

@testset "lag_blocks round-trips VARestimate.β_hat's row convention" begin
    Φ = BVAR.lag_blocks(est_p.β_hat, lags_p, true)
    @test length(Φ) == lags_p
    @test all(size(Φℓ) == (2, 2) for Φℓ in Φ)
    for ℓ in 1:lags_p
        block = est_p.β_hat[(1 + (ℓ - 1) * 2 + 1):(1 + ℓ * 2), :]
        @test Φ[ℓ] ≈ block'
    end
end

@testset "nonorthogonalized_irf for a VAR(1)" begin
    Φ = [Φ_p]
    Ψ = BVAR.nonorthogonalized_irf(Φ, 3)
    @test length(Ψ) == 4
    @test Ψ[1] == I(2)
    @test Ψ[2] ≈ Φ_p
    @test Ψ[3] ≈ Φ_p^2
    @test Ψ[4] ≈ Φ_p^3
end

@testset "nonorthogonalized_irf for a VAR(2)" begin
    Φ2 = [[0.4 0.1; 0.0 0.3], [0.2 0.0; 0.05 0.1]]
    Ψ = BVAR.nonorthogonalized_irf(Φ2, 2)
    @test Ψ[1] == I(2)
    @test Ψ[2] ≈ Φ2[1]
    @test Ψ[3] ≈ Φ2[1]^2 + Φ2[2]
end

@testset "impulse_responses horizon 0 returns the impact matrix" begin
    Φ = [Φ_p]
    impact = [1.0 0.0; 0.5 1.0]
    H = BVAR.impulse_responses(Φ, impact, 5)
    @test length(H) == 6
    @test H[1] ≈ impact
    @test H[2] ≈ Φ_p * impact
end

@testset "long_run_multiplier at A = I reduces to the familiar long-run multiplier" begin
    n = 2
    A = Matrix(1.0I, n, n)
    B = Φ_p'  # stored convention, single lag, no constant
    Ξ = long_run_multiplier(A, B, 1, false)
    @test Ξ ≈ inv(I - Φ_p)
end

@testset "impulse_response(::StructuralDraws) at A = I matches nonorthogonalized_irf" begin
    n = 2
    A = [Matrix(1.0I, n, n)]
    B = [Matrix(Φ_p')]
    D = [[1.0, 1.0]]
    draws = BVAR.StructuralDraws(A, B, D, 1, n, [:y1, :y2], false)
    irf = impulse_response(draws; horizon = 3)
    @test irf isa BVAR.IRFdraws
    @test irf.method == :hamilton_structural
    Ψ = BVAR.nonorthogonalized_irf([Φ_p], 3)
    @test irf.H[1] ≈ Ψ
end
