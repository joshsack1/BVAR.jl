# Speed comparison: the default stacked OLS method vs the FixedEffectModels method.
# Run with: julia --project=benchmark benchmark/benchmarks.jl
using BenchmarkTools
using BayesianVectorAutoregressions
using DataFrames
using Random

"""
    simulate_data(T_obs, n; rng = Xoshiro(42))

Simulates a stable VAR(1) with ``\\Phi_1 = 0.5 I`` and standard normal
innovations, returning a ``T_{obs} \\times n`` matrix for benchmarking.
"""
function simulate_data(T_obs, n; rng = Xoshiro(42))
    Y = zeros(T_obs, n)
    for t in 2:T_obs
        Y[t, :] = 0.5 * Y[t - 1, :] + randn(rng, n)
    end
    return Y
end

println("Benchmark: estimate_var method = :ols (default) vs method = :fem")
println("="^76)
for (T_obs, n, p) in ((200, 3, 2), (1000, 6, 4))
    Y = simulate_data(T_obs, n)
    end_vec = Symbol.(:y, 1:n)
    df = DataFrame(Y, end_vec)
    b_ols = @benchmark estimate_var($df, $end_vec, $p; method = :ols)
    b_fem = @benchmark estimate_var($df, $end_vec, $p; method = :fem)
    t_ols = median(b_ols).time
    t_fem = median(b_fem).time
    println("T = $T_obs, n = $n, p = $p")
    println("  method = :ols (default): $(BenchmarkTools.prettytime(t_ols))")
    println("  method = :fem:           $(BenchmarkTools.prettytime(t_fem))")
    println("  ratio (:fem / :ols):     $(round(t_fem / t_ols; digits = 1))x")
    println()
end
println("""
The default :ols method solves all equations in one stacked least squares
problem, while :fem runs one full FixedEffectModels fit per equation. The
two agree to numerical precision; :fem remains available for
FixedEffectModels' regression diagnostics.""")
