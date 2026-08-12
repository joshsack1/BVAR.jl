# Speed comparison: the FixedEffectModels estimator vs the hand-rolled OLS.
# Run with: julia --project=benchmark benchmark/benchmarks.jl
using BenchmarkTools
using BVAR
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

println("Benchmark: estimate_var (FixedEffectModels) vs BVAR.ols_var (hand-rolled)")
println("="^76)
for (T_obs, n, p) in ((200, 3, 2), (1000, 6, 4))
    Y = simulate_data(T_obs, n)
    end_vec = Symbol.(:y, 1:n)
    df = DataFrame(Y, end_vec)
    b_fem = @benchmark estimate_var($df, $end_vec, $p)
    b_ols = @benchmark BVAR.ols_var($Y, $p, true)
    t_fem = median(b_fem).time
    t_ols = median(b_ols).time
    println("T = $T_obs, n = $n, p = $p")
    println("  estimate_var (FixedEffectModels): $(BenchmarkTools.prettytime(t_fem))")
    println("  ols_var      (hand-rolled):       $(BenchmarkTools.prettytime(t_ols))")
    println("  ratio (FEM / hand-rolled):        $(round(t_fem / t_ols; digits = 1))x")
    println()
end
println("""
The hand-rolled OLS is expected to be faster: it is a single stacked linear
solve, while estimate_var runs one full FixedEffectModels fit per equation.
estimate_var is the production estimator because it also supplies the
standard errors and diagnostics the prior-building stage consumes.""")
