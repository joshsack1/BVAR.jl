using Pkg
Pkg.activate(".")
#%%
using BVAR
using DataFrames
using Random
#%%
# Create sample data for testing
Random.seed!(123)
n_obs = 100
df = DataFrame(
    y1 = cumsum(randn(n_obs)),  # Random walk (unit root)
    y2 = cumsum(randn(n_obs)),  # Random walk (unit root)
    y3 = randn(n_obs)           # Stationary series
)
#%%
# Define endogenous variables
end_vars = [:y1, :y2, :y3]
#%%
println("Testing ADF Tests:")
println("==================")
adf_results = adf_tests(df, end_vars)
#%%
# This needs to be corrected
#=for (i, var) in enumerate(end_vars)=#
#=    println("Variable: $var")=#
#=    for (j, test) in enumerate(adf_results[i])=#
#=        test_types = ["No intercept/trend", "Constant", "Constant + trend"]=#
#=        println("  $(test_types[j]): p-value = $(pvalue(test))")=#
#=    end=#
#=    println()=#
#=end=#

println("Testing Johansen Trace Test:")
println("============================")
trace_stats, eigenvals = johansen_trace_test(df, end_vars, 2)
println("Trace Statistics: $trace_stats")
println("Eigenvalues: $eigenvals")
