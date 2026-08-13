using Pkg
Pkg.activate(".")
#%%
using BayesianVectorAutoregressions
using DataFrames
using Random
#%%
# Create sample data for testing
Random.seed!(123)
n_obs = 100
df = DataFrame(
    y1 = cumsum(randn(n_obs)),  # Random walk (unit root)
    y2 = cumsum(randn(n_obs)),  # Random walk (unit root)
    y3 = randn(n_obs),           # Stationary series
)
#%%
# Define endogenous variables
end_vars = [:y1, :y2, :y3]
#%%
println("Testing ADF Tests:")
println("==================")
adf_results = adf_tests(df, end_vars)#=for (i, var) in enumerate(end_vars)=##=    println("Variable: $var")=##=    for (j, test) in enumerate(adf_results[i])=##=        test_types = ["No intercept/trend", "Constant", "Constant + trend"]=##=        println("  $(test_types[j]): p-value = $(pvalue(test))")=##=    end=##=    println()=##=end=#
#%%
# This needs to be corrected









println("Testing Johansen Trace Test:")
println("============================")
trace_stats, eigenvals = johansen_trace_test(df, end_vars, 2)
println("Trace Statistics: $trace_stats")
println("Eigenvalues: $eigenvals")
#%%
println("Testing Frequentist VAR Estimation, Priors, and Bayesian Estimation:")
println("======================================================================")
est = estimate_var(df, end_vars, 2)
println("β̂:")
display(est.β_hat)
#%%
# A conjugate family: sample_posterior draws directly and i.i.d. from the
# closed-form Normal-Wishart posterior, no MCMC involved.
nw_prior = build_prior(df, end_vars, est, :normal_wishart)
nw_draws = sample_posterior(nw_prior, est; ndraws = 500)
println("\nNormalWishartPrior posterior mean of β (500 direct draws):")
display(sum(nw_draws.β) / length(nw_draws.β))
#%%
# The one family with no closed form: sample_posterior instead runs a
# Turing Gibbs/GibbsConditional sampler, but is called exactly the same way.
niw_prior = build_prior(
    df,
    end_vars,
    est,
    :independent_niw;
    hyperparameter_method = :fixed,
    hyperparameters = (λ1 = 0.2, λ2 = 0.5, λ3 = 1.0, λ4 = 1e5),
)
niw_draws = sample_posterior(niw_prior, est; ndraws = 500)
println("\nIndependentNIWPrior posterior mean of β (500 Gibbs draws):")
display(sum(niw_draws.β) / length(niw_draws.β))
