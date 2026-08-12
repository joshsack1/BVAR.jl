module BVAR

# Define Dependencies
using DataFrames
using HypothesisTests
using LinearAlgebra
using Distributions
using FixedEffectModels
using Statistics
using Parameters

# Include the data testing functions that will actually be used
include("data-testing.jl")
# Include Information Criterion Testing
include("information-criterion.jl")
# Include Frequentist VAR Estimation
include("var-estimation.jl")

export adf_tests, johansen_trace_test

export aic, bic, hq, fpe

export estimate_var, VARestimate

end # module BVAR
