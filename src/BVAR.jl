module BVAR

# Define Dependencies
using DataFrames
using HypothesisTests
using LinearAlgebra
using Distributions
using FixedEffectModels
using Statistics

# Include the data testing functions that will actually be used
include("data-testing.jl")
# Include Information Criterion Testing
include("information-criterion.jl")

export adf_tests, johansen_trace_test

export aic, bic, hq, fpe

end # module BVAR
