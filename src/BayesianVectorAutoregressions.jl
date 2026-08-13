module BayesianVectorAutoregressions

# Define Dependencies
using DataFrames
using HypothesisTests
using LinearAlgebra
using Distributions
using FixedEffectModels
using Statistics
using Parameters
using SpecialFunctions
using Random
using Turing

# Include the data testing functions that will actually be used
include("data-testing.jl")
# Include Information Criterion Testing
include("information-criterion.jl")
# Include Frequentist VAR Estimation
include("var-estimation.jl")
# Include Bayesian VAR Priors
include("priors/priors.jl")
# Include Bayesian VAR Estimation
include("bayesian-estimation/bayesian-estimation.jl")
# Include Structural Identification and Impulse Response Functions
include("structural-identification/structural-identification.jl")

export adf_tests, johansen_trace_test

export aic, bic, hq, fpe

export estimate_var, VARestimate

export build_prior,
    MinnesotaPrior,
    NormalWishartPrior,
    IndependentNIWPrior,
    AsymmetricConjugatePrior,
    BaumeisterHamiltonPrior

export sample_posterior, BVARdraws

export structural_prior,
    StructuralPrior,
    hamilton_structural_prior,
    HamiltonStructuralPrior,
    sample_structural,
    StructuralDraws,
    det_sign_restriction,
    long_run_sign_restriction

export identify_short_run, identify_sign_restrictions

export impulse_response, IRFdraws, long_run_multiplier

end # module BayesianVectorAutoregressions
