# Create a struct for a VAR result
struct VARresult{T<:Real}
    Σ::AbstractMatrix{T}
    obs::Int64
    params::Int64
    vars::Int64
end
# Create a function to estimate the VAR results necessary for the information criterion to be calculated
function generate_VARresult(
    data::AbstractMatrix{T},
    lags::Int64,
    has_constant::Bool = true,
) where {T<:Real}
    obs, vars = size(data)
    @assert lags > 0 "Need a Positive Number of Lags"
    @assert obs > lags "Cannot have more lags than obervations"
    effective_observations = obs - lags
    current_period = data[(lags + 1):end, :]
    regressors = Vector{Matrix{T}}()
    if has_constant
        push!(regressors, ones(T, effective_observations, 1))
    end
    for lag in lags
        lagged_column = data[(lags + 1 - lag):(obs - lag), :]
        push!(regressors, lagged_column)
    end
    X = reduce(hcat, regressors)
    β_hat_matrix = (X' * X) \ (X' * current_period)
    predictions = X * β_hat_matrix
    ε = current_period - predictions
    Σ = (ε' * ε) / effective_observations
    params = size(X, 2) * vars
    return VARresult(Σ, obs, params, vars)
end
# Create functions for the information criterion
"""
    aic(result::VARresult{T}) where T<:Real

Computes the Akaike Information Criterion Where:

``AIC = \\ln |\\Sigma| + \\frac{2k}{T}``

Where ``\\Sigma`` is the residual covariance matrix, k is the number of parameters,
and T is the number of observations.
"""
function aic(result::VARresult{T}) where {T<:Real}
    @unpack Σ, params, obs = result
    return logdet(Σ) + (2 * params) / obs
end

"""
    bic(result::VARresult{T}) where T<: Real

Computes the Bayesian Information Criterion Where:

``BIC = \\ln |\\Sigma| + \\frac{k \\cdot \\ln(T)}{T}``

Where ``\\Sigma`` is the residual covariance matrix, k is the number of parameters,
and T is the number of observations.
"""
function bic(result::VARresult{T}) where {T<:Real}
    @unpack Σ, params, obs = result
    return logdet(Σ) + (params * log(obs)) / obs
end

"""
    hq(result::VARresult{T}) where T<: Real

Computes the Hanna-Quinn Information Criterion Where

``HQ = \\ln |\\Sigma| + \\frac{2k * \\ln(\\ln(T))}{T}``

Where ``\\Sigma`` is the residual covariance matrix, k is the number of parameters,
and T is the number of observations.
"""
function hq(result::VARresult{T}) where {T<:Real}
    @unpack Σ, params, obs = result
    return logdet(Σ) + (2 * params * log(log(obs))) / obs
end

"""
    fpe(result::VARresult{T}) where T<: Real

Computes Final Prediction Error Where:

``FPE = \\left(\\frac{T+k}{T-k}\\right)^n \\cdot |\\Sigma|``

Where ``\\Sigma`` is the residual covariance matrix, k is the number of parameters,
T is the number of observations, and n is the number of variables.
"""
function fpe(result::VARresult{T}) where {T<:Real}
    @unpack Σ, params, obs, vars = result
    scalling_factor = ((obs + params) / (obs - params))^vars
    return scalling_factor * det(Σ)
end
