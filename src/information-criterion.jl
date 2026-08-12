# Create a struct for a VAR result
struct VARresult{T<:Real}
    Σ::AbstractMatrix{T}
    obs::Int
    params::Int
    vars::Int
end
# Hand-rolled OLS fit of a VAR(p), the reference implementation for the estimation stage
"""
    ols_var(
        data::AbstractMatrix{T},
        lags::Int,
        has_constant::Bool = true,
    ) where {T<:Real}

Reference OLS fit of a VAR(p) by stacked least squares:

``\\hat{\\beta} = (X'X)^{-1} X'Y, \\quad \\hat{\\Sigma} = \\frac{\\varepsilon'\\varepsilon}{T}``

Where the columns of X are ordered as the constant (if present), followed by
lag 1 of every variable, then lag 2, and so on. Returns a named tuple
`(β_hat, Σ, ε, X, T_eff)`. This is the internal reference implementation;
the production estimator is `estimate_var`.
"""
function ols_var(
    data::AbstractMatrix{T},
    lags::Int,
    has_constant::Bool = true,
) where {T<:Real}
    obs, vars = size(data)
    @assert lags > 0 "Need a Positive Number of Lags"
    @assert obs > lags "Cannot have more lags than obervations"
    T_eff = obs - lags
    current_period = data[(lags + 1):end, :]
    regressors = Vector{Matrix{T}}()
    if has_constant
        push!(regressors, ones(T, T_eff, 1))
    end
    for lag in 1:lags
        lagged_column = data[(lags + 1 - lag):(obs - lag), :]
        push!(regressors, lagged_column)
    end
    X = reduce(hcat, regressors)
    β_hat = (X' * X) \ (X' * current_period)
    ε = current_period - X * β_hat
    Σ = (ε' * ε) / T_eff
    return (β_hat = β_hat, Σ = Σ, ε = ε, X = X, T_eff = T_eff)
end
# Create a function to estimate the VAR results necessary for the information criterion to be calculated
"""
    generate_VARresult(
        data::AbstractMatrix{T},
        lags::Int,
        has_constant::Bool=true,
    ) where {T<:Real}

This function will take in a matrix of the data (perhaps created with the `get_endogenous` function),
the number of lags, and a boolean as to whether the data has a constant (defaulting to true),
and use it to return an object of the type `VARresult`, which has only the information
necessary to calculate the information criterion: the residual covariance matrix, the number of
observations, the number of parameters, and the number of variables.
"""
function generate_VARresult(
    data::AbstractMatrix{T},
    lags::Int,
    has_constant::Bool = true,
) where {T<:Real}
    vars = size(data, 2)
    fit = ols_var(data, lags, has_constant)
    params = size(fit.X, 2) * vars
    return VARresult(fit.Σ, fit.T_eff, params, vars)
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

Computes the Hannan-Quinn Information Criterion Where

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
