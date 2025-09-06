"""
    aic(Σ::AbstractMatrix{T}, obs::Int, params::Int) where T<:Real

Computes the Akaike Information Criterion Where:

``AIC = \\ln |\\Sigma| + \\frac{2k}{T}``

Where ``\\Sigma`` is the residual covariance matrix, k is the number of parameters,
and T is the number of observations.
"""
function aic(Σ::AbstractMatrix{T}, obs::Int, params::Int) where {T<:Real}
    return logdet(Σ) + (2 * params) / obs
end

"""
    bic(Σ::AbstractMatrix{T}, obs::Int, params::Int) where T<: Real

Computes the Bayesian Information Criterion Where:

``BIC = \\ln |\\Sigma| + \\frac{k \\cdot \\ln(T)}{T}``

Where ``\\Sigma`` is the residual covariance matrix, k is the number of parameters,
and T is the number of observations.
"""
function bic(Σ::AbstractMatrix{T}, obs::Int, params::Int) where {T<:Real}
    return logdet(Σ) + (params * log(obs)) / obs
end

"""
    hq(Σ::AbstractMatrix{T}, obs::Int, params::Int) where T<: Real

Computes the Hanna-Quinn Information Criterion Where

``HQ = \\ln |\\Sigma| + \\frac{2k * \\ln(\\ln(T))}{T}``

Where ``\\Sigma`` is the residual covariance matrix, k is the number of parameters,
and T is the number of observations.
"""
function hq(Σ::AbstractMatrix{T}, obs::Int, params::Int) where {T<:Real}
    return logdet(Σ) + (n * params * log(log(obs))) / obs
end

"""
    fpe(Σ::AbstractMatrix{T}, obs::Int, params::Int, vars::Int) where T<: Real

Computes Final Prediction Error Where:

``FPE = \\left(\\frac{T+k}{T-k}\\right)^n \\cdot |\\Sigma|``

Where ``\\Sigma`` is the residual covariance matrix, k is the number of parameters,
T is the number of observations, and n is the number of variables.
"""
function fpe(Σ::AbstractMatrix{T}, obs::Int, params::Int, vars::Int) where {T<: Real}
    scalling_factor = ((obs + params) / (obs - params))^vars 
    return scalling_factor * det(Σ)
end
