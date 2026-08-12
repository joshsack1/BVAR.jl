# Create a struct for the frequentist VAR estimation (stage 4)
struct VARestimate{T<:Real}
    β_hat::Matrix{T}
    Σ::Matrix{T}
    se::Matrix{T}
    XᵀX::Matrix{T}
    obs::Int
    lags::Int
    vars::Int
    names::Vector{Symbol}
    include_constant::Bool
end
# Fit the reduced-form VAR by stacked OLS or equation by equation with FixedEffectModels
"""
    estimate_var(
        df::DataFrame,
        end_vec::Vector{Symbol},
        lags::Int;
        include_constant = true,
        method = :ols,
    )

Fits the reduced-form VAR(p)

``Y_t = c + \\Phi_1 Y_{t-1} + \\cdots + \\Phi_p Y_{t-p} + \\varepsilon_t``

and returns a `VARestimate` holding the coefficient matrix ``\\hat{\\beta}``,
the maximum likelihood residual covariance

``\\hat{\\Sigma} = \\frac{\\varepsilon'\\varepsilon}{T}``

the per-equation OLS standard errors

``se_{ij} = \\sqrt{\\hat{\\sigma}_j^2 \\left[(X'X)^{-1}\\right]_{ii}}, \\quad \\hat{\\sigma}_j^2 = \\frac{\\varepsilon_j'\\varepsilon_j}{T - k}``

and the Gram matrix ``X'X`` needed to build priors in the Bayesian stage.
The default `method = :ols` solves all equations in one stacked least
squares problem via `ols_var`; `method = :fem` fits equation by equation
with `FixedEffectModels.reg`. The two agree to numerical precision, and
`:ols` is roughly thirty times faster. The rows of ``\\hat{\\beta}`` are
ordered as the constant (if present), followed by lag 1 of every variable
in `end_vec`, then lag 2, and so on.
"""
function estimate_var(
    df::DataFrame,
    end_vec::Vector{Symbol},
    lags::Int;
    include_constant = true,
    method = :ols,
)
    @assert method in (:ols, :fem) "method must be :ols or :fem"
    @assert all(String.(end_vec) .∈ Ref(names(df))) "All endogenous variables must be columns of the dataframe"
    Y = get_endogenous(df, end_vec)
    @assert all(y -> !ismissing(y) && isfinite(y), Y) "Endogenous data cannot contain missing or non-finite values"
    T_total = size(Y, 1)
    @assert lags > 0 "Need a Positive Number of Lags"
    @assert T_total > lags "Cannot have more lags than obervations"
    if method == :ols
        return fit_var_ols(Y, end_vec, lags, include_constant)
    else
        return fit_var_fem(Y, end_vec, lags, include_constant)
    end
end

"""
    fit_var_ols(
        Y::AbstractMatrix,
        end_vec::Vector{Symbol},
        lags::Int,
        include_constant::Bool,
    )

Fits the VAR(p) with the stacked least squares of `ols_var` and adds the
per-equation OLS standard errors

``se_{ij} = \\sqrt{\\hat{\\sigma}_j^2 \\left[(X'X)^{-1}\\right]_{ii}}, \\quad \\hat{\\sigma}_j^2 = \\frac{\\varepsilon_j'\\varepsilon_j}{T - k}``

returning a `VARestimate`. Internal; called by `estimate_var` for the
default `method = :ols`.
"""
function fit_var_ols(
    Y::AbstractMatrix,
    end_vec::Vector{Symbol},
    lags::Int,
    include_constant::Bool,
)
    n = size(Y, 2)
    fit = ols_var(Y, lags, include_constant)
    XᵀX = fit.X' * fit.X
    k = size(fit.X, 2)
    σ² = vec(sum(abs2, fit.ε; dims = 1)) / (fit.T_eff - k)
    se = sqrt.(diag(inv(XᵀX)) * σ²')
    @assert all(isfinite, se) "Non-finite standard errors: the regressors are collinear"
    return VARestimate(
        fit.β_hat,
        fit.Σ,
        se,
        XᵀX,
        fit.T_eff,
        lags,
        n,
        end_vec,
        include_constant,
    )
end

"""
    fit_var_fem(
        Y::AbstractMatrix,
        end_vec::Vector{Symbol},
        lags::Int,
        include_constant::Bool,
    )

Fits the VAR(p) equation by equation with `FixedEffectModels.reg`,
returning a `VARestimate`. Internal; called by `estimate_var` for
`method = :fem`.
"""
function fit_var_fem(
    Y::AbstractMatrix,
    end_vec::Vector{Symbol},
    lags::Int,
    include_constant::Bool,
)
    T_total, n = size(Y)
    T_eff = T_total - lags
    # Build the lagged dataframe with no missing rows, so reg drops nothing
    df_lagged = DataFrame()
    for (j, name) in enumerate(end_vec)
        df_lagged[!, name] = Y[(lags + 1):end, j]
    end
    lag_names = Symbol[]
    for lag in 1:lags
        for (j, name) in enumerate(end_vec)
            lag_name = Symbol(name, :_lag, lag)
            df_lagged[!, lag_name] = Y[(lags + 1 - lag):(T_total - lag), j]
            push!(lag_names, lag_name)
        end
    end
    # Build the right-hand side of the formula programmatically
    rhs = (include_constant ? term(1) : term(0)) + sum(term.(lag_names))
    expected_names =
        include_constant ? ["(Intercept)"; String.(lag_names)] : String.(lag_names)
    k = n * lags + (include_constant ? 1 : 0)
    β_hat = Matrix{Float64}(undef, k, n)
    se = Matrix{Float64}(undef, k, n)
    ε = Matrix{Float64}(undef, T_eff, n)
    for (i, y_name) in enumerate(end_vec)
        m = reg(df_lagged, term(y_name) ~ rhs; save = :residuals)
        @assert coefnames(m) == expected_names "Unexpected coefficient ordering from reg()"
        β_hat[:, i] = coef(m)
        se[:, i] = stderror(m)
        ε[:, i] = disallowmissing(residuals(m))
    end
    @assert all(isfinite, se) "Non-finite standard errors: the regressors are collinear"
    Σ = (ε' * ε) / T_eff
    X_lags = Matrix{Float64}(df_lagged[!, lag_names])
    X = include_constant ? hcat(ones(T_eff), X_lags) : X_lags
    XᵀX = X' * X
    return VARestimate(β_hat, Σ, se, XᵀX, T_eff, lags, n, end_vec, include_constant)
end
