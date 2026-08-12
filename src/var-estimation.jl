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
# Fit the reduced-form VAR equation by equation with FixedEffectModels
"""
    estimate_var(
        df::DataFrame,
        end_vec::Vector{Symbol},
        lags::Int;
        include_constant = true,
    )

Fits the reduced-form VAR(p)

``Y_t = c + \\Phi_1 Y_{t-1} + \\cdots + \\Phi_p Y_{t-p} + \\varepsilon_t``

equation by equation with `FixedEffectModels.reg`, and returns a `VARestimate`
holding the coefficient matrix ``\\hat{\\beta}``, the maximum likelihood
residual covariance

``\\hat{\\Sigma} = \\frac{\\varepsilon'\\varepsilon}{T}``

the per-equation OLS standard errors, and the Gram matrix ``X'X`` needed to
build priors in the Bayesian stage. The rows of ``\\hat{\\beta}`` are ordered
as the constant (if present), followed by lag 1 of every variable in
`end_vec`, then lag 2, and so on — matching the reference `ols_var`.
"""
function estimate_var(
    df::DataFrame,
    end_vec::Vector{Symbol},
    lags::Int;
    include_constant = true,
)
    @assert all(String.(end_vec) .∈ Ref(names(df))) "All endogenous variables must be columns of the dataframe"
    Y = get_endogenous(df, end_vec)
    @assert all(y -> !ismissing(y) && isfinite(y), Y) "Endogenous data cannot contain missing or non-finite values"
    T_total, n = size(Y)
    @assert lags > 0 "Need a Positive Number of Lags"
    @assert T_total > lags "Cannot have more lags than obervations"
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
