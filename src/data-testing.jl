"""
    get_endogenous(df::DataFrame, end_vec::Vector{Symbol})

This function will ask for the variables in the dataframe that are endogenous.
It is a helper function to make the other tests faster computationally.
The vector `end_vec` should be a vector of the column names of all of the
endogenous variables.
"""
function get_endogenous(df::DataFrame, end_vec::Vector{Symbol})
    return Matrix([df[!, end_var] for end_var in end_vec])
end

"""
    adf_tests(df::DataFrame, end_vec::Vector{Symbol}

For each endogenous variable the dataframe provided, this function will return
a type 1, type 2, and a type 4 ADF test in a vector.
"""
function adf_tests(df::DataFrame, end_vec::Vector{Symbol})
    end_vars = [df[!, end_var] for end_var in end_vec]
    adf_types = [:none, :constant, :trend]
    return [ADFTest.(Ref(end_var), adf_types, Ref(1)) for end_var in end_vars]
end

"""
    johansen_trace_test(df::DataFrame, end_vec::Vector{Symbol}, p::Int; include_constant=true)

This function will calculate the Johansen Trace Test for cointegration
on the endogenous variables in your dataframe, returning the trace statistics
eigenvalues.
"""
function johansen_trace_test(
    df::DataFrame,
    end_vec::Vector{Symbol},
    p::Int;
    include_constant = true,
)
    Y = get_endogenous(df, end_vec)
    T, n = size(Y)
    # Create lagged differences and levels
    ΔY = diff(Y; dims = 1)
    Y_lag = Y[p:(end - 1), :]
    # Lagged differences for the VAR Part
    ΔY_lags = zeros(T - p, n * (p - 1))
    for i in 1:(p - 1)
        ΔY_lags[:, ((i - 1) * n + 1):(i * n)] = ΔY[(p - i):(end - i), :]
    end
    if include_constant
        X = hcat(ones(T - p), ΔY_lags)
    else
        X = ΔY_lags
    end
    # Use a projection matrix to get the residuals
    if size(X, 2) > 0
        P = I - X * inv(X'X) * X'
        R0 = P * ΔY[p:end, :]
        R1 = P * Y_lag
    else
        R0 = ΔY[p:end, :]
        R1 = Y_lag
    end
    # Correlation Analysis
    S00 = R0'R0 / (T - p)
    S01 = R0'R1 / (T - p)
    S11 = R1'R1 / (T - p)
    # Eigenvalues
    eigenvals = eigvals(inv(S11) * S01' * inv(S00) * S01)
    eigenvals = sort(real(eigenvals; rev = true))
    # Trace Statistics
    trace_stats = -T * cumsum(log.(1 .- eigenvals))
    return trace_stats, eigenvals
end
