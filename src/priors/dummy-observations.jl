# Four composable dummy-observation recipes. Each returns a `(Yd, Xd)` block
# in the shape `estimate_var` uses (rows = constant if present, then lag 1 of
# every variable, then lag 2, and so on); `conjugate.jl`'s `normal_wishart_prior`
# stacks the requested subset and converts them into prior moments via
# `dummy_gram`.

"""
    dummy_minnesota(Y::AbstractMatrix, lags::Int, include_constant::Bool; λ1 = 0.2, λ3 = 1.0)

Bańbura, Giannone & Reichlin (2010) dummy-observation implementation of the
Minnesota prior. One dummy row is added per (lag, variable) pair,

``y^{*}_{(\\ell,j)i} = \\begin{cases}\\hat\\sigma_j \\ell^{\\lambda_3}/\\lambda_1 & i=j \\\\ 0 & i \\neq j\\end{cases},
\\qquad x^{*}_{(\\ell,j)} \\text{ places } \\hat\\sigma_j \\ell^{\\lambda_3}/\\lambda_1 \\text{ on regressor } (\\ell,j) \\text{ and } 0 \\text{ elsewhere},``

which, combined through `dummy_gram`, shrinks own lags toward a random walk
and cross lags toward zero, tightening geometrically with the lag. Because
the resulting prior scale is a single matrix ``\\Omega_0`` shared across
equations (a structural requirement of the Normal-Wishart family), this
dummy implementation cannot reproduce the original Minnesota prior's
separate cross-equation tightness ``\\lambda_2`` — see `AsymmetricConjugatePrior`
for a conjugate family that can. Returns `(Yd, Xd)`.
"""
function dummy_minnesota(
    Y::AbstractMatrix{T},
    lags::Int,
    include_constant::Bool;
    λ1 = 0.2,
    λ3 = 1.0,
) where {T<:Real}
    n = size(Y, 2)
    σ_ar = sqrt.(ar_residual_variances(Y, lags))
    k = n * lags + (include_constant ? 1 : 0)
    offset = include_constant ? 1 : 0
    nrows = n * lags
    Yd = zeros(T, nrows, n)
    Xd = zeros(T, nrows, k)
    for ℓ in 1:lags, j in 1:n
        row = (ℓ - 1) * n + j
        weight = σ_ar[j] * T(ℓ)^λ3 / λ1
        Yd[row, j] = weight
        Xd[row, offset + (ℓ - 1) * n + j] = weight
    end
    return Yd, Xd
end

"""
    dummy_sum_of_coefficients(Y::AbstractMatrix, lags::Int, include_constant::Bool; λ_soc = 1.0)

Doan, Litterman & Sims (1984)/Sims (1993) sum-of-coefficients ("single unit
root") dummy prior: one dummy row per variable ``i``,

``y^{*}_{i,i'} = \\dfrac{\\bar y_i}{\\lambda_{soc}}\\,\\mathbb{1}[i'=i],
\\qquad x^{*}_{i}(\\ell,j) = \\dfrac{\\bar y_i}{\\lambda_{soc}}\\,\\mathbb{1}[j=i] \\text{ for every lag } \\ell,``

where ``\\bar y_i`` is the average of variable ``i`` over its first `lags`
observations. This encodes the belief that the sum of own-lag coefficients is
close to one and cross-lag coefficients close to zero — i.e. that each
series, taken alone, is close to a random walk, with no cointegrating
relationship being imposed. Smaller ``\\lambda_{soc}`` imposes the belief more
strongly. Returns `(Yd, Xd)`.
"""
function dummy_sum_of_coefficients(
    Y::AbstractMatrix{T},
    lags::Int,
    include_constant::Bool;
    λ_soc = 1.0,
) where {T<:Real}
    n = size(Y, 2)
    ȳ = vec(mean(Y[1:lags, :]; dims = 1))
    k = n * lags + (include_constant ? 1 : 0)
    offset = include_constant ? 1 : 0
    Yd = zeros(T, n, n)
    Xd = zeros(T, n, k)
    for i in 1:n
        weight = ȳ[i] / λ_soc
        Yd[i, i] = weight
        for ℓ in 1:lags
            Xd[i, offset + (ℓ - 1) * n + i] = weight
        end
    end
    return Yd, Xd
end

"""
    dummy_initial_observation(Y::AbstractMatrix, lags::Int, include_constant::Bool; λ_dio = 1.0)

Sims (1993) dummy-initial-observation ("co-persistence") prior: a single
dummy row

``y^{*} = \\bar y'/\\lambda_{dio}, \\qquad
x^{*} = \\left(1/\\lambda_{dio},\\ \\bar y'/\\lambda_{dio},\\ \\ldots,\\ \\bar y'/\\lambda_{dio}\\right)``

(the constant, if present, followed by ``\\bar y'`` repeated at every lag),
where ``\\bar y`` is the average of the first `lags` observations of every
variable. This encodes a belief that the system as a whole is close to its
own initial mean — a co-persistence belief that complements the per-variable
`dummy_sum_of_coefficients` prior, and also identifies the constant term when
the two are combined (see `dummy_gram`). Smaller ``\\lambda_{dio}`` imposes it
more strongly. Returns `(Yd, Xd)`.
"""
function dummy_initial_observation(
    Y::AbstractMatrix{T},
    lags::Int,
    include_constant::Bool;
    λ_dio = 1.0,
) where {T<:Real}
    n = size(Y, 2)
    ȳ = vec(mean(Y[1:lags, :]; dims = 1))
    k = n * lags + (include_constant ? 1 : 0)
    offset = include_constant ? 1 : 0
    Yd = reshape(ȳ ./ λ_dio, 1, n)
    Xd = zeros(T, 1, k)
    if include_constant
        Xd[1, 1] = one(T) / λ_dio
    end
    for ℓ in 1:lags, j in 1:n
        Xd[1, offset + (ℓ - 1) * n + j] = ȳ[j] / λ_dio
    end
    return Yd, Xd
end

"""
    dummy_long_run(Y::AbstractMatrix, lags::Int, include_constant::Bool, H::AbstractMatrix; λ_lr = 1.0)

Giannone, Lenza & Primiceri (2019) "prior for the long run": generalizes
`dummy_sum_of_coefficients` from single variables to arbitrary long-run
linear combinations. For each column ``h`` of the ``n \\times r`` matrix `H`
(each column a hypothesized long-run/cointegrating combination of the
variables), one dummy row

``y^{*} = \\dfrac{h'\\bar y}{\\lambda_{lr}}\\,h, \\qquad
x^{*}(\\ell,\\cdot) = \\dfrac{h'\\bar y}{\\lambda_{lr}}\\,h \\text{ for every lag } \\ell,``

where ``\\bar y`` is the average of the first `lags` observations. Passing
`H = I(n)` reproduces `dummy_sum_of_coefficients` exactly; passing a
cointegrating vector (e.g. an eigenvector from `johansen_trace_test`) encodes
that particular long-run relationship instead of treating every variable as
an independent random walk. Smaller ``\\lambda_{lr}`` imposes the restriction
more strongly. Returns `(Yd, Xd)`.
"""
function dummy_long_run(
    Y::AbstractMatrix{T},
    lags::Int,
    include_constant::Bool,
    H::AbstractMatrix;
    λ_lr = 1.0,
) where {T<:Real}
    n, r = size(H)
    @assert n == size(Y, 2) "H must have one row per variable"
    ȳ = vec(mean(Y[1:lags, :]; dims = 1))
    k = n * lags + (include_constant ? 1 : 0)
    offset = include_constant ? 1 : 0
    Yd = zeros(T, r, n)
    Xd = zeros(T, r, k)
    for c in 1:r
        h = H[:, c]
        weight = (h' * ȳ) / λ_lr
        Yd[c, :] = weight .* h
        for ℓ in 1:lags
            Xd[c, (offset + (ℓ - 1) * n + 1):(offset + ℓ * n)] = weight .* h
        end
    end
    return Yd, Xd
end
