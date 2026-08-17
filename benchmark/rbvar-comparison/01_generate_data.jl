# Stage 01 of the Julia<->R BVAR comparison: simulate the shared datasets.
#
# Both sides of the comparison must see byte-identical data, so the data is
# simulated once here (in Julia) and written to CSV for R to read back. The
# same files also carry the two things R cannot be trusted to reproduce
# itself:
#
#   * psi_<size>.csv  -- the prior residual-variance scales
#                        `ar_residual_variances(Y, p)`. R's `bv_psi(mode =
#                        "auto")` calls `auto_psi`, which takes
#                        `sqrt(arima(...)$sigma2)` -- a standard *deviation*
#                        where the prior wants a *variance* -- so psi is
#                        always passed to R explicitly.
#   * config.csv      -- the (size, T, n, p, ndraws, ...) grid, so stages 02
#                        and 03 cannot drift out of sync.
#
# Run: julia --project=benchmark/rbvar-comparison \
#            benchmark/rbvar-comparison/01_generate_data.jl

using BayesianVectorAutoregressions: ar_residual_variances
using Printf
using Random

const OUTDIR = joinpath(@__DIR__, "data")

"""
    simulate_data(T_obs, n; rng = Xoshiro(42))

Simulates a stable VAR(1) with ``\\Phi_1 = 0.5 I`` and standard normal
innovations, returning a `T_obs × n` matrix. Identical construction to
`simulate_data` in `benchmark/benchmarks.jl` (zero initial condition, no
intercept in the data-generating process; the estimated models still include
a constant).
"""
function simulate_data(T_obs::Int, n::Int; rng::Random.AbstractRNG = Xoshiro(42))
    Y = zeros(Float64, T_obs, n)
    for t in 2:T_obs
        Y[t, :] = 0.5 * Y[t - 1, :] + randn(rng, n)
    end
    return Y
end

"""
    write_matrix_csv(path, M, header)

Writes `M` to `path` as CSV with the given column `header`. Hand-rolled on
purpose (see the note in `Project.toml`): 17 significant digits round-trips a
`Float64` exactly, so R reads back the same bits Julia wrote.
"""
function write_matrix_csv(path::AbstractString, M::AbstractMatrix, header::Vector{String})
    @assert size(M, 2) == length(header) "header length must match the number of columns"
    open(path, "w") do io
        println(io, join(header, ","))
        for i in axes(M, 1)
            println(io, join((@sprintf("%.17g", M[i, j]) for j in axes(M, 2)), ","))
        end
    end
    return path
end

# (label, T_obs, n, lags). The three sizes span a short/narrow VAR, a long
# medium-width one, and a short/wide one (where k = n*p + 1 is largest
# relative to T).
const SIZES = (
    (label = "S", T_obs = 200, n = 3, lags = 2),
    (label = "M", T_obs = 800, n = 6, lags = 4),
    (label = "L", T_obs = 400, n = 10, lags = 5),
)

# Draw counts shared by both languages.
#   PARITY: 25000 - 5000 burn-in = 20000 stored draws on the R side; Julia
#           draws 20000 i.i.d. for the secondary two-sample check.
#   TIMING: 2000 - 1000 burn-in = 1000 stored draws on the R side, matched by
#           ndraws = 1000 in Julia.
const N_DRAW_PARITY = 25_000
const N_BURN_PARITY = 5_000
const N_DRAW_TIMING = 2_000
const N_BURN_TIMING = 1_000
const IRF_HORIZON_JULIA = 20            # Julia counts steps *after* impact ...
const IRF_PERIODS = IRF_HORIZON_JULIA + 1  # ... so 21 periods including impact,
                                           # which is R's `bv_irf(horizon=)`.

# Fixed hyperparameters, identical on both sides. λ4 = sqrt(1e7) so that the
# constant's prior variance λ4^2 equals R's `bv_mn(var = 1e7)` default.
const Λ1 = 0.2
const Λ3 = 1.0
const Λ4 = sqrt(1e7)

function main()
    mkpath(OUTDIR)
    config_rows = String[]
    for sz in SIZES
        # One RNG per size, seeded from the size itself, so adding or
        # reordering sizes never perturbs the others' data.
        rng = Xoshiro(42 + 1000 * sz.n + sz.T_obs)
        Y = simulate_data(sz.T_obs, sz.n; rng = rng)
        names = ["y$(j)" for j in 1:sz.n]

        data_path = joinpath(OUTDIR, "data_$(sz.label).csv")
        write_matrix_csv(data_path, Y, names)

        # Prior scales: MLE residual variance of a univariate AR(p) with
        # constant, per variable. This is exactly what the Julia prior uses
        # internally, and what gets handed to R as `bv_psi(mode = .)`.
        psi = ar_residual_variances(Y, sz.lags)
        @assert all(x -> x > 0 && isfinite(x), psi) "non-positive/non-finite psi"
        psi_path = joinpath(OUTDIR, "psi_$(sz.label).csv")
        write_matrix_csv(psi_path, reshape(psi, 1, :), names)

        push!(
            config_rows,
            join(
                (
                    sz.label,
                    sz.T_obs,
                    sz.n,
                    sz.lags,
                    sz.n * sz.lags + 1,          # K, incl. constant
                    sz.T_obs - sz.lags,          # N = T_eff
                    N_DRAW_PARITY,
                    N_BURN_PARITY,
                    N_DRAW_PARITY - N_BURN_PARITY,
                    N_DRAW_TIMING,
                    N_BURN_TIMING,
                    N_DRAW_TIMING - N_BURN_TIMING,
                    IRF_PERIODS,
                    @sprintf("%.17g", Λ1),
                    @sprintf("%.17g", Λ3),
                    @sprintf("%.17g", 2 * Λ3),   # R's alpha
                    @sprintf("%.17g", Λ4^2),     # R's var
                ),
                ",",
            ),
        )
        @printf(
            "wrote %-14s  T=%4d n=%2d p=%d  psi=[%s]\n",
            "data_$(sz.label).csv",
            sz.T_obs,
            sz.n,
            sz.lags,
            join((@sprintf("%.4f", v) for v in psi), ", ")
        )
    end

    open(joinpath(OUTDIR, "config.csv"), "w") do io
        println(
            io,
            "size,T,n,p,K,N,n_draw_parity,n_burn_parity,n_save_parity," *
            "n_draw_timing,n_burn_timing,n_save_timing,irf_periods," *
            "lambda1,lambda3,alpha,var_const",
        )
        for row in config_rows
            println(io, row)
        end
    end
    println("wrote config.csv ($(length(config_rows)) sizes)")
    return nothing
end

main()
