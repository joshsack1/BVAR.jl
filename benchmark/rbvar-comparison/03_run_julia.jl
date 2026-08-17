# Stage 03 of the Julia<->R BVAR comparison: the Julia side plus the verdict.
#
# For each dataset written by 01_generate_data.jl this script
#
#   PARITY    rebuilds the same Normal-Wishart prior R was given, takes the
#             CLOSED-FORM posterior moments (no sampling, no Monte Carlo error
#             on this side), and asks whether R's 20000-draw posterior means
#             are within Monte Carlo noise of them:
#
#                 z = |mean_R - analytic| / (sd_R / sqrt(n_save))
#
#             PASS iff max|z| < 5 over every element of beta (K x M) and
#             sigma (M x M). Using the analytic value as the reference means a
#             failure can only come from a genuine model mismatch, not from
#             two noisy estimates disagreeing.
#   SECONDARY a two-sample z between R's means and Julia's own 20000-draw
#             Monte Carlo means, which additionally exercises Julia's sampler
#             (`sample_posterior`) rather than only its posterior algebra.
#   TIMING    BenchmarkTools medians for 1000 posterior draws and for
#             Cholesky-identified IRFs over 21 periods on 1000 draws.
#
# The closed-form moments used:
#   E[beta]      = beta_bar
#   E[Sigma]     = S_bar / (nu_bar - n - 1)
#   Var(beta_ij) = Omega_bar_ii * S_bar_jj / (nu_bar - n - 1)
#       [law of total variance: beta | Sigma ~ MN(beta_bar, Omega_bar, Sigma)
#        gives Var(beta_ij | Sigma) = Omega_bar_ii * Sigma_jj and
#        E[beta | Sigma] is constant, so Var = Omega_bar_ii * E[Sigma_jj].]
#   Var(Sigma_ij) = ((d + 2) S_ij^2 + d S_ii S_jj) / ((d + 1) d^2 (d - 2)),
#        d = nu_bar - n - 1   [inverse-Wishart element variance; reduces to
#        the familiar 2 S_ii^2 / (d^2 (d - 2)) on the diagonal.]
# The two analytic sds are NOT used in the z statistic -- only as a sanity
# check that R's Monte Carlo sds (the z denominator) are the right size.
#
# Run: julia --project=benchmark/rbvar-comparison \
#            benchmark/rbvar-comparison/03_run_julia.jl

using BayesianVectorAutoregressions
using BayesianVectorAutoregressions: ar_residual_variances, gram_blocks,
                                     normal_wishart_posterior
using BenchmarkTools
using DataFrames
using LinearAlgebra
using Printf
using Random
using Statistics

BLAS.set_num_threads(1)

const DATA_DIR = joinpath(@__DIR__, "data")
const RESULTS_DIR = joinpath(@__DIR__, "results")

const Z_GATE = 5.0            # PASS iff max|z| < Z_GATE
const N_MC_JULIA = 20_000     # Julia draws for the secondary two-sample check
const MC_SEED = 20260817

# --- minimal CSV readers ----------------------------------------------------

"""
    read_matrix_csv(path) -> (M, header)

Reads a numeric CSV with one header line into a `Matrix{Float64}`.
"""
function read_matrix_csv(path::AbstractString)
    lines = filter(!isempty, strip.(readlines(path)))
    header = String.(split(lines[1], ','))
    rows = [parse.(Float64, split(l, ',')) for l in lines[2:end]]
    M = Matrix{Float64}(undef, length(rows), length(header))
    for (i, row) in enumerate(rows)
        @assert length(row) == length(header) "ragged row $i in $path"
        M[i, :] = row
    end
    return M, header
end

"""
    read_keyed_csv(path) -> Dict{String,String}

Reads a two-column `key,value` CSV (stage 02's `r_hyper_<size>.csv`).
"""
function read_keyed_csv(path::AbstractString)
    lines = filter(!isempty, strip.(readlines(path)))
    d = Dict{String,String}()
    for l in lines[2:end]
        i = findfirst(',', l)
        d[String(l[1:(i - 1)])] = String(l[(i + 1):end])
    end
    return d
end

"""
    read_config(path) -> Vector{Dict{String,String}}

Reads `data/config.csv` as one dictionary per row.
"""
function read_config(path::AbstractString)
    lines = filter(!isempty, strip.(readlines(path)))
    header = String.(split(lines[1], ','))
    return [Dict(zip(header, String.(split(l, ',')))) for l in lines[2:end]]
end

"""
    read_timings(path) -> Dict{String,Dict{String,Float64}}

Reads `results/r_timings.csv` (first column is the size label, the rest
numeric) into `size => (column => value)`.
"""
function read_timings(path::AbstractString)
    lines = filter(!isempty, strip.(readlines(path)))
    header = String.(split(lines[1], ','))
    out = Dict{String,Dict{String,Float64}}()
    for l in lines[2:end]
        fields = String.(split(l, ','))
        @assert length(fields) == length(header) "ragged row in $path"
        out[fields[1]] = Dict(zip(header[2:end], parse.(Float64, fields[2:end])))
    end
    return out
end

# --- analytic moments -------------------------------------------------------

"""
    beta_sd_analytic(Ω̄, S̄, ν̄, n)

Element-wise prior-to-posterior marginal sd of ``\\beta`` under the
matrix-t marginal implied by ``\\beta\\mid\\Sigma\\sim MN(\\bar\\beta,\\bar\\Omega,\\Sigma)``,
``\\Sigma\\sim IW(\\bar S,\\bar\\nu)``.
"""
function beta_sd_analytic(Ω̄::AbstractMatrix, S̄::AbstractMatrix, ν̄::Real, n::Int)
    d = ν̄ - n - 1
    @assert d > 0 "E[Sigma] undefined: nu_bar - n - 1 = $d"
    return sqrt.([Ω̄[i, i] * S̄[j, j] / d for i in axes(Ω̄, 1), j in axes(S̄, 1)])
end

"""
    sigma_sd_analytic(S̄, ν̄, n)

Element-wise sd of ``\\Sigma\\sim IW(\\bar S,\\bar\\nu)``.
"""
function sigma_sd_analytic(S̄::AbstractMatrix, ν̄::Real, n::Int)
    d = ν̄ - n - 1
    @assert d > 2 "IW element variance undefined: nu_bar - n - 3 = $(d - 2)"
    return sqrt.([
        ((d + 2) * S̄[i, j]^2 + d * S̄[i, i] * S̄[j, j]) / ((d + 1) * d^2 * (d - 2)) for
        i in axes(S̄, 1), j in axes(S̄, 2)
    ])
end

# --- per-size work ----------------------------------------------------------

struct SizeResult
    size::String
    T::Int
    n::Int
    p::Int
    K::Int
    n_save::Int
    # parity vs analytic
    beta_max_z::Float64
    beta_max_absdev::Float64
    sigma_max_z::Float64
    sigma_max_absdev::Float64
    pass::Bool
    # secondary two-sample (R MC vs Julia MC)
    beta_max_z2::Float64
    sigma_max_z2::Float64
    pass2::Bool
    # sd sanity (R MC sd / analytic sd)
    beta_sd_ratio_min::Float64
    beta_sd_ratio_max::Float64
    sigma_sd_ratio_min::Float64
    sigma_sd_ratio_max::Float64
    # lambda pin as realised by R
    lambda_mean::Float64
    lambda_sd::Float64
    mh_accept::Float64
    # timings, seconds
    jl_draws_s::Float64
    jl_irf_s::Float64
    r_bvar_s::Float64
    r_irf_core_s::Float64
    r_irf_full_s::Float64
end

function run_size(cfg::Dict{String,String}, rtim::Dict{String,Dict{String,Float64}})
    size_label = cfg["size"]
    n = parse(Int, cfg["n"])
    p = parse(Int, cfg["p"])
    T_obs = parse(Int, cfg["T"])
    K = parse(Int, cfg["K"])
    n_save = parse(Int, cfg["n_save_parity"])
    λ1 = parse(Float64, cfg["lambda1"])
    λ3 = parse(Float64, cfg["lambda3"])
    λ4 = sqrt(parse(Float64, cfg["var_const"]))
    irf_periods = parse(Int, cfg["irf_periods"])
    horizon = irf_periods - 1          # Julia counts steps after impact

    println("=== size $size_label (T=$T_obs, n=$n, p=$p, K=$K) ===")

    Y, names_str = read_matrix_csv(joinpath(DATA_DIR, "data_$(size_label).csv"))
    @assert size(Y) == (T_obs, n) "data_$(size_label).csv has unexpected shape $(size(Y))"
    end_vec = Symbol.(names_str)
    df = DataFrame(Y, end_vec)

    est = estimate_var(df, end_vec, p; include_constant = true, method = :ols)
    prior = build_prior(
        df,
        end_vec,
        est,
        :normal_wishart;
        hyperparameter_method = :fixed,
        hyperparameters = (
            λ1 = λ1,
            λ3 = λ3,
            λ4 = λ4,
            λ_soc = 1.0,
            λ_dio = 1.0,
            λ_lr = 1.0,
        ),
    )

    # The prior scales R was handed must be exactly the ones the Julia prior
    # built for itself; S0 = diag(sigma_hat^2) because nu0 = n + 2.
    psi_file, _ = read_matrix_csv(joinpath(DATA_DIR, "psi_$(size_label).csv"))
    psi = vec(psi_file)
    @assert isapprox(psi, ar_residual_variances(Y, p); rtol = 1e-12) "psi_$(size_label).csv does not match ar_residual_variances"
    @assert isapprox(diag(prior.S0), psi; rtol = 1e-12) "prior S0 diagonal != psi (nu0 convention changed?)"
    @assert isapprox(prior.Ω0[1, 1], λ4^2; rtol = 1e-12) "constant prior variance != lambda4^2"

    gram = gram_blocks(est)
    post = normal_wishart_posterior(prior, gram.XᵀX, gram.XᵀY, gram.YᵀY, est.obs)
    @assert size(post.β̄) == (K, n) "beta_bar is $(size(post.β̄)), expected ($K, $n)"

    β_analytic = Matrix(post.β̄)
    Σ_analytic = Matrix(post.S̄) ./ (post.ν̄ - n - 1)
    β_sd_a = beta_sd_analytic(post.Ω̄, post.S̄, post.ν̄, n)
    Σ_sd_a = sigma_sd_analytic(post.S̄, post.ν̄, n)

    # --- R output ---
    hyp = read_keyed_csv(joinpath(RESULTS_DIR, "r_hyper_$(size_label).csv"))
    @assert hyp["beta_dim_order"] == "n_save x K x M" "unexpected R beta layout"
    @assert hyp["beta_dim"] == "$(n_save)x$(K)x$(n)" "R beta dims $(hyp["beta_dim"]) != $(n_save)x$(K)x$(n)"
    expected_expl = vcat(
        "constant",
        ["$(v)-lag$(ℓ)" for ℓ in 1:p for v in names_str],
    )
    @assert String.(split(hyp["explanatories"], '|')) == expected_expl "R's design-matrix column order differs from Julia's [const, lag1 all vars, lag2 all vars, ...]"
    λ_mean = parse(Float64, hyp["lambda_mean"])
    λ_sd = parse(Float64, hyp["lambda_sd"])
    mh_acc = parse(Float64, hyp["mh_accept_rate"])
    @assert λ_sd < 1e-4 "lambda pin failed: sd = $λ_sd"
    @assert mh_acc > 0 "MH acceptance rate is zero"
    @assert isapprox(λ_mean, λ1; atol = 1e-6) "R's mean lambda $λ_mean is not pinned at $λ1"

    β_R_mean, _ = read_matrix_csv(joinpath(RESULTS_DIR, "r_beta_mean_$(size_label).csv"))
    β_R_sd, _ = read_matrix_csv(joinpath(RESULTS_DIR, "r_beta_sd_$(size_label).csv"))
    Σ_R_mean, _ = read_matrix_csv(joinpath(RESULTS_DIR, "r_sigma_mean_$(size_label).csv"))
    Σ_R_sd, _ = read_matrix_csv(joinpath(RESULTS_DIR, "r_sigma_sd_$(size_label).csv"))

    # --- PARITY: R Monte Carlo mean vs Julia analytic mean ---
    se = sqrt(n_save)
    β_z = abs.(β_R_mean .- β_analytic) ./ (β_R_sd ./ se)
    Σ_z = abs.(Σ_R_mean .- Σ_analytic) ./ (Σ_R_sd ./ se)
    β_max_z, β_max_dev = maximum(β_z), maximum(abs.(β_R_mean .- β_analytic))
    Σ_max_z, Σ_max_dev = maximum(Σ_z), maximum(abs.(Σ_R_mean .- Σ_analytic))
    pass = β_max_z < Z_GATE && Σ_max_z < Z_GATE

    β_ratio = β_R_sd ./ β_sd_a
    Σ_ratio = Σ_R_sd ./ Σ_sd_a

    @printf("  beta : max|z| = %8.3f   max|dev| = %.3e   (sd_R/sd_analytic in [%.4f, %.4f])\n",
        β_max_z, β_max_dev, minimum(β_ratio), maximum(β_ratio))
    @printf("  sigma: max|z| = %8.3f   max|dev| = %.3e   (sd_R/sd_analytic in [%.4f, %.4f])\n",
        Σ_max_z, Σ_max_dev, minimum(Σ_ratio), maximum(Σ_ratio))
    println("  parity vs analytic: ", pass ? "PASS" : "FAIL")

    # --- SECONDARY: R Monte Carlo vs Julia Monte Carlo, two-sample z ---
    mc = sample_posterior(prior, est; ndraws = N_MC_JULIA, rng = Xoshiro(MC_SEED))
    β_stack = reduce(hcat, (vec(b) for b in mc.β))      # (K*n) x N_MC
    Σ_stack = reduce(hcat, (vec(s) for s in mc.Σ))      # (n*n) x N_MC
    β_J_mean = reshape(vec(mean(β_stack; dims = 2)), K, n)
    β_J_sd = reshape(vec(std(β_stack; dims = 2)), K, n)
    Σ_J_mean = reshape(vec(mean(Σ_stack; dims = 2)), n, n)
    Σ_J_sd = reshape(vec(std(Σ_stack; dims = 2)), n, n)
    β_z2 = abs.(β_R_mean .- β_J_mean) ./ sqrt.(β_R_sd .^ 2 ./ n_save .+ β_J_sd .^ 2 ./ N_MC_JULIA)
    Σ_z2 = abs.(Σ_R_mean .- Σ_J_mean) ./ sqrt.(Σ_R_sd .^ 2 ./ n_save .+ Σ_J_sd .^ 2 ./ N_MC_JULIA)
    β_max_z2, Σ_max_z2 = maximum(β_z2), maximum(Σ_z2)
    pass2 = β_max_z2 < Z_GATE && Σ_max_z2 < Z_GATE
    @printf("  two-sample (R MC vs Julia MC): beta max|z| = %.3f  sigma max|z| = %.3f  %s\n",
        β_max_z2, Σ_max_z2, pass2 ? "PASS" : "FAIL")
    mc = nothing
    β_stack = Σ_stack = nothing
    GC.gc()

    # --- TIMING ---
    n_time = parse(Int, cfg["n_save_timing"])
    b_draws = @benchmark sample_posterior($prior, $est; ndraws = $n_time) samples = 7 seconds =
        120 evals = 1
    draws_1000 = sample_posterior(prior, est; ndraws = n_time, rng = Xoshiro(MC_SEED))
    b_irf = @benchmark identify_short_run($draws_1000; horizon = $horizon) samples = 7 seconds =
        120 evals = 1
    irfs = identify_short_run(draws_1000; horizon = horizon)
    @assert length(irfs.H) == n_time && length(irfs.H[1]) == irf_periods "Julia IRF shape: $(length(irfs.H)) draws x $(length(irfs.H[1])) periods, expected $n_time x $irf_periods"
    jl_draws_s = median(b_draws).time / 1e9
    jl_irf_s = median(b_irf).time / 1e9
    tt = rtim[size_label]
    @printf("  timing (s): julia draws=%.4f  julia irf=%.4f | R bvar=%.4f  R irf_core=%.4f  R irf()=%.4f\n",
        jl_draws_s, jl_irf_s, tt["r_bvar_median_s"], tt["r_irf_core_median_s"],
        tt["r_irf_full_median_s"])
    draws_1000 = nothing
    irfs = nothing
    GC.gc()

    return SizeResult(
        size_label, T_obs, n, p, K, n_save,
        β_max_z, β_max_dev, Σ_max_z, Σ_max_dev, pass,
        β_max_z2, Σ_max_z2, pass2,
        minimum(β_ratio), maximum(β_ratio), minimum(Σ_ratio), maximum(Σ_ratio),
        λ_mean, λ_sd, mh_acc,
        jl_draws_s, jl_irf_s,
        tt["r_bvar_median_s"], tt["r_irf_core_median_s"], tt["r_irf_full_median_s"],
    )
end

# --- output -----------------------------------------------------------------

const SUMMARY_COLS = [
    "size", "T", "n", "p", "K", "n_save",
    "beta_max_abs_z", "beta_max_abs_dev", "sigma_max_abs_z", "sigma_max_abs_dev",
    "parity_pass",
    "beta_max_abs_z_two_sample", "sigma_max_abs_z_two_sample", "two_sample_pass",
    "beta_sd_ratio_min", "beta_sd_ratio_max", "sigma_sd_ratio_min", "sigma_sd_ratio_max",
    "r_lambda_mean", "r_lambda_sd", "r_mh_accept_rate",
    "julia_draws_median_s", "julia_irf_median_s",
    "r_bvar_median_s", "r_irf_core_median_s", "r_irf_full_median_s",
    "ratio_r_over_julia_draws", "ratio_r_over_julia_irf_core",
]

function summary_row(r::SizeResult)
    f(x) = @sprintf("%.10e", x)
    return join(
        (
            r.size, r.T, r.n, r.p, r.K, r.n_save,
            f(r.beta_max_z), f(r.beta_max_absdev), f(r.sigma_max_z), f(r.sigma_max_absdev),
            r.pass ? "PASS" : "FAIL",
            f(r.beta_max_z2), f(r.sigma_max_z2), r.pass2 ? "PASS" : "FAIL",
            f(r.beta_sd_ratio_min), f(r.beta_sd_ratio_max),
            f(r.sigma_sd_ratio_min), f(r.sigma_sd_ratio_max),
            f(r.lambda_mean), f(r.lambda_sd), f(r.mh_accept),
            f(r.jl_draws_s), f(r.jl_irf_s),
            f(r.r_bvar_s), f(r.r_irf_core_s), f(r.r_irf_full_s),
            f(r.r_bvar_s / r.jl_draws_s), f(r.r_irf_core_s / r.jl_irf_s),
        ),
        ",",
    )
end

function print_markdown(results::Vector{SizeResult})
    println()
    println("## Parity (R 20000-draw MC mean vs Julia closed-form posterior)")
    println()
    println("| size | T | n | p | K | beta max\\|z\\| | beta max abs dev | sigma max\\|z\\| | sigma max abs dev | verdict |")
    println("|---|---|---|---|---|---|---|---|---|---|")
    for r in results
        @printf("| %s | %d | %d | %d | %d | %.2f | %.2e | %.2f | %.2e | %s |\n",
            r.size, r.T, r.n, r.p, r.K, r.beta_max_z, r.beta_max_absdev,
            r.sigma_max_z, r.sigma_max_absdev, r.pass ? "**PASS**" : "**FAIL**")
    end
    println()
    println("## Timing (median seconds, single-threaded BLAS)")
    println()
    println("| size | T | n | p | 1000 draws: Julia | 1000 draws: R | ratio | IRF h=21 x 1000: Julia | IRF core: R | ratio | R irf() incl. bands |")
    println("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results
        @printf("| %s | %d | %d | %d | %.4f | %.4f | %.1fx | %.4f | %.4f | %.1fx | %.4f |\n",
            r.size, r.T, r.n, r.p,
            r.jl_draws_s, r.r_bvar_s, r.r_bvar_s / r.jl_draws_s,
            r.jl_irf_s, r.r_irf_core_s, r.r_irf_core_s / r.jl_irf_s,
            r.r_irf_full_s)
    end
    println()
    println("## Hyperparameter pin actually realised by R")
    println()
    println("| size | mean lambda | sd lambda | MH acceptance |")
    println("|---|---|---|---|")
    for r in results
        @printf("| %s | %.12f | %.2e | %.3f |\n", r.size, r.lambda_mean, r.lambda_sd, r.mh_accept)
    end
    println()
    return nothing
end

function write_versions(path::AbstractString)
    cfg = BLAS.get_config()
    open(path, "w") do io
        println(io, "=== Julia ===")
        println(io, "julia version: ", VERSION)
        println(io, "word size: ", Sys.WORD_SIZE, "  cpu threads: ", Sys.CPU_THREADS)
        println(io, "julia threads: ", Threads.nthreads())
        println(io)
        println(io, "=== BLAS ===")
        println(io, "BLAS.get_config():")
        for lib in cfg.loaded_libs
            println(io, "  ", lib.libname, "  interface=", lib.interface)
        end
        println(io, "BLAS.get_num_threads(): ", BLAS.get_num_threads())
        println(io, "LAPACK: ", LinearAlgebra.LAPACK.version())
        println(io)
        println(io, "=== BLAS thread env ===")
        for v in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")
            println(io, v, "=", get(ENV, v, "<unset>"))
        end
        println(io)
        println(io, "=== Hardware / OS ===")
        for (label, cmd) in (
            ("chip", `sysctl -n machdep.cpu.brand_string`),
            ("os", `sw_vers -productVersion`),
            ("build", `sw_vers -buildVersion`),
        )
            out = try
                strip(read(cmd, String))
            catch
                "<unavailable>"
            end
            println(io, label, ": ", out)
        end
    end
    return path
end

function main()
    config = read_config(joinpath(DATA_DIR, "config.csv"))

    rtim = read_timings(joinpath(RESULTS_DIR, "r_timings.csv"))

    results = SizeResult[]
    for cfg in config
        push!(results, run_size(cfg, rtim))
    end

    open(joinpath(RESULTS_DIR, "summary.csv"), "w") do io
        println(io, join(SUMMARY_COLS, ","))
        for r in results
            println(io, summary_row(r))
        end
    end
    println("\nwrote results/summary.csv")
    write_versions(joinpath(RESULTS_DIR, "julia_versions.txt"))
    println("wrote results/julia_versions.txt")

    print_markdown(results)

    all_pass = all(r -> r.pass, results)
    all_pass2 = all(r -> r.pass2, results)
    println("PARITY (analytic gate, max|z| < $Z_GATE): ", all_pass ? "PASS" : "FAIL")
    println("SECONDARY (two-sample, max|z| < $Z_GATE): ", all_pass2 ? "PASS" : "FAIL")
    all_pass || error("parity gate failed; see results/summary.csv")
    return nothing
end

main()
