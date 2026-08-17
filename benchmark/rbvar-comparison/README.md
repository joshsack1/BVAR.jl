# Cross-language comparison: BayesianVectorAutoregressions.jl vs R's `BVAR`

A self-contained harness that checks this package's conjugate Normal-Wishart
BVAR against the reference R implementation — Kuschnig & Vashold's
[`BVAR`](https://cran.r-project.org/package=BVAR) (CRAN 1.0.5; *JSS* 100(14),
2021) — on identical data with an identical prior, and then times both.

It answers two separate questions:

1. **Parity.** Do the two packages describe the *same* posterior? The gate is
   deliberately one-sided: Julia contributes the **closed-form** posterior
   moments (no Monte Carlo error), R contributes 20 000 draws, and the test is
   whether R's draw means sit within Monte Carlo noise of the analytic values,
   `z = |mean_R − analytic| / (sd_R / sqrt(20000))`, with `max|z| < 5` over
   every element of `beta` (K×M) and `sigma` (M×M). A failure therefore
   implies a real modelling difference, not two noisy estimates disagreeing.
   A secondary two-sample z between R's means and Julia's own 20 000-draw
   Monte Carlo means additionally exercises Julia's sampler.
2. **Speed.** Median wall-clock seconds to produce 1 000 stored posterior
   draws, and to compute Cholesky-identified impulse responses over 21
   periods (impact + 20) for 1 000 draws.

## Requirements

- **R** ≥ 4.0 with **`BVAR`** ≥ 1.0.5:
  ```sh
  Rscript -e 'install.packages("BVAR", repos = "https://cloud.r-project.org")'
  ```
- **Julia** ≥ 1.12 (the package's own `[compat]` floor). The environment in
  this directory is instantiated automatically on first run; to do it
  explicitly:
  ```sh
  julia --project=benchmark/rbvar-comparison -e 'using Pkg; Pkg.instantiate()'
  ```

`Project.toml` carries only `BayesianVectorAutoregressions` (as a path
dependency on the repository root), `BenchmarkTools`, `DataFrames`, and four
stdlibs. There is no CSV package on either side: **all CSV I/O here is
hand-rolled** (a few lines of `split`/`join`/`@printf` in Julia,
`readLines`/`formatC` in R). That is deliberate — the harness should measure
the package under test and nothing else, and the environment should resolve
in seconds and not drift.

## How to run

```sh
./benchmark/rbvar-comparison/run.sh
```

The script is the only supported entry point: it pins `OPENBLAS_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS` and `OMP_NUM_THREADS` to `1` (timings between two
languages mean nothing if one of them is quietly using more cores), then runs
the three stages in order. It exits non-zero if the parity gate fails.
Override the interpreters with `RSCRIPT=… JULIA=… ./run.sh`.

Stages, runnable individually if you are iterating:

| stage | what it does |
|---|---|
| `01_generate_data.jl` | simulates the three shared datasets, writes `data/data_*.csv`, the prior scales `data/psi_*.csv`, and `data/config.csv` |
| `02_run_rbvar.R` | R parity run (25 000 draws, 5 000 burn-in) + timing runs; writes `results/r_*` |
| `03_run_julia.jl` | Julia closed-form moments, parity verdict, Julia timings; writes `results/summary.csv` and prints a markdown table |

## The three datasets

Stable VAR(1) with `Phi_1 = 0.5 I` and standard normal innovations — the same
construction as `simulate_data` in `benchmark/benchmarks.jl` — at three sizes,
each from a fixed `Xoshiro` seed so the data is reproducible bit for bit:

| size | T | n | p | K = n·p + 1 |
|---|---|---|---|---|
| S | 200 | 3 | 2 | 7 |
| M | 800 | 6 | 4 | 25 |
| L | 400 | 10 | 5 | 51 |

Both packages build the same design matrix `X = [const, lag 1 of all
variables, lag 2 of all variables, …]` with variables in data-column order,
and stage 03 asserts this by comparing R's `explanatories` names against the
expected order rather than assuming it.

## Matching the priors

The Julia side uses the direct Kadiyala–Karlsson Minnesota Normal-Wishart
prior:

```julia
est   = estimate_var(df, end_vec, p; include_constant = true, method = :ols)
prior = build_prior(df, end_vec, est, :normal_wishart;
                    hyperparameter_method = :fixed,
                    hyperparameters = (λ1 = 0.2, λ3 = 1.0, λ4 = sqrt(1e7),
                                       λ_soc = 1.0, λ_dio = 1.0, λ_lr = 1.0))
```

which gives a diagonal `Ω0` with `λ4²` on the constant and
`(λ1/ℓ^λ3)² / σ̂_j²` on lag ℓ of variable j, a random-walk prior mean `β0`,
`ν0 = n + 2`, and `S0 = diag(σ̂_j²)·(ν0 − n − 1) = diag(σ̂_j²)`.

| Julia | R | value |
|---|---|---|
| `λ1` | `bv_lambda(mode = )` | 0.2 |
| `λ3` | `bv_alpha(mode = )` = 2·λ3 | 2 |
| `λ4²` | `bv_mn(var = )` | 1e7 |
| `β0` random walk | `bv_mn(b = )` | 1 |
| `σ̂_j²` = `ar_residual_variances(Y, p)` | `bv_psi(mode = )` | per dataset |

Two things about that table are load-bearing:

- **`psi` is always passed explicitly.** R's `bv_psi(mode = "auto")` routes
  through `auto_psi`, which sets the mode to `sqrt(arima(x, c(p,0,0))$sigma2)`
  — a standard *deviation* where `Ω0` and `S0` want a *variance*. The
  automatic route would therefore not match any Minnesota prior as usually
  written, so stage 01 computes `ar_residual_variances(Y, p)` and stage 02
  feeds exactly that to R. `bv_psi(mode = <numeric vector>)` needs no
  `min`/`max`: it auto-derives them as `mode/100` and `mode*100`.
- **`lambda` is pinned, not sampled.** `bvar()` refuses to run with an empty
  `hyper` ("Please provide at least one hyperparameter"), so λ is nominally
  sampled but pinned two ways at once:
  `bv_lambda(mode = 0.2, sd = 1e-6, min = 1e-4, max = 5)` makes the gamma
  hyperprior a spike, and `bv_mh(scale_hess = 1e-12)` shrinks the
  Metropolis-Hastings proposal to match. The second half matters: with the
  default `scale_hess = 0.01` the proposal standard deviation is ≈ 0.12
  against a spike prior, so acceptance collapses to ~0.2 % and the chain
  freezes at whatever value it first accepted — typically ~1e-5 away from
  0.2. With both, acceptance is ~1/2 and the stored λ draws have mean 0.2 to
  ~1e-10 with sd ~1e-11. Stage 02 asserts `sd(λ) < 1e-4` and
  `accepted > 0`; stage 03 re-checks the realised pin and additionally
  requires `|mean(λ) − 0.2| < 1e-6`.

Conditional on those hyperparameters the two posteriors are the same
mathematics: R's `draw_post` forms
`S_post = psi + sse + (beta_hat − b)' Ω0⁻¹ (beta_hat − b)` and draws
`Sigma ~ IW(S_post, N + M + 2)` then `beta | Sigma ~ MN(beta_hat, (X'X +
Ω0⁻¹)⁻¹, Sigma)`, which is Julia's `normal_wishart_posterior` with
`ν̄ = ν0 + T_eff = (n + 2) + (T − p)` written differently.

## Impulse responses and the horizon off-by-one

R's `bv_irf(horizon = h)` counts periods **including** impact:
`compute_irf` writes the shock into slice 1 and iterates `2:h`. Julia's
`identify_short_run(draws; horizon = h)` counts steps **after** impact and
returns `h + 1` matrices. The harness therefore pairs
`bv_irf(horizon = 21)` with `identify_short_run(; horizon = 20)`, and both
stages assert the resulting shapes.

R's public `irf()` also computes 16/84 % quantile bands, which
`identify_short_run` does not, so stage 02 times **both**: `irf_core`, the
bare `get_beta_comp` + `compute_irf` loop that is the exact analogue of
`identify_short_run`, and the full `irf()` call. `summary.csv` reports the
`irf_core` ratio as the like-for-like number.

## What it produces

Committed, in `results/`:

| file | contents |
|---|---|
| `summary.csv` | one row per size: parity z statistics, max absolute deviations, PASS/FAIL, two-sample z, sd sanity ratios, realised λ pin, and every timing plus the R/Julia ratios |
| `r_beta_mean_<size>.csv`, `r_beta_sd_<size>.csv` | R posterior mean and sd of `beta`, K rows × M columns, rows in `[const, lag 1 …]` order |
| `r_sigma_mean_<size>.csv`, `r_sigma_sd_<size>.csv` | R posterior mean and sd of `sigma`, M × M |
| `r_hyper_<size>.csv` | realised λ mean/sd/min/max, optimiser mode, MH acceptance, `beta` dimension order, explanatory and variable names |
| `r_timings.csv` | R median/min/max seconds for `bvar()`, `irf_core`, and `irf()` |
| `r_versions.txt` | `sessionInfo()`, `La_library()`, BLAS/LAPACK versions, thread env |
| `julia_versions.txt` | Julia version, `BLAS.get_config()`, thread counts, chip and macOS build |

Only summaries are ever written — means, sds and timings. Raw draws (up to
20 000 × 51 × 10 per size) stay in memory and are discarded, which keeps the
whole `results/` directory a few tens of kilobytes.

`data/` holds the generated inputs (`data_*.csv`, `psi_*.csv`,
`config.csv`). They are deterministic outputs of stage 01 and are kept so the
R stage can be rerun on its own.

## Caveat on the timing comparison

The two "1 000 stored draws" numbers are not the same algorithm, and cannot
be: R's `bvar()` treats λ as a parameter, so producing 1 000 stored draws
means an L-BFGS-B marginal-likelihood optimisation followed by 2 000
Metropolis-Hastings iterations, each re-forming the prior and evaluating the
marginal likelihood. Julia's `sample_posterior` takes the hyperparameters as
given and draws 1 000 i.i.d. samples from the exact conjugate posterior. The
comparison is therefore "cost of 1 000 usable posterior draws as each package
is designed to produce them", not a like-for-like kernel benchmark. The IRF
comparison (`irf_core` vs `identify_short_run`) *is* like for like.
