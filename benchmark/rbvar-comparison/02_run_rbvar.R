# Stage 02 of the Julia<->R BVAR comparison: the R side.
#
# For each dataset written by 01_generate_data.jl this script does two things:
#
#   PARITY  one long run (25000 draws, 5000 burn-in => 20000 stored) whose
#           posterior mean and sd of beta (K x M) and sigma (M x M) are
#           written out for stage 03 to compare against the closed-form
#           Normal-Wishart posterior moments.
#   TIMING  1 warm-up + 5 timed runs of (a) bvar() producing 1000 stored
#           draws and (b) impulse responses over a prebuilt 1000-draw object.
#
# Matching R's prior to the Julia one (see README for the full mapping):
#   lambda = lambda1 = 0.2          alpha = 2*lambda3 = 2
#   var    = lambda4^2 = 1e7        b     = 1 (random-walk prior mean)
#   psi    = ar_residual_variances(Y, p), passed EXPLICITLY. R's
#            bv_psi(mode = "auto") uses sqrt(arima(.)$sigma2), a standard
#            deviation where the prior scale wants a variance, so the
#            automatic route is never used here.
#
# Pinning the hyperparameters. bvar() refuses to run with an empty `hyper`
# ("Please provide at least one hyperparameter"), so lambda is nominally
# sampled but pinned two ways at once:
#   1. bv_lambda(sd = 1e-6) makes the gamma hyperprior a spike at 0.2;
#   2. bv_mh(scale_hess = 1e-12) shrinks the Metropolis-Hastings proposal to
#      match. Without (2), the default scale_hess = 0.01 proposes steps of
#      ~0.12 against a spike prior: acceptance collapses to ~0.2% and the
#      chain freezes at whatever value it first accepted, typically ~1e-5 off
#      0.2. With both, acceptance is ~2/3 and the stored lambda draws have
#      mean 0.2 to ~1e-10 and sd ~3e-11.
# Stage 03 re-checks the realised pin from r_hyper_<size>.csv.
#
# Run: Rscript benchmark/rbvar-comparison/02_run_rbvar.R

suppressMessages(library(BVAR))

script_dir <- local({
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 1L) dirname(normalizePath(sub("^--file=", "", file_arg))) else getwd()
})
data_dir <- file.path(script_dir, "data")
results_dir <- file.path(script_dir, "results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

N_TIMING_REPS <- 5L
LAMBDA_PRIOR_SD <- 1e-6
MH_SCALE_HESS <- 1e-12

# --- helpers ----------------------------------------------------------------

# Hand-rolled numeric CSV writer (see the note in Project.toml on why this
# harness carries no I/O dependencies). 16 significant digits in scientific
# notation round-trips a double to well inside the Monte Carlo error.
write_num_csv <- function(x, path, col_names) {
  x <- as.matrix(x)
  stopifnot(ncol(x) == length(col_names))
  lines <- c(
    paste(col_names, collapse = ","),
    apply(x, 1L, function(row) paste(formatC(row, format = "e", digits = 16), collapse = ","))
  )
  writeLines(lines, path)
  invisible(path)
}

make_priors <- function(psi) {
  bv_priors(
    hyper = "lambda",
    mn = bv_mn(
      lambda = bv_lambda(mode = 0.2, sd = LAMBDA_PRIOR_SD, min = 1e-4, max = 5),
      alpha = bv_alpha(mode = 2),
      psi = bv_psi(mode = psi), # min/max auto-derive to mode/100, mode*100
      var = 1e7,
      b = 1
    )
  )
}

# median of `reps` timed elapsed seconds, after one untimed warm-up
time_median <- function(expr_fun, reps) {
  invisible(expr_fun())
  secs <- vapply(seq_len(reps), function(i) unname(system.time(expr_fun())["elapsed"]), numeric(1L))
  list(median = median(secs), min = min(secs), max = max(secs))
}

# The exact analogue of Julia's identify_short_run: Cholesky-identified IRFs
# for every stored draw, with no confidence bands. Public irf() additionally
# computes 16/84% quantile bands, so both are timed.
irf_core <- function(x, periods) {
  meta <- x[["meta"]]
  M <- meta[["M"]]; K <- meta[["K"]]; lags <- meta[["lags"]]
  n_save <- meta[["n_save"]]
  out <- array(NA_real_, c(n_save, M, periods, M))
  for (i in seq_len(n_save)) {
    beta_comp <- BVAR:::get_beta_comp(x[["beta"]][i, , ], K, M, lags)
    out[i, , , ] <- BVAR:::compute_irf(
      beta_comp = beta_comp, sigma = x[["sigma"]][i, , ], M = M, lags = lags,
      horizon = periods, identification = TRUE, sign_restr = NULL, zero = FALSE
    )
  }
  out
}

# --- main loop --------------------------------------------------------------

config <- read.csv(file.path(data_dir, "config.csv"), stringsAsFactors = FALSE)
timing_rows <- list()

for (r in seq_len(nrow(config))) {
  cfg <- config[r, ]
  size <- cfg$size
  cat(sprintf(
    "=== size %s (T=%d, n=%d, p=%d) ===\n", size, cfg$T, cfg$n, cfg$p
  ))

  data <- read.csv(file.path(data_dir, sprintf("data_%s.csv", size)))
  psi <- as.numeric(read.csv(file.path(data_dir, sprintf("psi_%s.csv", size)))[1L, ])
  stopifnot(ncol(data) == cfg$n, length(psi) == cfg$n, nrow(data) == cfg$T)
  priors <- make_priors(psi)
  mh <- bv_mh(scale_hess = MH_SCALE_HESS)

  ## PARITY run -------------------------------------------------------------
  set.seed(20260817L)
  t0 <- Sys.time()
  fit <- bvar(
    data, lags = cfg$p,
    n_draw = cfg$n_draw_parity, n_burn = cfg$n_burn_parity, n_thin = 1L,
    priors = priors, mh = mh, verbose = FALSE
  )
  cat(sprintf("  parity run: %.1fs\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))

  # Verify the documented layout rather than trusting it: beta is
  # n_save x K x M, sigma is n_save x M x M.
  stopifnot(identical(dim(fit$beta), c(as.integer(cfg$n_save_parity), as.integer(cfg$K), as.integer(cfg$n))))
  stopifnot(identical(dim(fit$sigma), c(as.integer(cfg$n_save_parity), as.integer(cfg$n), as.integer(cfg$n))))

  beta_mean <- apply(fit$beta, c(2L, 3L), mean)
  beta_sd <- apply(fit$beta, c(2L, 3L), sd)
  sigma_mean <- apply(fit$sigma, c(2L, 3L), mean)
  sigma_sd <- apply(fit$sigma, c(2L, 3L), sd)

  vnames <- fit$variables
  write_num_csv(beta_mean, file.path(results_dir, sprintf("r_beta_mean_%s.csv", size)), vnames)
  write_num_csv(beta_sd, file.path(results_dir, sprintf("r_beta_sd_%s.csv", size)), vnames)
  write_num_csv(sigma_mean, file.path(results_dir, sprintf("r_sigma_mean_%s.csv", size)), vnames)
  write_num_csv(sigma_sd, file.path(results_dir, sprintf("r_sigma_sd_%s.csv", size)), vnames)

  lambda <- fit$hyper[, "lambda"]
  acc_rate <- fit$meta$accepted / fit$meta$n_draw
  writeLines(
    c(
      "key,value",
      sprintf("lambda_mean,%s", formatC(mean(lambda), format = "e", digits = 16)),
      sprintf("lambda_sd,%s", formatC(sd(lambda), format = "e", digits = 16)),
      sprintf("lambda_min,%s", formatC(min(lambda), format = "e", digits = 16)),
      sprintf("lambda_max,%s", formatC(max(lambda), format = "e", digits = 16)),
      sprintf("lambda_optim_mode,%s", formatC(fit$optim$par[[1L]], format = "e", digits = 16)),
      sprintf("mh_accept_rate,%s", formatC(acc_rate, format = "e", digits = 16)),
      sprintf("mh_accepted,%d", fit$meta$accepted),
      sprintf("n_draw,%d", fit$meta$n_draw),
      sprintf("n_save,%d", fit$meta$n_save),
      sprintf("N,%d", fit$meta$N),
      sprintf("K,%d", fit$meta$K),
      sprintf("M,%d", fit$meta$M),
      sprintf("beta_dim,%s", paste(dim(fit$beta), collapse = "x")),
      sprintf("beta_dim_order,%s", "n_save x K x M"),
      sprintf("explanatories,%s", paste(fit$explanatories, collapse = "|")),
      sprintf("variables,%s", paste(vnames, collapse = "|"))
    ),
    file.path(results_dir, sprintf("r_hyper_%s.csv", size))
  )
  cat(sprintf(
    "  lambda: mean=%.12f sd=%.3e  MH accept=%.3f (%d/%d)\n",
    mean(lambda), sd(lambda), acc_rate, fit$meta$accepted, fit$meta$n_draw
  ))
  stopifnot(sd(lambda) < 1e-4, fit$meta$accepted > 0)

  rm(fit); invisible(gc())

  ## TIMING runs ------------------------------------------------------------
  set.seed(90210L)
  t_bvar <- time_median(function() {
    bvar(
      data, lags = cfg$p,
      n_draw = cfg$n_draw_timing, n_burn = cfg$n_burn_timing, n_thin = 1L,
      priors = priors, mh = mh, verbose = FALSE
    )
  }, N_TIMING_REPS)

  # Prebuilt 1000-draw object, so IRF timing excludes estimation.
  set.seed(90211L)
  fit_small <- bvar(
    data, lags = cfg$p,
    n_draw = cfg$n_draw_timing, n_burn = cfg$n_burn_timing, n_thin = 1L,
    priors = priors, mh = mh, verbose = FALSE
  )
  stopifnot(fit_small$meta$n_save == cfg$n_save_timing)

  periods <- as.integer(cfg$irf_periods)
  t_irf_core <- time_median(function() irf_core(fit_small, periods), N_TIMING_REPS)
  t_irf_full <- time_median(function() {
    irf(fit_small, bv_irf(horizon = periods, fevd = FALSE))
  }, N_TIMING_REPS)

  # Horizon semantics check: R's `horizon` counts periods INCLUDING impact
  # (compute_irf writes the shock into slice 1, then iterates 2:horizon), so
  # horizon = 21 matches Julia's identify_short_run(horizon = 20).
  irf_obj <- irf(fit_small, bv_irf(horizon = periods, fevd = FALSE))
  stopifnot(identical(
    dim(irf_obj$irf),
    c(as.integer(cfg$n_save_timing), as.integer(cfg$n), periods, as.integer(cfg$n))
  ))
  cat(sprintf(
    "  irf dim = %s (periods incl. impact = %d)\n",
    paste(dim(irf_obj$irf), collapse = "x"), periods
  ))

  timing_rows[[size]] <- data.frame(
    size = size, T = cfg$T, n = cfg$n, p = cfg$p,
    n_save = cfg$n_save_timing, irf_periods = periods,
    r_bvar_median_s = t_bvar$median, r_bvar_min_s = t_bvar$min, r_bvar_max_s = t_bvar$max,
    r_irf_core_median_s = t_irf_core$median,
    r_irf_core_min_s = t_irf_core$min, r_irf_core_max_s = t_irf_core$max,
    r_irf_full_median_s = t_irf_full$median,
    r_irf_full_min_s = t_irf_full$min, r_irf_full_max_s = t_irf_full$max,
    reps = N_TIMING_REPS,
    stringsAsFactors = FALSE
  )
  cat(sprintf(
    "  timing medians: bvar(1000 saved)=%.3fs  irf_core=%.3fs  irf()=%.3fs\n",
    t_bvar$median, t_irf_core$median, t_irf_full$median
  ))

  rm(fit_small, irf_obj); invisible(gc())
}

timings <- do.call(rbind, timing_rows)
num_cols <- vapply(timings, is.numeric, logical(1L))
timings[num_cols] <- lapply(timings[num_cols], function(x) formatC(x, format = "e", digits = 10))
writeLines(
  c(
    paste(names(timings), collapse = ","),
    apply(timings, 1L, function(row) paste(row, collapse = ","))
  ),
  file.path(results_dir, "r_timings.csv")
)
cat("wrote results/r_timings.csv\n")

## Environment record --------------------------------------------------------
ver <- c(
  "=== R session ===",
  capture.output(print(sessionInfo())),
  "",
  "=== BLAS / LAPACK ===",
  paste("La_library:", La_library()),
  paste("extSoftVersion BLAS:", extSoftVersion()[["BLAS"]]),
  capture.output(print(La_version())),
  "",
  "=== BVAR ===",
  paste("BVAR version:", as.character(packageVersion("BVAR"))),
  "",
  "=== BLAS thread env ===",
  paste(
    c("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"),
    Sys.getenv(c("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")),
    sep = "="
  )
)
writeLines(ver, file.path(results_dir, "r_versions.txt"))
cat("wrote results/r_versions.txt\n")
