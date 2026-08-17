#!/usr/bin/env bash
# End-to-end driver for the Julia<->R BVAR comparison.
#
#   01_generate_data.jl  (Julia)  simulate the shared datasets + psi + config
#   02_run_rbvar.R       (R)      parity + timing runs of the R BVAR package
#   03_run_julia.jl      (Julia)  Julia side, parity verdict, timing table
#
# Timings are only comparable if neither language is quietly using more cores
# than the other, so all three BLAS thread knobs are pinned to 1 for every
# stage (Julia additionally calls BLAS.set_num_threads(1) internally, and the
# R side records what it saw in results/r_versions.txt).
#
# Usage: ./run.sh      (from anywhere; paths are resolved relative to this file)

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSCRIPT="${RSCRIPT:-Rscript}"
JULIA="${JULIA:-julia}"

command -v "$RSCRIPT" >/dev/null 2>&1 || {
  echo "error: Rscript not found on PATH (override with RSCRIPT=/path/to/Rscript)" >&2
  exit 1
}
command -v "$JULIA" >/dev/null 2>&1 || {
  echo "error: julia not found on PATH (override with JULIA=/path/to/julia)" >&2
  exit 1
}
"$RSCRIPT" -e 'if (!requireNamespace("BVAR", quietly = TRUE)) {
  stop("The R package BVAR is not installed. Run:\n",
       "  Rscript -e '\''install.packages(\"BVAR\", repos = \"https://cloud.r-project.org\")'\''")
}' >/dev/null

mkdir -p "$HERE/data" "$HERE/results"

echo "--- 01_generate_data.jl -------------------------------------------------"
"$JULIA" --project="$HERE" "$HERE/01_generate_data.jl"

echo "--- 02_run_rbvar.R -----------------------------------------------------"
"$RSCRIPT" "$HERE/02_run_rbvar.R"

echo "--- 03_run_julia.jl ----------------------------------------------------"
"$JULIA" --project="$HERE" "$HERE/03_run_julia.jl"

echo
echo "done. results in $HERE/results"
