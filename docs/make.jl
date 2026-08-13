using Documenter
using BayesianVectorAutoregressions

# GR must render headless, both in CI and for a local build.
ENV["GKSwstype"] = "100"

# No jldoctest blocks exist in src/ today; this makes any future one get
# `using BayesianVectorAutoregressions` for free.
DocMeta.setdocmeta!(
    BayesianVectorAutoregressions,
    :DocTestSetup,
    :(using BayesianVectorAutoregressions);
    recursive = true,
)

makedocs(;
    sitename = "BayesianVectorAutoregressions.jl",
    modules = [BayesianVectorAutoregressions],
    authors = "Joshua Sack <jsack@joshsack.com>",
    format = Documenter.HTML(;
        canonical = "https://joshsack1.github.io/BayesianVectorAutoregressions.jl",
        edit_link = "main",
        prettyurls = get(ENV, "CI", "false") == "true",
        assets = String[],
    ),
    # Pins pwd() to docs/ for every evaluated block, so each plotting page can
    # pull in the shared theme with a bare `include("plot-theme.jl")`.
    workdir = @__DIR__,
    # Every docstring in src/ must be spliced into some page. The Public/Private
    # @autodocs pairs in api/ partition all 18 docstring-bearing files, so a new
    # src/ file that nobody adds to a Pages list fails the build.
    checkdocs = :all,
    # There are no jldoctest blocks, and every interesting output here is a float
    # matrix from an RNG- and BLAS-dependent sampler, so doctests would be flaky
    # by construction. The executed @example blocks cover "does it still run".
    doctest = false,
    # Strict: broken @ref links and uncovered docstrings fail the build.
    warnonly = false,
    pages = [
        "Home" => "index.md",
        # Nav labels are parsed as Markdown, so they must not start with
        # "<digit>." — that reads as an ordered list and fails the build.
        "Guide" => [
            "Stages 1-2 — Pre-Estimation" => "guide/pre-estimation.md",
            "Stage 3 — Frequentist VAR" => "guide/var.md",
            "Stage 4a — Priors" => "guide/priors.md",
            "Stage 4b — Posterior Sampling" => "guide/bayesian.md",
            "Stage 5 — Structural Identification" => "guide/structural.md",
            "Impulse Responses" => "guide/irf.md",
        ],
        "API Reference" => [
            "Data Testing" => "api/data-testing.md",
            "Lag Selection" => "api/lag-selection.md",
            "VAR Estimation" => "api/var-estimation.md",
            "Priors" => "api/priors.md",
            "Hyperparameters" => "api/hyperparameters.md",
            "Posterior Sampling" => "api/bayesian-estimation.md",
            "Structural Identification" => "api/structural.md",
            "Impulse Responses" => "api/irf.md",
        ],
        "Bibliography" => "references.md",
        "How This Package Was Built" => "provenance.md",
    ],
)

deploydocs(;
    repo = "github.com/joshsack1/BayesianVectorAutoregressions.jl",
    devbranch = "main",
    push_preview = false,
)
