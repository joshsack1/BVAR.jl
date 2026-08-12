# Shared result type for the Bayesian VAR estimation stage (stage 6): a
# uniform container of posterior draws, whichever of the five prior families
# produced them.

"""
    BVARdraws{T<:Real}

Posterior draws of a reduced-form VAR ``Y_t = c + \\Phi_1 Y_{t-1} + \\cdots +
\\Phi_p Y_{t-p} + \\varepsilon_t``, ``\\varepsilon_t \\sim N(0,\\Sigma)``, produced
by `sample_posterior` from one of the five `AbstractVARPrior` families.

Fields `β`/`Σ` are each a `Vector` of length `ndraws`, holding one draw of
the stacked coefficient matrix (``k\\times n``) and residual covariance
(``n\\times n``) per entry, regardless of family — including
`IndependentNIWPrior`, whose draws come from a Turing `Gibbs`/
`GibbsConditional` sampler rather than the direct Monte Carlo draws the other
four families use. For `MinnesotaPrior`, which places no prior on
``\\Sigma``, every entry of `Σ` is the same fixed ``\\text{diag}(\\hat\\sigma_j^2)``
— a degenerate "draw" repeated `ndraws` times, kept only so the struct's
shape stays uniform across families.

- `β::Vector{Matrix{T}}` — posterior draws of the coefficient matrix.
- `Σ::Vector{Matrix{T}}` — posterior draws of the residual covariance.
- `family::Symbol` — the prior family that produced these draws (one of
  `:minnesota`, `:normal_wishart`, `:independent_niw`, `:asymmetric_conjugate`,
  `:hamilton_baumeister`).
- `lags::Int`, `vars::Int`, `names::Vector{Symbol}` — carried over from the
  `VARestimate`/prior the draws were sampled from.
- `include_constant::Bool` — carried over from the `VARestimate`, so
  downstream consumers (stage 7's `lag_blocks`) can locate lag blocks in `β`
  without re-deriving it from `size(β, 1)`.
"""
struct BVARdraws{T<:Real}
    β::Vector{Matrix{T}}
    Σ::Vector{Matrix{T}}
    family::Symbol
    lags::Int
    vars::Int
    names::Vector{Symbol}
    include_constant::Bool
end
