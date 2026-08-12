# Impulse-response machinery (stage 7): converts stored k×n coefficient
# blocks (the row convention shared by VARestimate.β_hat, BVARdraws.β, and
# structural B — constant, if present, then lag 1 of every variable, then
# lag 2, ...) into physics-convention lag matrices, builds the companion
# form, and computes nonorthogonalized and structural impulse responses.
# Shared by both the Baumeister-Hamilton structural path
# (hamilton-structural.jl) and the traditional Cholesky/sign-restriction path
# (identification.jl).

"""
    lag_blocks(S::AbstractMatrix, lags::Int, include_constant::Bool)

Splits a stored ``k\\times n`` coefficient matrix (the row convention shared
by `VARestimate.β_hat`, `BVARdraws.β`, and structural `B`: constant, if
present, then lag 1 of every variable, then lag 2, …) into the
``n\\times n`` lag matrices ``\\Phi_1,\\ldots,\\Phi_p`` of the column-vector
convention ``Y_t = c + \\Phi_1 Y_{t-1} + \\cdots + \\Phi_p Y_{t-p} +
\\varepsilon_t``. Returns a `Vector` of length `lags`.
"""
function lag_blocks(S::AbstractMatrix{T}, lags::Int, include_constant::Bool) where {T<:Real}
    n = size(S, 2)
    offset = include_constant ? 1 : 0
    return [Matrix{T}(S[(offset + (ℓ - 1) * n + 1):(offset + ℓ * n), :]') for ℓ in 1:lags]
end

"""
    companion_matrix(Φ::Vector{<:AbstractMatrix})

Builds the ``np\\times np`` companion matrix

``F = \\begin{bmatrix}\\Phi_1 & \\Phi_2 & \\cdots & \\Phi_p\\\\ I_n & 0 & \\cdots & 0\\\\
\\vdots & & \\ddots & \\vdots\\\\ 0 & \\cdots & I_n & 0\\end{bmatrix}``

of the VAR(p) with lag matrices `Φ`, whose powers generate the
nonorthogonalized impulse responses (`nonorthogonalized_irf`).
"""
function companion_matrix(Φ::Vector{<:AbstractMatrix{T}}) where {T<:Real}
    p = length(Φ)
    n = size(Φ[1], 1)
    F = zeros(T, n * p, n * p)
    for i in 1:p
        F[1:n, ((i - 1) * n + 1):(i * n)] = Φ[i]
    end
    p > 1 && (F[(n + 1):end, 1:(n * (p - 1))] = I(n * (p - 1)))
    return F
end

"""
    nonorthogonalized_irf(Φ::Vector{<:AbstractMatrix}, horizon::Int)

Computes the nonorthogonalized impulse-response matrices ``\\Psi_0=I_n``,
``\\Psi_s=\\sum_{\\ell=1}^{\\min(s,p)}\\Phi_\\ell\\Psi_{s-\\ell}`` for
``s=0,\\ldots,``horizon, via the leading ``n\\times n`` block of powers of
the companion matrix (`companion_matrix`; Hamilton 1994, p. 260). Returns a
`Vector` of length `horizon + 1`.
"""
function nonorthogonalized_irf(Φ::Vector{<:AbstractMatrix{T}}, horizon::Int) where {T<:Real}
    p = length(Φ)
    n = size(Φ[1], 1)
    F = companion_matrix(Φ)
    Ψ = Vector{Matrix{T}}(undef, horizon + 1)
    Ψ[1] = Matrix{T}(I, n, n)
    Fˢ = Matrix{T}(I, n * p, n * p)
    for s in 1:horizon
        Fˢ = Fˢ * F
        Ψ[s + 1] = Fˢ[1:n, 1:n]
    end
    return Ψ
end

"""
    impulse_responses(Φ::Vector{<:AbstractMatrix}, impact::AbstractMatrix, horizon::Int)

Computes the structural impulse responses ``H_s = \\Psi_s \\cdot
\\text{impact}`` for ``s=0,\\ldots,``horizon (Hamilton 1994, pp. 260, 331),
where `impact` converts one-standard-deviation reduced-form shocks into
structural ones — ``A^{-1}`` for the Baumeister & Hamilton structural path
(`impulse_response`), or a Cholesky factor (`identify_short_run`) or its
random rotation (`identify_sign_restrictions`) for the traditional path.
Returns a `Vector` of length `horizon + 1`.
"""
function impulse_responses(
    Φ::Vector{<:AbstractMatrix{T}},
    impact::AbstractMatrix,
    horizon::Int,
) where {T<:Real}
    Ψ = nonorthogonalized_irf(Φ, horizon)
    return [Ψₛ * impact for Ψₛ in Ψ]
end

"""
    long_run_multiplier(A::AbstractMatrix, B::AbstractMatrix, lags::Int, include_constant::Bool)

Computes the cumulative long-run effect of a unit structural shock,

``\\Xi = \\left(A - \\sum_{\\ell=1}^{p} B_\\ell\\right)^{-1},``

where ``B_\\ell`` are `B`'s lag blocks (`lag_blocks`). At the reduced form
(``A=I``) this is the familiar long-run multiplier ``(I-\\Phi_1-\\cdots
-\\Phi_p)^{-1}``. Expressing a restriction on ``\\Xi`` (e.g. a sign
restriction, à la Blanchard & Quah 1989) is how long-run restrictions enter
the general structural-prior framework (`StructuralPrior`), via a closure
over `(A,B)` in its `restrictions` field — see `long_run_sign_restriction`.
"""
function long_run_multiplier(
    A::AbstractMatrix,
    B::AbstractMatrix,
    lags::Int,
    include_constant::Bool,
)
    Bℓ = lag_blocks(B, lags, include_constant)
    return inv(A - sum(Bℓ))
end

"""
    impulse_response(draws::StructuralDraws; horizon::Int = 20)

Computes the structural impulse response functions

``H_s = \\Psi_s A^{-1}, \\qquad \\Psi_0 = I, \\qquad
\\Psi_s = \\sum_{\\ell=1}^{\\min(s,p)} \\Phi_\\ell \\Psi_{s-\\ell},``

for ``s = 0,\\ldots,``horizon, one set per draw, where ``\\Phi_\\ell =
A^{-1}B_\\ell`` are the reduced-form lag matrices implied by the structural
draw. Shared with the traditional short-run/sign-restriction identification
path (`identify_short_run`, `identify_sign_restrictions`) via the
lower-level `impulse_responses`, which takes any impact matrix (``A^{-1}``
here; a Cholesky factor or its rotation there). Returns an `IRFdraws`.
"""
function impulse_response(draws::StructuralDraws{T}; horizon::Int = 20) where {T<:Real}
    ndraws = length(draws.A)
    H = Vector{Vector{Matrix{T}}}(undef, ndraws)
    for d in 1:ndraws
        A_inv = inv(draws.A[d])
        Bℓ = lag_blocks(draws.B[d], draws.lags, draws.include_constant)
        Φ = [A_inv * Bᵢ for Bᵢ in Bℓ]
        H[d] = impulse_responses(Φ, A_inv, horizon)
    end
    return IRFdraws(H, horizon, draws.lags, draws.vars, draws.names, :hamilton_structural)
end
