"""
    ols!(x, A, b; x0, tol, maxiter, mu, zero_tol, deadline)

`min ‖A x − b‖₂` by PCG on `(AᵀA + μ I) x = Aᵀb` ([`Symmetrized`](@ref)).

Krylov vectors are `similar(x)`. Column scale `1/(‖col‖₂+ε)` is filled for a
host CSC matrix; any other matrix keeps the identity scale. Stop when
`(√(rᵀPr)` or `‖r‖) < tol·(1+‖Aᵀb‖)` and `‖r‖_∞ < max(1e-10, eps(T))`.
`μ = 1e-12` is the ridge of the reference OLS. The absolute `1e-10` is the
Liu–Lu test. Both are raised to `eps(T)` when they vanish next to an O(1)
entry. The default iteration cap `max(10 * max(m, n), 1000)` is a fuse:
the residual test and `deadline` stop the loop. `deadline` is an absolute `time()`.
"""
function ols!(
        x::AbstractVector{T},
        A::AbstractMatrix,
        b::AbstractVector;
        x0 = nothing,
        tol::Real = 1.0e-8,
        maxiter::Int = 0,
        mu::Real = 1.0e-12,
        zero_tol::Real = 1.0e-8,
        deadline::Float64 = Inf,
    ) where {T}
    m, n = size(A)
    length(x) == n || throw(DimensionMismatch("x length $(length(x)) ≠ ncols $n"))
    length(b) == m || throw(DimensionMismatch("b length $(length(b)) ≠ nrows $m"))
    (m == 0 || n == 0) && return x
    copyto!(x, x0 === nothing ? x : x0)
    At = A isa SparseMatrixCSC ? transpose(A) : sametype_transpose(A)
    Atb = similar(x)
    mul!(Atb, At, b)
    P = similar(x)
    _ols_scale!(P, A, zero_tol)
    # Fuse, not a paper constant. The Liu–Lu test and `deadline` stop first.
    maxit = maxiter > 0 ? maxiter : max(10 * max(m, n), 1000)
    return _ols_pcg_loop!(
        # `1e-12` vanishes next to an O(1) entry in `Float32` (`1 + μ == 1`).
        x, _Ridge(Symmetrized(A, At), max(T(mu), eps(T))), Atb, P;
        tol = T(tol), maxiter = maxit, deadline,
    )
end

"""`Pⱼ = 1/(‖A_{·j}‖₂+ε)` via `column_norm` on a host CSC matrix; identity otherwise."""
function _ols_scale!(P::AbstractVector{T}, A::AbstractMatrix, ε::Real) where {T}
    fill!(P, one(T))
    (A isa SparseMatrixCSC && P isa Array) || return P
    εT = T(ε)
    for j in eachindex(P)
        P[j] = inv(column_norm(A, j, 2) + εT)
    end
    return P
end

# `op + μ I`.
struct _Ridge{T, S}
    op::S
    mu::T
end
function LinearAlgebra.mul!(y, R::_Ridge, x)
    mul!(y, R.op, x)
    @. y = y + R.mu * x
    return y
end

function _ols_pcg_loop!(x, op, Atb, pscale; tol::Real, maxiter::Int, deadline::Float64 = Inf)
    T = eltype(x)
    r, w, z, p = similar(x), similar(x), similar(x), similar(x)
    mul!(w, op, x)
    @. r = Atb - w
    @. z = pscale * r
    copyto!(p, z)
    ρ = dot(r, z)
    scale = one(T) + norm(Atb)
    # `1e-10` is finer than `eps(Float32)`, so the absolute stop would never fire.
    tolT, abs_inf = T(tol), max(T(1.0e-10), eps(T))
    for _ in 1:maxiter
        time() >= deadline && break
        nrm = norm(r)
        if (sqrt(max(ρ, zero(ρ))) < tolT * scale || nrm < tolT * scale) && norm(r, Inf) < abs_inf
            return x
        end
        mul!(w, op, p)
        denom = dot(w, p)
        if !(denom > 0) || !isfinite(denom) || !isfinite(ρ) || ρ == 0
            break
        end
        α = ρ / denom
        axpy!(α, p, x)
        axpy!(-α, w, r)
        @. z = pscale * r
        ρ_new = dot(r, z)
        if !isfinite(ρ_new)
            break
        end
        axpby!(one(T), z, ρ_new / ρ, p)  # p ← z + β p
        ρ = ρ_new
    end
    return x
end
