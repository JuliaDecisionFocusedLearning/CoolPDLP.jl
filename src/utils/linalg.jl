"""
    GPUSparseMatrix

Union of the sparse formats that CoolPDLP implements with its own kernels.
"""
const GPUSparseMatrix = Union{GPUSparseMatrixCOO, GPUSparseMatrixCSR, GPUSparseMatrixELL}

# The formats implement `sparse_mul!`, and `mul!` forwards to it with exactly the signatures of
# `sparse_mul!`: a broader one would be ambiguous with LinearAlgebra's generic `mul!`, and would
# also catch the products with mismatched storage that this generic `mul!` used to serve.
for (M, AV) in (
        (:GPUSparseMatrixCOO, :(GPUSparseMatrixCOO{T, Ti, V})),
        (:GPUSparseMatrixCSR, :(GPUSparseMatrixCSR{T, Ti, V})),
        (:GPUSparseMatrixELL, :(GPUSparseMatrixELL{T, Ti})),
    )
    @eval begin
        function LinearAlgebra.mul!(
                c::V, A::$AV, b::V, α::Number, β::Number
            ) where {T <: Number, Ti, V <: DenseVector{T}}
            return sparse_mul!(c, A, b, α, β)
        end
        function LinearAlgebra.mul!(
                c::DenseMatrix{T}, A::$M{T}, b::DenseMatrix{T}, α::Number, β::Number
            ) where {T <: Number}
            return sparse_mul!(c, A, b, α, β)
        end
    end
end

"""
    map(f, A::GPUSparseMatrix)

Apply `f` to the stored entries of `A`, keeping its sparsity structure. Unlike `map` on a
`SparseMatrixCSC`, the implicit zeros are left alone, so this requires `f(0) == 0`.
"""
function Base.map(f, A::GPUSparseMatrix)
    iszero(f(zero(eltype(A)))) ||
        throw(ArgumentError("map on a CoolPDLP sparse matrix needs a function that maps zero to zero"))
    return _map_nzval(f, A)
end
_map_nzval(f, A::GPUSparseMatrixCSR) = GPUSparseMatrixCSR(A.m, A.n, A.rowptr, A.colval, map(f, A.nzval))
_map_nzval(f, A::GPUSparseMatrixCOO) = GPUSparseMatrixCOO(A.m, A.n, A.rowval, A.colval, map(f, A.nzval))
_map_nzval(f, A::GPUSparseMatrixELL) = GPUSparseMatrixELL(A.m, A.n, A.colval, map(f, A.nzval))

"""
    sametype_transpose(A::AbstractMatrix)

Return a matrix of the same type of `A` containing `transpose(A)` (as opposed to a `Transpose{...}` wrapper).

The default implementation is just `convert(typeof(A), transpose(A))` but it may need to be overloaded for certain matrix types.
"""
sametype_transpose(A::AbstractMatrix) = convert(typeof(A), transpose(A))

zero!(x::AbstractArray) = fill!(x, zero(eltype(x)))
one!(x::AbstractArray) = fill!(x, one(eltype(x)))

# helpers for `mul!` implementations
struct One <: Number end
Base.:*(::One, x::Number) = x
struct Zero <: Number end
Base.:*(::Zero, ::Number) = Zero()
Base.:+(x::Number, ::Zero) = x

"""
    colnorm(x)

Return the Euclidean norm of `x`, or one norm per column if `x` is batched.
"""
colnorm(v::AbstractVector) = norm(v)
colnorm(m::AbstractMatrix) = norm.(eachcol(m))

"""
    colnorm!!(dest, x)

Compute the Euclidean norm of `x`, or one norm per column if `x` is batched, into `dest`.

Unlike [`colnorm`](@ref), the batched (`AbstractMatrix`) method computes `sqrt(sum(abs2, ...))`
directly instead of `LinearAlgebra.norm`'s scaled, overflow-safe algorithm: it needs to stay a
single allocation-free, GPU-broadcastable reduction, since it runs every iteration on the hot
path (see [`primal_weight_update!!`](@ref) and [`kkt_errors!`](@ref)). This means `colnorm!!` can
over/underflow on badly scaled columns where `colnorm` would not; that tradeoff is intentional
here, since the values it norms are primal-dual residuals that are not expected to approach the
extremes of the floating-point range in practice.
"""
colnorm!!(::Number, v::AbstractVector) = norm(v)
function colnorm!!(dest::AbstractVector, m::AbstractMatrix)
    sum!(abs2, transpose(dest), m)
    dest .= sqrt.(dest)
    return dest
end

"""
    coldot(a, b)

Return the scalar product of `a` and `b`, or one scalar product per column if either is batched.
"""
coldot(a::AbstractVector, b::AbstractVector) = dot(a, b)
coldot(a::AbstractVecOrMat, b::AbstractVecOrMat) = vec(sum(a .* b; dims = 1))

"""
    colsum!!(dest, x)

Compute the sum of `x`, or one sum per column if `x` is batched, into `dest`.
"""
colsum!!(::Number, v::AbstractVector) = sum(v)
function colsum!!(dest::AbstractVector, m::AbstractMatrix)
    sum!(transpose(dest), m)
    return dest
end

@inline positive_part(a::Number) = max(a, zero(a))
@inline negative_part(a::Number) = -min(a, zero(a))

@inline function safe(x::T) where {T <: AbstractFloat}
    if x == typemax(T)
        return prevfloat(x)
    elseif x == typemin(T)
        return nextfloat(x)
    else
        return x
    end
end

"""
    safeprod_left(left, right)

Compute `left * right`, except that `right` is forced to zero whenever `left` is infinite.

This is used to evaluate terms like `l * y⁺` in the dual objective, where `l` is a (possibly
infinite) constraint/variable bound and `y⁺` its associated Lagrange multiplier. The PDHG
update and [`proj_multiplier`](@ref) guarantee that the multiplier paired with an infinite
bound is mathematically zero, but not necessarily *exactly* zero in floating point (e.g. a
free row's multiplier is computed as a `σ`/`inv(σ)` round-trip that can leave a tiny nonzero
residual). Checking `iszero(right)` instead of always zeroing out would let such residuals
turn into `±Inf` (and, once several rows are summed, `Inf - Inf = NaN`) even though nothing
is actually wrong with the solution.
"""
@inline safeprod_left(left, right) = ifelse(isinf(left), zero(right), left * right)

"""
    proj_multiplier(λ, l, u)

Project `λ` onto the feasible space of the (double) Lagrange multiplier `λ⁺ - λ⁻` associated with the constraint `l ≤ x ≤ u`, where `l` and/or `u` might be infinite.
"""
@inline function proj_multiplier(λ::T, l::T, u::T) where {T <: Number}
    lmin = l == typemin(T)
    umax = u == typemax(T)
    return ifelse(
        lmin,
        ifelse(
            umax,
            zero(T),
            -negative_part(λ)
        ),
        ifelse(
            umax,
            positive_part(λ),
            λ
        )
    )
end

"""
    combine(l, u)

Return the largest finite absolute value between the two bounds, or zero if neither is finite.
"""
function combine(l::Number, u::Number)
    ls = ifelse(isfinite(l), abs(l), zero(l))
    us = ifelse(isfinite(u), abs(u), zero(u))
    return max(zero(l), ls, us)
end

"""
    Symmetrized

Represent a symmetric matrix `Kᵀ * K` lazily.
"""
struct Symmetrized{T <: Number, V <: DenseVector{T}, M <: AbstractMatrix{T}, Mt <: AbstractMatrix{T}}
    K::M
    Kᵀ::Mt
    scratch::V
end

function Symmetrized(K::AbstractMatrix, Kᵀ::AbstractMatrix)
    scratch = allocate(get_backend(K), eltype(K), size(K, 1))
    return Symmetrized(K, Kᵀ, scratch)
end

Base.eltype(sym::Symmetrized) = eltype(sym.K)
Base.size(sym::Symmetrized) = (n = size(sym.K, 2); (n, n))
Base.size(sym::Symmetrized, d::Int) = d in (1, 2) ? size(sym.K, 2) : 1

function LinearAlgebra.mul!(y, sym::Symmetrized, x)
    check_mul_dims(y, sym, x)
    (; K, Kᵀ, scratch) = sym
    mul!(scratch, K, x)
    mul!(y, Kᵀ, scratch)
    return y
end

"""
    spectral_norm_upper_bound(K, Kᵀ)

Return the cheap and rigorous upper bound `sqrt(‖K‖₁ ‖K‖∞) ≥ ‖K‖₂`, computed with two
matrix-vector products by `map(abs, K)` and `map(abs, Kᵀ)` so that it works on every backend.
"""
function spectral_norm_upper_bound(K::AbstractMatrix{T}, Kᵀ::AbstractMatrix{T}) where {T <: Number}
    m, n = size(K)
    (m == 0 || n == 0) && return zero(real(T))
    backend = get_backend(K)
    ones_n = allocate(backend, T, n)
    ones_m = allocate(backend, T, m)
    fill!(ones_n, one(T))
    fill!(ones_m, one(T))
    row_sums = allocate(backend, T, m)
    col_sums = allocate(backend, T, n)
    mul!(row_sums, map(abs, K), ones_n)
    mul!(col_sums, map(abs, Kᵀ), ones_m)
    norm_inf = real(maximum(row_sums))
    norm_1 = real(maximum(col_sums))
    return sqrt(norm_1 * norm_inf)
end

"""
    KrylovOperator

The linear operator `K` with its explicit transpose `Kᵀ`, for Krylov.jl, which multiplies by
`K'`: `adjoint` swaps them, so that only `mul!(y, K, x)` and `mul!(y, Kᵀ, x)` are needed.
"""
struct KrylovOperator{M, Mt}
    K::M
    Kᵀ::Mt
end

Base.size(op::KrylovOperator) = size(op.K)
Base.size(op::KrylovOperator, d::Integer) = size(op.K, d)
Base.eltype(op::KrylovOperator) = eltype(op.K)
Base.adjoint(op::KrylovOperator) = KrylovOperator(op.Kᵀ, op.K)
LinearAlgebra.mul!(y::AbstractVector, op::KrylovOperator, x::AbstractVector) = mul!(y, op.K, x)

"""
    SPECTRAL_NORM_KRYLOV_DIM

Number of Golub–Kahan steps between two restarts of [`spectral_norm`](@ref), which bounds its
memory to `SPECTRAL_NORM_KRYLOV_DIM + 1` vectors of each size of `K`.
"""
const SPECTRAL_NORM_KRYLOV_DIM = 32

"""
    top_singular_value!(b, K, Kᵀ, k)

Run `k` steps of Golub–Kahan bidiagonalization of `K` from `b`, and return the largest
singular value `σ` of the `k × k` bidiagonal matrix together with the bound `residual` such
that `K` has a singular value within `residual` of `σ`. Overwrite `b` with the corresponding
left Ritz vector, to restart from.
"""
function top_singular_value!(b, K, Kᵀ, k::Integer)
    R = real(eltype(K))
    _, U, _, L = golub_kahan(KrylovOperator(K, Kᵀ), b, k; allow_breakdown = true)
    d = R[L[i, i] for i in 1:k]
    e = R[L[i + 1, i] for i in 1:(k - 1)]
    β = R(L[k + 1, k])
    all(isfinite, d) && all(isfinite, e) && isfinite(β) ||
        throw(ArgumentError("spectral_norm: the matrix has NaN or infinite entries (or overflows in $(eltype(K)))"))
    F = svd(Bidiagonal(d, e, :L))
    i = argmax(F.S)
    # `K Vₖ = Uₖ Bₖ + β uₖ₊₁ eₖᵀ` and `Kᵀ Uₖ = Vₖ Bₖᵀ`: the top singular triplet `(σ, p, q)`
    # of `Bₖ` leaves the residual `β |qₖ|`, and `Uₖ p` is its left Ritz vector
    p = allocate(get_backend(b), eltype(b), k)
    copyto!(p, convert(Vector{eltype(b)}, F.U[:, i]))
    mul!(b, view(U, :, 1:k), p)
    return F.S[i], β * abs(F.V[k, i])
end

"""
    spectral_norm(K, Kᵀ; reltol, maxiter)

Return an iterative estimate of the spectral norm `‖K‖₂`, starting from a deterministic
random vector and using at most `maxiter` products by both `K` and `Kᵀ`.

The estimate `σ` is accepted once `K` provably has a singular value within `reltol * σ` of
it, and then lies below `‖K‖₂` by at most about that much when `σ` approximates the top of the
spectrum. `maxiter` should be chosen independently of the size of `K` (see #95). The iteration
restarts every [`SPECTRAL_NORM_KRYLOV_DIM`](@ref) steps, which bounds its memory independently of `maxiter`.

This never throws on non-convergence: it then warns (at every call, since each one is a
different matrix that gets a conservative step size) and returns the minimum between the
inflated estimate `1.01 * (σ + residual)` and the rigorous bound `sqrt(‖K‖₁ ‖K‖∞)`, so as to
err on the large side (a smaller, safe step size) rather than the small side (a divergent
step size).

This runs with data-dependent control flow and is not meant to be traced by Reactant: it is
only called from `initialize`, outside of the traced `solve!` loop. Any element type is
supported, and an `ArgumentError` is thrown if `K` has a NaN or infinite entry.
"""
function spectral_norm(
        K::AbstractMatrix{<:Number},
        Kᵀ::AbstractMatrix{<:Number};
        reltol::Number,
        maxiter::Integer,
    )
    T = eltype(K)
    R = real(T)
    m, n = size(K)
    (m == 0 || n == 0) && return zero(R)
    b = allocate(get_backend(K), T, m)
    b_cpu = adapt(CPU(), b)  # StableRNGs doesn't work on GPU
    randn!(StableRNG(0), b_cpu)
    copyto!(b, b_cpu)
    σ, residual = zero(R), typemax(R)
    remaining = maxiter
    while remaining > 0
        k = min(SPECTRAL_NORM_KRYLOV_DIM, remaining)
        σ, residual = top_singular_value!(b, K, Kᵀ, k)
        residual <= R(reltol) * σ && return σ
        remaining -= k
    end
    bound = R(spectral_norm_upper_bound(K, Kᵀ))
    estimate = min(R(1.01) * (σ + residual), bound)
    @warn "Spectral norm estimation did not converge within $maxiter iterations (relative residual $(residual / σ) > reltol = $reltol), using the conservative estimate $estimate (iterative value $σ, rigorous bound $bound)"
    return estimate
end

column_norm(A::AbstractMatrix, j::Integer, p) = norm(view(A, :, j), p)
column_norm(A::SparseMatrixCSC, j::Integer, p) = norm(view(nonzeros(A), nzrange(A, j)), p)

@inline _power_term(x, p) = iszero(x) ? zero(x) : abs(x)^p

"""
    column_power_sum(A, j, p)

Return `sum_i |A[i, j]|^p`. Zero entries are treated as contributing `0` (not `0^0 = 1`),
so the sparse and dense cases are aligned.
"""
column_power_sum(A::AbstractMatrix, j::Integer, p) =
    sum(x -> _power_term(x, p), view(A, :, j); init = zero(eltype(A)))
column_power_sum(A::SparseMatrixCSC, j::Integer, p) =
    sum(x -> _power_term(x, p), view(nonzeros(A), nzrange(A, j)); init = zero(eltype(A)))

mynnz(A::AbstractSparseMatrix) = nnz(A)
mynnz(A::AbstractMatrix) = prod(size(A))

indtype(::AbstractSparseMatrix{T, Ti}) where {T, Ti} = Ti

"""
    sparse_mul!(c, A, b, α, β)
    sparse_mul!(c::AbstractMatrix, A, b::AbstractMatrix, α, β; workgroupsize = nothing)

Kernel-launching body of `mul!` for the formats in [`GPUSparseMatrix`](@ref). Kept separate so
that the Reactant extension's overlay of `mul!` can reach it without calling `mul!` again.

For a batch of vectors, `workgroupsize` is passed on to the kernel launch, where `nothing` lets
KernelAbstractions choose it.
"""
function sparse_mul! end
