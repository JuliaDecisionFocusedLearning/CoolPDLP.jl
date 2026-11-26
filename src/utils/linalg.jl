"""
    GPUSparseMatrix

Union of the sparse formats that CoolPDLP implements with its own kernels.
"""
const GPUSparseMatrix = Union{GPUSparseMatrixCOO, GPUSparseMatrixCSR, GPUSparseMatrixELL}

# The formats are not wrappers around another array: say so, so that packages walking the
# wrapper chain (e.g. Reactant's `ancestor`) do not have to guess, and warn.
Adapt.parent_type(::Type{M}) where {M <: GPUSparseMatrix} = M

# The formats implement `sparse_mul!`, and `mul!` forwards to it. Vectors may be strided, like
# the column views that Krylov.jl multiplies: those on different backends throw an
# `ArgumentError` rather than fall back to the generic `mul!`, which reads `A` entry by entry.
# The element type of `A` may differ from that of the vectors (mixed precision): the kernels
# read `A` in its own precision and accumulate in that of `c`.
for M in (:GPUSparseMatrixCOO, :GPUSparseMatrixCSR, :GPUSparseMatrixELL)
    @eval begin
        function LinearAlgebra.mul!(
                c::StridedVector{T}, A::$M, b::StridedVector{T}, α::Number, β::Number
            ) where {T <: Number}
            return sparse_mul!(c, A, b, α, β)
        end
        function LinearAlgebra.mul!(
                c::DenseMatrix{T}, A::$M, b::DenseMatrix{T}, α::Number, β::Number
            ) where {T <: Number}
            return sparse_mul!(c, A, b, α, β)
        end
    end
end

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

# `K` with its explicit transpose, since Krylov.jl multiplies by `K'`
struct KrylovOperator{M, Mt}
    K::M
    Kᵀ::Mt
end
Base.size(op::KrylovOperator) = size(op.K)
Base.eltype(op::KrylovOperator) = eltype(op.K)
Base.adjoint(op::KrylovOperator) = KrylovOperator(op.Kᵀ, op.K)
LinearAlgebra.mul!(y::AbstractVector, op::KrylovOperator, x::AbstractVector) = mul!(y, op.K, x)

"""
    spectral_norm(K, Kᵀ; reltol, maxiter)

Return an iterative estimate `σ` of the spectral norm `‖K‖₂`, from a deterministic start and
with at most `maxiter` products by each of `K` and `Kᵀ`, accepted once `K` has a singular
value within `reltol * σ` of it. `maxiter` should not depend on the size of `K` (#95).

On non-convergence, warn and return the conservative estimate `σ + residual`.
"""
function spectral_norm(
        K::AbstractMatrix{<:Number},
        Kᵀ::AbstractMatrix{<:Number};
        reltol::Number,
        maxiter::Integer,
    )
    T = eltype(K)
    m, n = size(K)
    (m == 0 || n == 0) && return zero(T)
    # LAPACK in `Float32` and `Float64`, GenericLinearAlgebra for other precisions
    if !hasmethod(LinearAlgebra.svd!, Tuple{Bidiagonal{T, Vector{T}}})
        msg = "`spectral_norm` needs the SVD of a `Bidiagonal{$T}`: load GenericLinearAlgebra to provide it"
        throw(ArgumentError(msg))
    end
    b = copyto!(allocate(get_backend(K), T, m), randn(StableRNG(0), T, m))
    b′ = similar(b, min(32, maxiter))
    σ, residual = zero(T), typemax(T)
    for start in 1:32:maxiter
        # restart Golub–Kahan every `k ≤ 32` steps, from the top left Ritz vector
        k = min(32, maxiter - start + 1)
        _, U, _, L = golub_kahan(KrylovOperator(K, Kᵀ), b, k; allow_breakdown = true)
        nz = nonzeros(L)  # stored column by column: `αᵢ = L[i, i]` and `βᵢ₊₁ = L[i + 1, i]`
        F = LinearAlgebra.svd!(Bidiagonal(nz[1:2:(2k - 1)], nz[2:2:(2k - 2)], :L))
        i = argmax(F.S)
        σ, residual = T(F.S[i]), nz[2k] * T(abs(F.V[k, i]))
        residual <= reltol * σ && return σ
        mul!(b, view(U, :, 1:k), copyto!(view(b′, 1:k), view(F.U, :, i)))
    end
    estimate = σ + residual
    @warn "Spectral norm estimate did not converge within $maxiter iterations (residual $residual > $(reltol * σ)), using the conservative estimate $estimate"
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
