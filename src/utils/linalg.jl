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

@inline safeprod_left(left, right) = ifelse(isinf(left), right, left * right)

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
    ls = isfinite(l) ? abs(l) : zero(l)
    us = isfinite(u) ? abs(u) : zero(u)
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
Base.size(sym::Symmetrized, ::Int) = size(sym.K, 2)

function LinearAlgebra.mul!(y, sym::Symmetrized, x)
    (; K, Kᵀ, scratch) = sym
    mul!(scratch, K, x)
    mul!(y, Kᵀ, scratch)
    return y
end

"""
    spectral_norm(K, Kᵀ)

Compute the spectral norm of `K` with the power method from IterativeSolvers.jl.
"""
function spectral_norm(
        K::AbstractMatrix{<:Number},
        Kᵀ::AbstractMatrix{<:Number};
        kwargs...
    )
    x0 = allocate(get_backend(K), eltype(K), size(K, 2))
    x0_cpu = adapt(CPU(), x0)  # StableRNGs doesn't work on GPU
    randn!(StableRNG(0), x0_cpu)
    copyto!(x0, x0_cpu)
    KᵀK = Symmetrized(K, Kᵀ)
    λ, _ = powm!(KᵀK, x0; kwargs...)
    return sqrt(λ)
end

column_norm(A::AbstractMatrix, j::Integer, p) = norm(view(A, :, j), p)
column_norm(A::SparseMatrixCSC, j::Integer, p) = norm(view(nonzeros(A), nzrange(A, j)), p)

mynnz(A::AbstractSparseMatrix) = nnz(A)
mynnz(A::AbstractMatrix) = prod(size(A))

indtype(::AbstractSparseMatrix{T, Ti}) where {T, Ti} = Ti
