zero!(x::AbstractArray) = fill!(x, zero(eltype(x)))
one!(x::AbstractArray) = fill!(x, one(eltype(x)))

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

@inline proj_box(x::Number, l::Number, u::Number) = min(u, max(l, x))

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
struct Symmetrized{T <: Number, V <: DenseVector{T}, M <: AbstractMatrix{T}}
    K::M
    Kᵀ::M
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
    randn!(StableRNG(0), x0)
    KᵀK = Symmetrized(K, Kᵀ)
    λ, _ = powm!(KᵀK, x0; kwargs...)
    return sqrt(λ)
end

column_norm(A::AbstractMatrix, j::Integer, p) = norm(view(A, :, j), p)
column_norm(A::SparseMatrixCSC, j::Integer, p) = norm(view(nonzeros(A), nzrange(A, j)), p)

mynnz(A::AbstractSparseMatrix) = nnz(A)
mynnz(A::AbstractMatrix) = prod(size(A))

indtype(::AbstractSparseMatrix{T, Ti}) where {T, Ti} = Ti
