zero!(x::AbstractArray) = fill!(x, zero(eltype(x)))

@inline positive_part(a::Number) = max(a, zero(a))
@inline negative_part(a::Number) = -min(a, zero(a))

@inline proj_box(x::Number, l::Number, u::Number) = min(u, max(l, x))

sqnorm(v::AbstractVector{<:Number}) = dot(v, v)

struct Symmetrized{T <: Number, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
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

column_view(A::SparseMatrixCSC, j::Integer) = view(nonzeros(A), nzrange(A, j))


"""
    sort_columns(A::SparseMatrixCSC)

Return a version `A_sorted` of `A` where the columns have been sorted in order of increasing number of nonzeros, along with the associated column permutation `p` such that `A_sorted == A[:, p]`
"""
function sort_columns(A::SparseMatrixCSC)
    (; colptr, rowval, nzval) = A
    col_lengths = diff(colptr)
    col_perm = sortperm(col_lengths)
    new_colptr = similar(colptr)
    new_rowval = similar(rowval)
    new_nzval = similar(nzval)
    k = 1
    for (new_j, j) in enumerate(col_perm)
        new_colptr[new_j] = k
        for l in colptr[j]:(colptr[j + 1] - 1)
            new_rowval[k] = rowval[l]
            new_nzval[k] = nzval[l]
            k += 1
        end
    end
    new_colptr[end] = nnz(A) + 1
    return SparseMatrixCSC(A.m, A.n, new_colptr, new_rowval, new_nzval), col_perm
end
