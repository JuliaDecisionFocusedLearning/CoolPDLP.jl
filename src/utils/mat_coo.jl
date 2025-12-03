"""
    GPUSparseMatrixCOO

# Fields

$(TYPEDFIELDS)
"""
struct GPUSparseMatrixCOO{
        T <: Number,
        Ti <: Integer,
        V <: DenseVector{T},
        Vi <: DenseVector{Ti},
    } <: AbstractSparseMatrix{T, Ti}
    m::Int
    n::Int
    rowval::Vi
    colval::Vi
    nzval::V
end

Base.size(A::GPUSparseMatrixCOO) = (A.m, A.n)

SparseArrays.nnz(A::GPUSparseMatrixCOO) = length(A.nzval)
SparseArrays.nonzeros(A::GPUSparseMatrixCOO) = A.nzval

function Base.getindex(A::GPUSparseMatrixCOO{T}, i::Integer, j::Integer) where {T}
    (; rowval, colval, nzval) = A
    for k in eachindex(rowval, colval, nzval)
        if rowval[k] == i && colval[k] == j
            return nzval[k]
        end
    end
    return zero(T)
end

function KernelAbstractions.get_backend(A::GPUSparseMatrixCOO)
    return common_backend(A.rowval, A.colval, A.nzval)
end

function Adapt.adapt_structure(to, A::GPUSparseMatrixCOO)
    return GPUSparseMatrixCOO(
        A.m,
        A.n,
        adapt(to, A.rowval),
        adapt(to, A.colval),
        adapt(to, A.nzval)
    )
end

function GPUSparseMatrixCOO(A::SparseMatrixCSC{T, Ti}) where {T, Ti}
    rowval, colval, nzval = findnz(A)
    return GPUSparseMatrixCOO(A.m, A.n, rowval, colval, nzval)
end

@kernel function spmv_coo!(
        c::DenseVector{T},
        A_rowval::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::DenseVector{T},
        b::DenseVector{T},
        α::Number,
    ) where {T, Ti}
    k = @index(Global, Linear)
    i, j, v = A_rowval[k], A_colval[k], A_nzval[k]
    Atomix.@atomic c[i] += α * v * b[j]
end

function LinearAlgebra.mul!(
        c::V,
        A::GPUSparseMatrixCOO{T, Ti, V},
        b::V,
        α::Number,
        β::Number
    ) where {T <: Number, Ti, V <: DenseVector{T}}
    c .*= β
    backend = common_backend(c, A, b)
    kernel! = spmv_coo!(backend)
    kernel!(c, A.rowval, A.colval, A.nzval, b, α; ndrange = length(A.nzval))
    return c
end
