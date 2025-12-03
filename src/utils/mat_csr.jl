"""
    GPUSparseMatrixCSR

# Fields

$(TYPEDFIELDS)
"""
struct GPUSparseMatrixCSR{
        T <: Number,
        Ti <: Integer,
        V <: DenseVector{T},
        Vi <: DenseVector{Ti},
    } <: AbstractSparseMatrix{T, Ti}
    m::Int
    n::Int
    rowptr::Vi
    colval::Vi
    nzval::V
end

Base.size(A::GPUSparseMatrixCSR) = (A.m, A.n)

SparseArrays.nnz(A::GPUSparseMatrixCSR) = length(A.nzval)
SparseArrays.nonzeros(A::GPUSparseMatrixCSR) = A.nzval

function Base.getindex(
        A::GPUSparseMatrixCSR{T, Ti}, i::Integer, j::Integer
    ) where {T, Ti}
    (; rowptr, colval, nzval) = A
    k1 = rowptr[i]
    k2 = rowptr[i + 1] - 1
    if k1 > k2
        return zero(T)
    else
        k = k1 + searchsortedfirst(view(colval, k1:k2), j) - 1
        if k > k2 || colval[k] != j
            return zero(T)
        else
            return nzval[k]
        end
    end
end

function KernelAbstractions.get_backend(A::GPUSparseMatrixCSR)
    return common_backend(A.rowptr, A.colval, A.nzval)
end

function Adapt.adapt_structure(to, A::GPUSparseMatrixCSR)
    return GPUSparseMatrixCSR(
        A.m,
        A.n,
        adapt(to, A.rowptr),
        adapt(to, A.colval),
        adapt(to, A.nzval)
    )
end

function GPUSparseMatrixCSR(A::SparseMatrixCSC{T, Ti}) where {T, Ti}
    At = sparse(transpose(A))
    return GPUSparseMatrixCSR(At.n, At.m, At.colptr, At.rowval, At.nzval)
end

@kernel function spmv_csr!(
        c::DenseVector{T},
        A_rowptr::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::DenseVector{T},
        b::DenseVector{T},
        α::Number,
        β::Number
    ) where {T, Ti}
    i = @index(Global, Linear)
    s = zero(T)
    for k in A_rowptr[i]:(A_rowptr[i + Ti(1)] - Ti(1))
        j = A_colval[k]
        s += A_nzval[k] * b[j]
    end
    c[i] = α * s + β * c[i]
end

function LinearAlgebra.mul!(
        c::V,
        A::GPUSparseMatrixCSR{T, Ti, V},
        b::V,
        α::Number,
        β::Number
    ) where {T <: Number, Ti, V <: DenseVector{T}}
    backend = common_backend(c, A, b)
    kernel! = spmv_csr!(backend)
    kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β; ndrange = size(A, 1))
    return c
end
