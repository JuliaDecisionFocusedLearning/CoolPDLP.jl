abstract type AbstractGPUSparseArrayCSR{T, Ti, N} <: SparseArrays.AbstractSparseArray{T, Ti, N} end

"""
    GPUSparseMatrixCSR

# Fields

$(TYPEDFIELDS)
"""
struct GPUSparseMatrixCSR{
        T <: Number,
        Ti <: Integer,
        V <: AbstractVector{T},
        Vi <: DenseVector{Ti},
    } <: AbstractGPUSparseArrayCSR{T, Ti, 2}
    m::Int
    n::Int
    rowptr::Vi
    colval::Vi
    nzval::V
end

Adapt.@adapt_structure GPUSparseMatrixCSR
Base.size(A::GPUSparseMatrixCSR) = (A.m, A.n)

"""
    BatchedGPUSparseMatrixCSR

# Fields

$(TYPEDFIELDS)
"""
struct BatchedGPUSparseMatrixCSR{
        T <: Number,
        Ti <: Integer,
        V <: DenseMatrix{T},
        Vi <: DenseVector{Ti},
    } <: AbstractGPUSparseArrayCSR{T, Ti, 3}
    m::Int
    n::Int
    rowptr::Vi
    colval::Vi
    nzval::V
end

Adapt.@adapt_structure BatchedGPUSparseMatrixCSR
Base.size(A::BatchedGPUSparseMatrixCSR) = (A.m, A.n, size(A.nzval, 2))

SparseArrays.nnz(A::AbstractGPUSparseArrayCSR) = length(A.nzval)
SparseArrays.nonzeros(A::AbstractGPUSparseArrayCSR) = A.nzval

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

function Base.view(
        A::BatchedGPUSparseMatrixCSR{T, Ti}, ::Colon, ::Colon, k::Integer,
    ) where {T, Ti}
    return GPUSparseMatrixCSR(
        A.m,
        A.n,
        A.rowptr,
        A.colval,
        view(A.nzval, :, k)
    )
end
function Base.getindex(
        A::BatchedGPUSparseMatrixCSR{T, Ti}, i::Integer, j::Integer, k::Integer,
    ) where {T, Ti}
    return view(A, :, :, k)[i, j]
end

function KernelAbstractions.get_backend(A::AbstractGPUSparseArrayCSR)
    return common_backend(A.rowptr, A.colval, A.nzval)
end

function GPUSparseMatrixCSR(A::SparseMatrixCSC{T, Ti}) where {T, Ti}
    At_csc = SparseMatrixCSC(transpose(A))
    return GPUSparseMatrixCSR(At_csc.n, At_csc.m, At_csc.colptr, At_csc.rowval, At_csc.nzval)
end

function SparseArrays.SparseMatrixCSC(A::GPUSparseMatrixCSR)
    At_csc = SparseMatrixCSC(A.n, A.m, Vector(A.rowptr), Vector(A.colval), Vector(A.nzval))
    return SparseMatrixCSC(transpose(At_csc))
end

function sametype_transpose(A::GPUSparseMatrixCSR)
    A_csc = SparseMatrixCSC(A)
    return adapt(
        get_backend(A),
        GPUSparseMatrixCSR(A_csc.n, A_csc.m, A_csc.colptr, A_csc.rowval, A_csc.nzval)
    )
end

@kernel function spmv_csr!(
        c::AbstractVector{T},
        A_rowptr::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::AbstractVector{T},
        b::AbstractVector{T},
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

@kernel function spbmv_csr!(
        c::DenseMatrix{T},
        A_rowptr::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::DenseMatrix{T},
        b::DenseVector{T},
        α::Number,
        β::Number
    ) where {T, Ti}
    i, batch_idx = @index(Global, NTuple)
    s = zero(T)
    for k in A_rowptr[i]:(A_rowptr[i + Ti(1)] - Ti(1))
        j = A_colval[k]
        s += A_nzval[k, batch_idx] * b[j]
    end
    c[i, batch_idx] = α * s + β * c[i, batch_idx]
end

function LinearAlgebra.mul!(
        c::AbstractVector{T},
        A::GPUSparseMatrixCSR{T},
        b::AbstractVector{T},
        α::Number,
        β::Number
    ) where {T <: Number}
    backend = common_backend(c, A, b)
    kernel! = spmv_csr!(backend)
    kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β; ndrange = size(A, 1))
    return c
end

function LinearAlgebra.mul!(
        c::M,
        A::BatchedGPUSparseMatrixCSR{T, Ti, M},
        b::V,
        α::Number,
        β::Number
    ) where {T <: Number, Ti, M <: DenseMatrix{T}, V <: DenseVector{T}}
    backend = common_backend(c, A, b)
    kernel! = spbmv_csr!(backend)
    kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β; ndrange = size(c))
    return c
end

@kernel function spmm_csr!(
        c::AbstractMatrix{T},
        A_rowptr::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::AbstractVector{T},
        b::AbstractMatrix{T},
        α::Number,
        β::Number
    ) where {T, Ti}
    i, batch_idx = @index(Global, NTuple)
    s = zero(T)
    for k in A_rowptr[i]:(A_rowptr[i + Ti(1)] - Ti(1))
        j = A_colval[k]
        s += A_nzval[k] * b[j, batch_idx]
    end
    c[i, batch_idx] = α * s + β * c[i, batch_idx]
end

@kernel function spbmm_csr!(
        c::DenseMatrix{T},
        A_rowptr::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::DenseMatrix{T},
        b::DenseMatrix{T},
        α::Number,
        β::Number
    ) where {T, Ti}
    i, batch_idx = @index(Global, NTuple)
    s = zero(T)
    for k in A_rowptr[i]:(A_rowptr[i + Ti(1)] - Ti(1))
        j = A_colval[k]
        s += A_nzval[k, batch_idx] * b[j, batch_idx]
    end
    c[i, batch_idx] = α * s + β * c[i, batch_idx]
end

function LinearAlgebra.mul!(
        c::AbstractMatrix{T},
        A::GPUSparseMatrixCSR{T},
        b::AbstractMatrix{T},
        α::Number,
        β::Number
    ) where {T <: Number}
    backend = common_backend(c, A, b)
    kernel! = spmm_csr!(backend)
    kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β; ndrange = size(c))
    return c
end

function LinearAlgebra.mul!(
        c::M,
        A::BatchedGPUSparseMatrixCSR{T, Ti, M},
        b::M,
        α::Number,
        β::Number
    ) where {T <: Number, Ti, M <: DenseMatrix{T}}
    backend = common_backend(c, A, b)
    kernel! = spbmm_csr!(backend)
    kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β; ndrange = size(c))
    return c
end
