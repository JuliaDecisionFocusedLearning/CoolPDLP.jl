"""
    GPUSparseMatrixCOO

# Fields

$(TYPEDFIELDS)
"""
struct GPUSparseMatrixCOO{
        T <: Number,
        # no `<: Integer` bound: under `Reactant` a traced index array has element type
        # `TracedRNumber{Int32}`, a `Number` that is not an `Integer`, and bounding `Ti`
        # would make the whole matrix untraceable
        Ti,
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

function SparseArrays.SparseMatrixCSC(A::GPUSparseMatrixCOO)
    return sparse(Vector(A.rowval), Vector(A.colval), Vector(A.nzval), A.m, A.n)
end

function GPUSparseMatrixCOO(A::SparseMatrixCSC{T, Ti}) where {T, Ti}
    rowval, colval, nzval = findnz(A)
    return GPUSparseMatrixCOO(A.m, A.n, rowval, colval, nzval)
end

function sametype_transpose(A::GPUSparseMatrixCOO)
    return GPUSparseMatrixCOO(A.n, A.m, A.colval, A.rowval, A.nzval)
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
    return spmul!(c, A, b, α, β)
end

function spmul!(
        c::V,
        A::GPUSparseMatrixCOO{T, Ti, V},
        b::V,
        α::Number,
        β::Number
    ) where {T <: Number, Ti, V <: DenseVector{T}}
    check_mul_dims(c, A, b)
    backend = common_backend(c, A, b)
    kernel! = spmv_coo!(backend)
    if iszero(β)
        zero!(c)
    elseif !isone(β)
        c .*= β
    end
    if isone(α)
        kernel!(c, A.rowval, A.colval, A.nzval, b, One(); ndrange = length(A.nzval))
    else
        kernel!(c, A.rowval, A.colval, A.nzval, b, α; ndrange = length(A.nzval))
    end
    return c
end

@kernel function spmm_coo!(
        c::DenseVector{T},
        A_rowval::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::DenseVector{T},
        b::DenseVector{T},
        m::Int,
        n::Int,
        α::Number,
    ) where {T, Ti}
    # `c` and `b` are the batches flattened column-major, see `spmm_csr!`
    k, batch_idx = @index(Global, NTuple)
    i, j, v = A_rowval[k], A_colval[k], A_nzval[k]
    Atomix.@atomic c[i + (batch_idx - 1) * m] += α * v * b[j + (batch_idx - 1) * n]
end

function LinearAlgebra.mul!(
        c::DenseMatrix{T},
        A::GPUSparseMatrixCOO{T},
        b::DenseMatrix{T},
        α::Number,
        β::Number
    ) where {T <: Number}
    return spmul!(c, A, b, α, β)
end

function spmul!(
        c::DenseMatrix{T},
        A::GPUSparseMatrixCOO{T},
        b::DenseMatrix{T},
        α::Number,
        β::Number
    ) where {T <: Number}
    check_mul_dims(c, A, b)
    spmm!(vec(c), A, vec(b), size(c, 2), α, β)
    return c
end

function spmm!(
        c::DenseVector{T},
        A::GPUSparseMatrixCOO{T},
        b::DenseVector{T},
        nb::Int,
        α::Number,
        β::Number
    ) where {T <: Number}
    backend = common_backend(c, A, b)
    m, n = size(A)
    kernel! = spmm_coo!(backend)
    if iszero(β)
        zero!(c)
    elseif !isone(β)
        c .*= β
    end
    ndrange = (length(A.nzval), nb)
    if isone(α)
        kernel!(c, A.rowval, A.colval, A.nzval, b, m, n, One(); ndrange)
    else
        kernel!(c, A.rowval, A.colval, A.nzval, b, m, n, α; ndrange)
    end
    return c
end
