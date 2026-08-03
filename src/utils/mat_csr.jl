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

"""
    BatchedGPUSparseMatrixCSR(As)

Stack matrices sharing a single sparsity pattern into one batched matrix.
"""
function BatchedGPUSparseMatrixCSR(As::AbstractVector{<:AbstractMatrix})
    ref = GPUSparseMatrixCSR(first(As))
    nzval = stack(As) do A
        A === first(As) && return ref.nzval
        csr = GPUSparseMatrixCSR(A)
        if csr.rowptr != ref.rowptr || csr.colval != ref.colval
            throw(
                ArgumentError("The instances of a batch must share a single sparsity pattern")
            )
        end
        return csr.nzval
    end
    return BatchedGPUSparseMatrixCSR(ref.m, ref.n, ref.rowptr, ref.colval, nzval)
end

"""
    BatchedGPUSparseMatrixCSR(A, nbinstances)

Repeat `A` into a batch of `nbinstances` identical matrices.
"""
function BatchedGPUSparseMatrixCSR(A::AbstractMatrix, nbinstances::Integer)
    (; m, n, rowptr, colval, nzval) = GPUSparseMatrixCSR(A)
    return BatchedGPUSparseMatrixCSR(m, n, rowptr, colval, repeat(nzval, 1, nbinstances))
end

function SparseArrays.SparseMatrixCSC(A::GPUSparseMatrixCSR)
    At_csc = SparseMatrixCSC(A.n, A.m, Vector(A.rowptr), Vector(A.colval), Vector(A.nzval))
    return SparseMatrixCSC(transpose(At_csc))
end

function sametype_transpose(A::BatchedGPUSparseMatrixCSR)
    instances = map(axes(A, 3)) do i
        return SparseMatrixCSC(transpose(SparseMatrixCSC(view(A, :, :, i))))
    end
    return adapt(get_backend(A), BatchedGPUSparseMatrixCSR(instances))
end

function sametype_transpose(A::GPUSparseMatrixCSR)
    A_csc = SparseMatrixCSC(A)
    return adapt(
        get_backend(A),
        GPUSparseMatrixCSR(A_csc.n, A_csc.m, A_csc.colptr, A_csc.rowval, A_csc.nzval)
    )
end

@kernel function spmv_csr!(
        c::StridedVector{T},
        A_rowptr::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::AbstractVector{T},
        b::StridedVector{T},
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

"""
    batchval(v, k, batch_idx)

Read entry `k` of the `batch_idx`-th instance of `v`, which holds either one value per instance or a single value shared by the whole batch.

Both methods resolve at compile time, so a kernel using them costs the same as one indexing its operands directly.
"""
@inline batchval(v::AbstractVector, k, ::Integer) = v[k]
@inline batchval(m::AbstractMatrix, k, batch_idx::Integer) = m[k, batch_idx]

@kernel function spmm_csr!(
        c::StridedMatrix{T},
        A_rowptr::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::AbstractVecOrMat{T},
        b::StridedVecOrMat{T},
        α::Number,
        β::Number
    ) where {T, Ti}
    i, batch_idx = @index(Global, NTuple)
    s = zero(T)
    for k in A_rowptr[i]:(A_rowptr[i + Ti(1)] - Ti(1))
        j = A_colval[k]
        s += batchval(A_nzval, k, batch_idx) * batchval(b, j, batch_idx)
    end
    c[i, batch_idx] = α * s + β * c[i, batch_idx]
end

function LinearAlgebra.mul!(
        c::StridedVector{T},
        A::GPUSparseMatrixCSR{T},
        b::StridedVector{T},
        α::Number,
        β::Number
    ) where {T <: Number}
    backend = common_backend(c, A, b)
    kernel! = spmv_csr!(backend)
    α_is_one = isone(α)
    β_is_zero = iszero(β)
    if α_is_one && β_is_zero
        kernel!(c, A.rowptr, A.colval, A.nzval, b, One(), Zero(); ndrange = size(A, 1))
    elseif α_is_one
        kernel!(c, A.rowptr, A.colval, A.nzval, b, One(), β; ndrange = size(A, 1))
    elseif β_is_zero
        kernel!(c, A.rowptr, A.colval, A.nzval, b, α, Zero(); ndrange = size(A, 1))
    else
        kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β; ndrange = size(A, 1))
    end
    return c
end

function LinearAlgebra.mul!(
        c::StridedMatrix{T},
        A::BatchedGPUSparseMatrixCSR{T},
        b::StridedVector{T},
        α::Number,
        β::Number
    ) where {T <: Number}
    backend = common_backend(c, A, b)
    kernel! = spmm_csr!(backend)
    kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β; ndrange = size(c))
    return c
end

function LinearAlgebra.mul!(
        c::StridedMatrix{T},
        A::GPUSparseMatrixCSR{T},
        b::StridedMatrix{T},
        α::Number,
        β::Number
    ) where {T <: Number}
    backend = common_backend(c, A, b)
    kernel! = spmm_csr!(backend)
    kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β; ndrange = size(c))
    return c
end

function LinearAlgebra.mul!(
        c::StridedMatrix{T},
        A::BatchedGPUSparseMatrixCSR{T},
        b::StridedMatrix{T},
        α::Number,
        β::Number
    ) where {T <: Number}
    backend = common_backend(c, A, b)
    kernel! = spmm_csr!(backend)
    kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β; ndrange = size(c))
    return c
end
