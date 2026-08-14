"""
    GPUSparseMatrixELL

Every row is padded to the length of the longest row, so a single unusually dense row makes
this format allocate `m × d` dense storage, where `d` is the maximum number of nonzeros in a
row. Prefer [`GPUSparseMatrixCSR`](@ref) or [`GPUSparseMatrixCOO`](@ref) when row lengths vary
widely.

# Fields

$(TYPEDFIELDS)
"""
struct GPUSparseMatrixELL{
        T <: Number,
        Ti <: Integer,
        Mv <: DenseMatrix{T},
        Mi <: DenseMatrix{Ti},
    } <: AbstractSparseMatrix{T, Ti}
    m::Int
    n::Int
    colval::Mi
    nzval::Mv
end

Base.size(A::GPUSparseMatrixELL) = (A.m, A.n)

SparseArrays.nnz(A::GPUSparseMatrixELL) = sum(!=(0), A.colval)

function Base.getindex(
        A::GPUSparseMatrixELL{T, Ti}, i::Integer, j::Integer
    ) where {T, Ti}
    (; colval, nzval) = A
    k2 = size(colval, 2)
    colval_row = view(colval, i, :)
    while k2 > 0 && colval_row[k2] == 0
        k2 -= 1
    end
    k = searchsortedfirst(view(colval_row, 1:k2), j)
    if 1 <= k <= length(colval_row) && colval_row[k] == j
        return nzval[i, k]
    else
        return zero(T)
    end
end

function KernelAbstractions.get_backend(A::GPUSparseMatrixELL)
    return common_backend(A.colval, A.nzval)
end

function Adapt.adapt_structure(to, A::GPUSparseMatrixELL)
    return GPUSparseMatrixELL(
        A.m,
        A.n,
        adapt(to, A.colval),
        adapt(to, A.nzval)
    )
end

function GPUSparseMatrixELL(A::SparseMatrixCSC{T, Ti}) where {T, Ti}
    m, n = size(A)
    A_csr = GPUSparseMatrixCSR(A)
    d = m == 0 ? 0 : maximum(diff(A_csr.rowptr))
    colval = similar(A.rowval, m, d)
    nzval = similar(A.nzval, m, d)
    fill!(colval, zero(Ti))
    fill!(nzval, zero(T))
    for i in axes(A, 1)
        k1, k2 = A_csr.rowptr[i], A_csr.rowptr[i + 1] - 1
        for k in k1:k2
            colval[i, k - k1 + 1] = A_csr.colval[k]
            nzval[i, k - k1 + 1] = A_csr.nzval[k]
        end
    end
    return GPUSparseMatrixELL(m, n, colval, nzval)
end

function SparseArrays.SparseMatrixCSC(A::GPUSparseMatrixELL)
    I = Matrix(map(ind -> Tuple(ind)[1], CartesianIndices(A.colval)))
    J = Matrix(A.colval)
    V = Matrix(A.nzval)
    inds = J .> 0
    return sparse(I[inds], J[inds], V[inds], A.m, A.n)
end

function sametype_transpose(A::GPUSparseMatrixELL)
    At = SparseMatrixCSC(transpose(SparseMatrixCSC(A)))
    return adapt(get_backend(A), GPUSparseMatrixELL(At))
end

@kernel function spmv_ell!(
        c::DenseVector{T},
        A_colval::AbstractMatrix{Ti},
        A_nzval::AbstractMatrix{T},
        b::DenseVector{T},
        α::Number,
        β::Number
    ) where {T, Ti}
    i = @index(Global, Linear)
    s = zero(T)
    for k in axes(A_colval, 2)
        j = A_colval[i, k]
        if j != zero(Ti)
            s += A_nzval[i, k] * b[j]
        end
    end
    c[i] = α * s + β * c[i]
end

function LinearAlgebra.mul!(
        c::V,
        A::GPUSparseMatrixELL{T, Ti},
        b::V,
        α::Number,
        β::Number
    ) where {T <: Number, Ti, V <: DenseVector{T}}
    check_mul_dims(c, A, b)
    backend = common_backend(c, A, b)
    kernel! = spmv_ell!(backend)
    α_is_one = isone(α)
    β_is_zero = iszero(β)
    if α_is_one && β_is_zero
        kernel!(c, A.colval, A.nzval, b, One(), Zero(); ndrange = size(A, 1))
    elseif α_is_one
        kernel!(c, A.colval, A.nzval, b, One(), β; ndrange = size(A, 1))
    elseif β_is_zero
        kernel!(c, A.colval, A.nzval, b, α, Zero(); ndrange = size(A, 1))
    else
        kernel!(c, A.colval, A.nzval, b, α, β; ndrange = size(A, 1))
    end
    return c
end

@kernel function spmm_ell!(
        c::DenseMatrix{T},
        A_colval::AbstractMatrix{Ti},
        A_nzval::AbstractMatrix{T},
        b::DenseMatrix{T},
        α::Number,
        β::Number
    ) where {T, Ti}
    i, batch_idx = @index(Global, NTuple)
    s = zero(T)
    for k in axes(A_colval, 2)
        j = A_colval[i, k]
        if j != zero(Ti)
            s += A_nzval[i, k] * b[j, batch_idx]
        end
    end
    c[i, batch_idx] = α * s + β * c[i, batch_idx]
end

function LinearAlgebra.mul!(
        c::DenseMatrix{T},
        A::GPUSparseMatrixELL{T},
        b::DenseMatrix{T},
        α::Number,
        β::Number
    ) where {T <: Number}
    check_mul_dims(c, A, b)
    backend = common_backend(c, A, b)
    kernel! = spmm_ell!(backend)
    α_is_one = isone(α)
    β_is_zero = iszero(β)
    if α_is_one && β_is_zero
        kernel!(c, A.colval, A.nzval, b, One(), Zero(); ndrange = size(c))
    elseif α_is_one
        kernel!(c, A.colval, A.nzval, b, One(), β; ndrange = size(c))
    elseif β_is_zero
        kernel!(c, A.colval, A.nzval, b, α, Zero(); ndrange = size(c))
    else
        kernel!(c, A.colval, A.nzval, b, α, β; ndrange = size(c))
    end
    return c
end
