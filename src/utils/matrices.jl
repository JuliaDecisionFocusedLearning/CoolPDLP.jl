abstract type AbstractGPUSparseMatrix{T, Ti} <: AbstractMatrix{T} end

Base.size(A::AbstractGPUSparseMatrix) = (A.m, A.n)

"""
    GPUSparseMatrixCOO

# Fields

$(TYPEDFIELDS)
"""
struct GPUSparseMatrixCOO{
        T <: Number,
        Ti <: Integer,
        Vv <: AbstractVector{T},
        Vi <: AbstractVector{Ti},
    } <: AbstractGPUSparseMatrix{T, Ti}
    m::Int
    n::Int
    rowval::Vi
    colval::Vi
    nzval::Vv
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

function Base.getindex(A::GPUSparseMatrixCOO{T}, i::Integer, j::Integer) where {T}
    (; rowval, colval, nzval) = A
    for k in eachindex(rowval, colval, nzval)
        if rowval[k] == i && colval[k] == j
            return nzval[k]
        end
    end
    return zero(T)
end

SparseArrays.nnz(A::GPUSparseMatrixCOO) = length(A.nzval)

@kernel function spmv_coo!(
        c::AbstractVector{T},
        A_rowval::AbstractVector{Ti},
        A_colval::AbstractVector{Ti},
        A_nzval::AbstractVector{T},
        b::AbstractVector{T},
        α::Number,
    ) where {T, Ti}
    k = @index(Global, Linear)
    i, j, v = A_rowval[k], A_colval[k], A_nzval[k]
    Atomix.@atomic c[i] += α * v * b[j]
end

function LinearAlgebra.mul!(
        c::AbstractVector,
        A::GPUSparseMatrixCOO,
        b::AbstractVector,
        α::Number,
        β::Number
    )
    c .*= β
    backend = common_backend(c, A, b)
    kernel! = spmv_coo!(backend)
    kernel!(c, A.rowval, A.colval, A.nzval, b, α; ndrange = length(A.nzval))
    return c
end


"""
    GPUSparseMatrixCSR

# Fields

$(TYPEDFIELDS)
"""
struct GPUSparseMatrixCSR{
        T <: Number,
        Ti <: Integer,
        Vv <: AbstractVector{T},
        Vi <: AbstractVector{Ti},
    } <: AbstractGPUSparseMatrix{T, Ti}
    m::Int
    n::Int
    rowptr::Vi
    colval::Vi
    nzval::Vv
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

SparseArrays.nnz(A::GPUSparseMatrixCSR) = length(A.nzval)

@kernel function spmv_csr!(
        c::AbstractVector{T},
        A_rowptr::AbstractVector{Ti},
        A_colval::AbstractVector{Ti},
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

function LinearAlgebra.mul!(
        c::AbstractVector,
        A::GPUSparseMatrixCSR,
        b::AbstractVector,
        α::Number,
        β::Number
    )
    backend = common_backend(c, A, b)
    kernel! = spmv_csr!(backend)
    kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β; ndrange = size(A, 1))
    return c
end

"""
    GPUSparseMatrixELL

# Fields

$(TYPEDFIELDS)
"""
struct GPUSparseMatrixELL{
        T <: Number,
        Ti <: Integer,
        Mv <: AbstractMatrix{T},
        Mi <: AbstractMatrix{Ti},
    } <: AbstractGPUSparseMatrix{T, Ti}
    m::Int
    n::Int
    colval::Mi
    nzval::Mv
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
    d = maximum(diff(A_csr.rowptr))
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

SparseArrays.nnz(A::GPUSparseMatrixELL) = sum(!=(0), A.colval)

@kernel function spmv_ell!(
        c::AbstractVector{T},
        A_colval::AbstractMatrix{Ti},
        A_nzval::AbstractMatrix{T},
        b::AbstractVector{T},
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
        c::AbstractVector,
        A::GPUSparseMatrixELL,
        b::AbstractVector,
        α::Number,
        β::Number
    )
    backend = common_backend(c, A, b)
    kernel! = spmv_ell!(backend)
    kernel!(c, A.colval, A.nzval, b, α, β; ndrange = size(A, 1))
    return c
end
