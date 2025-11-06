abstract type AbstractDeviceSparseMatrix{Tv, Ti} <: AbstractMatrix{Tv} end

Base.size(A::AbstractDeviceSparseMatrix) = (A.m, A.n)

"""
    DeviceSparseMatrixCOO

# Fields

$(TYPEDFIELDS)
"""
struct DeviceSparseMatrixCOO{
        Tv <: Number,
        Ti <: Integer,
        Vv <: AbstractVector{Tv},
        Vi <: AbstractVector{Ti},
    } <: AbstractDeviceSparseMatrix{Tv, Ti}
    m::Ti
    n::Ti
    rowval::Vi
    colval::Vi
    nzval::Vv
end

function KernelAbstractions.get_backend(A::DeviceSparseMatrixCOO)
    return common_backend(A.rowval, A.colval, A.nzval)
end

function Adapt.adapt_structure(to, A::DeviceSparseMatrixCOO)
    return DeviceSparseMatrixCOO(
        A.m,
        A.n,
        adapt(to, A.rowval),
        adapt(to, A.colval),
        adapt(to, A.nzval)
    )
end

function DeviceSparseMatrixCOO(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    rowval, colval, nzval = findnz(A)
    return DeviceSparseMatrixCOO(Ti(A.m), Ti(A.n), rowval, colval, nzval)
end

function Base.getindex(A::DeviceSparseMatrixCOO{Tv}, i::Integer, j::Integer) where {Tv}
    (; rowval, colval, nzval) = A
    for k in eachindex(rowval, colval, nzval)
        if rowval[k] == i && colval[k] == j
            return nzval[k]
        end
    end
    return zero(Tv)
end

@kernel function spmv_coo!(
        c::AbstractVector{Tv},
        A_rowval::AbstractVector{Ti},
        A_colval::AbstractVector{Ti},
        A_nzval::AbstractVector{Tv},
        b::AbstractVector{Tv},
        α::Number,
    ) where {Tv, Ti}
    k = @index(Global, Linear)
    i, j, v = A_rowval[k], A_colval[k], A_nzval[k]
    Atomix.@atomic c[i] += α * v * b[j]
end

function LinearAlgebra.mul!(
        c::AbstractVector,
        A::DeviceSparseMatrixCOO,
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
    DeviceSparseMatrixCSR

# Fields

$(TYPEDFIELDS)
"""
struct DeviceSparseMatrixCSR{
        Tv <: Number,
        Ti <: Integer,
        Vv <: AbstractVector{Tv},
        Vi <: AbstractVector{Ti},
    } <: AbstractDeviceSparseMatrix{Tv, Ti}
    m::Ti
    n::Ti
    rowptr::Vi
    colval::Vi
    nzval::Vv
end

function KernelAbstractions.get_backend(A::DeviceSparseMatrixCSR)
    return common_backend(A.rowptr, A.colval, A.nzval)
end

function Adapt.adapt_structure(to, A::DeviceSparseMatrixCSR)
    return DeviceSparseMatrixCSR(
        A.m,
        A.n,
        adapt(to, A.rowptr),
        adapt(to, A.colval),
        adapt(to, A.nzval)
    )
end

function DeviceSparseMatrixCSR(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    At = sparse(transpose(A))
    return DeviceSparseMatrixCSR(Ti(At.n), Ti(At.m), At.colptr, At.rowval, At.nzval)
end

function Base.getindex(
        A::DeviceSparseMatrixCSR{Tv, Ti}, i::Integer, j::Integer
    ) where {Tv, Ti}
    (; rowptr, colval, nzval) = A
    k1 = rowptr[i]
    k2 = rowptr[i + 1] - 1
    if k1 > k2
        return zero(Tv)
    else
        k = k1 + searchsortedfirst(view(colval, k1:k2), j) - 1
        if k > k2 || colval[k] != j
            return zero(Tv)
        else
            return nzval[k]
        end
    end
end

@kernel function spmv_csr!(
        c::AbstractVector{Tv},
        A_rowptr::AbstractVector{Ti},
        A_colval::AbstractVector{Ti},
        A_nzval::AbstractVector{Tv},
        b::AbstractVector{Tv},
        α::Number,
        β::Number
    ) where {Tv, Ti}
    i = @index(Global, Linear)
    s = zero(Tv)
    for k in A_rowptr[i]:(A_rowptr[i + Ti(1)] - Ti(1))
        j = A_colval[k]
        s += A_nzval[k] * b[j]
    end
    c[i] = α * s + β * c[i]
end

function LinearAlgebra.mul!(
        c::AbstractVector,
        A::DeviceSparseMatrixCSR,
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
    DeviceSparseMatrixELL

# Fields

$(TYPEDFIELDS)
"""
struct DeviceSparseMatrixELL{
        Tv <: Number,
        Ti <: Integer,
        Mv <: AbstractMatrix{Tv},
        Mi <: AbstractMatrix{Ti},
    } <: AbstractDeviceSparseMatrix{Tv, Ti}
    m::Ti
    n::Ti
    colval::Mi
    nzval::Mv
end

function KernelAbstractions.get_backend(A::DeviceSparseMatrixELL)
    return common_backend(A.colval, A.nzval)
end

function Adapt.adapt_structure(to, A::DeviceSparseMatrixELL)
    return DeviceSparseMatrixELL(
        A.m,
        A.n,
        adapt(to, A.colval),
        adapt(to, A.nzval)
    )
end

function DeviceSparseMatrixELL(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    m, n = size(A)
    A_csr = DeviceSparseMatrixCSR(A)
    d = maximum(diff(A_csr.rowptr))
    colval = similar(A.rowval, m, d)
    nzval = similar(A.nzval, m, d)
    fill!(colval, zero(Ti))
    fill!(nzval, zero(Tv))
    for i in axes(A, 1)
        k1, k2 = A_csr.rowptr[i], A_csr.rowptr[i + 1] - 1
        for k in k1:k2
            colval[i, k - k1 + 1] = A_csr.colval[k]
            nzval[i, k - k1 + 1] = A_csr.nzval[k]
        end
    end
    return DeviceSparseMatrixELL(m, n, colval, nzval)
end

function Base.getindex(
        A::DeviceSparseMatrixELL{Tv, Ti}, i::Integer, j::Integer
    ) where {Tv, Ti}
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
        return zero(Tv)
    end
end

@kernel function spmv_ell!(
        c::AbstractVector{Tv},
        A_colval::AbstractMatrix{Ti},
        A_nzval::AbstractMatrix{Tv},
        b::AbstractVector{Tv},
        α::Number,
        β::Number
    ) where {Tv, Ti}
    i = @index(Global, Linear)
    s = zero(Tv)
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
        A::DeviceSparseMatrixELL,
        b::AbstractVector,
        α::Number,
        β::Number
    )
    backend = common_backend(c, A, b)
    kernel! = spmv_ell!(backend)
    kernel!(c, A.colval, A.nzval, b, α, β; ndrange = size(A, 1))
    return c
end
