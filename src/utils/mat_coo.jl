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

function SparseArrays.SparseMatrixCSC(A::GPUSparseMatrixCOO)
    return sparse(Vector(A.rowval), Vector(A.colval), Vector(A.nzval), A.m, A.n)
end

function GPUSparseMatrixCOO(A::SparseMatrixCSC{T, Ti}) where {T, Ti}
    # `findnz(A)` lists the nonzeros column by column; transposing first lists them row by
    # row, which is what lets [`spmv_coo!`](@ref) accumulate a whole row before touching
    # `c`. Any order gives the right answer, this one just gives fewer atomics.
    At = SparseMatrixCSC(transpose(A))
    colval, rowval, nzval = findnz(At)
    return GPUSparseMatrixCOO(A.m, A.n, rowval, colval, nzval)
end

function sametype_transpose(A::GPUSparseMatrixCOO)
    # swapping the two index vectors would transpose in place and for free, but it would
    # leave the result ordered by its own columns, and the transpose is multiplied just as
    # often as the matrix itself, so it is worth rebuilding in row order
    return adapt(get_backend(A), GPUSparseMatrixCOO(SparseMatrixCSC(transpose(SparseMatrixCSC(A)))))
end

"""
    COO_SLICE

Number of consecutive nonzeros each work item of the COO kernels takes.

Two effects pull against each other. A work item pays one atomic update per row it touches,
so longer slices mean fewer atomics -- but it walks its slice one entry at a time, so the
lanes of a warp sit `COO_SLICE` entries apart while they do it, and longer slices cost
coalescing on every read. Four is where they balance: across fourteen MIPLIB instances and
three batch sizes it was within 1.08x of the best length per case (worst 1.71x), where
matching the slice to the average row length -- which sounds right, one atomic per row --
was 1.87x off and as much as 11.7x off on matrices with long rows.
"""
const COO_SLICE = 4

"""
    spmv_coo!(c, A_rowval, A_colval, A_nzval, b, α, Val(NPT))

Each work item consumes [`COO_SLICE`](@ref) consecutive nonzeros, sums the ones that share
a row, and issues a single atomic update per row it touches. Correct whatever order the nonzeros are
stored in; fast when they are grouped by row, which is how [`GPUSparseMatrixCOO`](@ref)
builds them.

Work is split by nonzero rather than by row, so a matrix with one enormous row costs no
more here than an even one -- unlike [`spmv_csr!`](@ref), whose sub-groups still have to
walk the longest row.

`c` must already hold `β * c` on entry.
"""
@kernel function spmv_coo!(
        c::DenseVector{T},
        A_rowval::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::DenseVector{T},
        b::DenseVector{T},
        α::Number,
        ::Val{NPT}
    ) where {T, Ti, NPT}
    t = @index(Global, Linear)
    nz = length(A_nzval)
    kstart = (t - 1) * NPT + 1
    if kstart <= nz
        kstop = min(kstart + NPT - 1, nz)
        k = kstart
        @inbounds while k <= kstop
            row = A_rowval[k]
            s = zero(T)
            while k <= kstop && A_rowval[k] == row
                s += A_nzval[k] * b[A_colval[k]]
                k += 1
            end
            Atomix.@atomic c[row] += α * s
        end
    end
end

"""
    launch_spmv_coo!(c, A, b, α, backend)

Launch [`spmv_coo!`](@ref) over one work item per [`COO_SLICE`](@ref) nonzeros.
"""
function launch_spmv_coo!(c, A::GPUSparseMatrixCOO, b, α::Number, backend)
    kernel! = spmv_coo!(backend)
    ndrange = cld(nnz(A), COO_SLICE)
    kernel!(c, A.rowval, A.colval, A.nzval, b, α, Val(COO_SLICE); ndrange)
    return nothing
end

function LinearAlgebra.mul!(
        c::V,
        A::GPUSparseMatrixCOO{T, Ti, V},
        b::V,
        α::Number,
        β::Number
    ) where {T <: Number, Ti, V <: DenseVector{T}}
    check_mul_dims(c, A, b)
    backend = common_backend(c, A, b)
    if iszero(β)
        zero!(c)
    elseif !isone(β)
        c .*= β
    end
    if isone(α)
        launch_spmv_coo!(c, A, b, One(), backend)
    else
        launch_spmv_coo!(c, A, b, α, backend)
    end
    return c
end

"""
    spmm_coo!(c, A_rowval, A_colval, A_nzval, b, α, Val(NPT))

Batched counterpart of [`spmv_coo!`](@ref), launched over `(cld(nnz, NPT), size(c, 2))`.
"""
@kernel function spmm_coo!(
        c::DenseMatrix{T},
        A_rowval::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::DenseVector{T},
        b::DenseMatrix{T},
        α::Number,
        ::Val{NPT}
    ) where {T, Ti, NPT}
    t, q = @index(Global, NTuple)
    nz = length(A_nzval)
    kstart = (t - 1) * NPT + 1
    if kstart <= nz
        kstop = min(kstart + NPT - 1, nz)
        k = kstart
        @inbounds while k <= kstop
            row = A_rowval[k]
            s = zero(T)
            while k <= kstop && A_rowval[k] == row
                s += A_nzval[k] * b[A_colval[k], q]
                k += 1
            end
            Atomix.@atomic c[row, q] += α * s
        end
    end
end

"""
    launch_spmm_coo!(c, A, b, α, backend)

Launch [`spmm_coo!`](@ref) over one work item per [`COO_SLICE`](@ref) nonzeros.
"""
function launch_spmm_coo!(c, A::GPUSparseMatrixCOO, b, α::Number, backend)
    kernel! = spmm_coo!(backend)
    ndrange = (cld(nnz(A), COO_SLICE), size(c, 2))
    kernel!(c, A.rowval, A.colval, A.nzval, b, α, Val(COO_SLICE); ndrange)
    return nothing
end

function LinearAlgebra.mul!(
        c::DenseMatrix{T},
        A::GPUSparseMatrixCOO{T},
        b::DenseMatrix{T},
        α::Number,
        β::Number
    ) where {T <: Number}
    check_mul_dims(c, A, b)
    backend = common_backend(c, A, b)
    if iszero(β)
        zero!(c)
    elseif !isone(β)
        c .*= β
    end
    if isone(α)
        launch_spmm_coo!(c, A, b, One(), backend)
    else
        launch_spmm_coo!(c, A, b, α, backend)
    end
    return c
end
