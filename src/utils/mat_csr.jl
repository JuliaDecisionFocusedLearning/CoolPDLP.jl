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

"""
    CSR_WORKGROUP

Workgroup size used by the cooperative CSR kernels. Fixed rather than left to the backend's
occupancy heuristic because [`spmv_csr_vector!`](@ref) sizes its local memory from it.
"""
const CSR_WORKGROUP = 256

"""
    MAX_SUBGROUP

Largest number of work items [`subgroup_size`](@ref) will put on a single row.
"""
const MAX_SUBGROUP = 32

"""
    subgroup_size(A)

Number of work items to assign to each row of `A`: the largest power of two no greater than
the average number of nonzeros per row, capped at [`MAX_SUBGROUP`](@ref).

One work item per row is the obvious way to write an SpMV, but it makes a row's cost
proportional to its length, so a single long row stalls the whole kernel while the rest of
the device idles. Real LP constraint matrices are strongly row-skewed -- a budget or
knapsack row touching every variable next to rows touching three -- and that cost the
one-work-item-per-row kernel a factor of 40-60 against cuSPARSE on the larger MIPLIB 2017
instances (`square41`, whose longest row holds 17,575 nonzeros against a mean of 338, ran
in 12.3ms where cuSPARSE took 0.2ms). Splitting each row across a sub-group both shortens
the longest row's critical path and makes the lanes read consecutive `nzval`/`colval`
entries.

Sizing the sub-group from the *average* row rather than the longest one is what keeps
matrices with short rows -- where a wide sub-group would leave most lanes idle -- from
regressing: on an L40S this rule was worth a geometric mean of 5.5x across a corpus of
random and MIPLIB matrices, with a worst case of 1.03x, where selecting on the longest row
instead lost 5x on a matrix averaging three nonzeros per row.
"""
function subgroup_size(A::GPUSparseMatrixCSR)
    m = size(A, 1)
    m == 0 && return 1
    return min(prevpow(2, max(nnz(A) ÷ m, 1)), MAX_SUBGROUP)
end

"""
    spmv_csr!(c, A_rowptr, A_colval, A_nzval, b, α, β)

One work item per row. Used when the rows are too short for [`spmv_csr_vector!`](@ref)'s
sub-groups to pay for themselves.
"""
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
    @inbounds for k in A_rowptr[i]:(A_rowptr[i + Ti(1)] - Ti(1))
        s += A_nzval[k] * b[A_colval[k]]
    end
    @inbounds c[i] = α * s + β * c[i]
end

"""
    spmv_csr_vector!(c, A_rowptr, A_colval, A_nzval, b, α, β, Val(S))

`S` work items per row, reduced through local memory. Launched over `S * size(A, 1)` work
items rounded up to a whole number of workgroups, so it must check `row` against the number
of rows itself.

The reduction uses local memory rather than sub-group shuffles, which keeps it backend
agnostic at the cost of a few barriers. `@synchronize` splits the kernel into regions on
the CPU backend, and only `@uniform` values, `@localmem` arrays and top-level
`x = @index(...)` statements survive across them, which is why the indices below are
recomputed in each region instead of being carried over.
"""
@kernel function spmv_csr_vector!(
        c::DenseVector{T},
        A_rowptr::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::DenseVector{T},
        b::DenseVector{T},
        α::Number,
        β::Number,
        ::Val{S}
    ) where {T, Ti, S}
    gid = @index(Global, Linear)
    lid = @index(Local, Linear)
    @uniform G = prod(@groupsize())
    @uniform nsteps = trailing_zeros(S)
    @uniform m = length(c)
    tmp = @localmem T (G,)

    row = (gid - 1) ÷ S + 1
    lane = (gid - 1) % S
    s = zero(T)
    if row <= m
        @inbounds for k in (A_rowptr[row] + Ti(lane)):Ti(S):(A_rowptr[row + Ti(1)] - Ti(1))
            s += A_nzval[k] * b[A_colval[k]]
        end
    end
    @inbounds tmp[lid] = s
    @synchronize

    for u in 1:nsteps
        lane = (gid - 1) % S
        span = S >> u
        @inbounds if lane < span
            tmp[lid] += tmp[lid + span]
        end
        @synchronize
    end

    row = (gid - 1) ÷ S + 1
    lane = (gid - 1) % S
    if lane == 0 && row <= m
        @inbounds c[row] = α * tmp[lid] + β * c[row]
    end
end

"""
    launch_spmv_csr!(c, A, b, α, β, backend, Val(S))

Launch [`spmv_csr_vector!`](@ref) with `S` work items per row.
"""
function launch_spmv_csr!(
        c, A::GPUSparseMatrixCSR, b, α::Number, β::Number, backend, ::Val{S}
    ) where {S}
    kernel! = spmv_csr_vector!(backend, CSR_WORKGROUP)
    ndrange = cld(size(A, 1) * S, CSR_WORKGROUP) * CSR_WORKGROUP
    kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β, Val(S); ndrange)
    return nothing
end

"""
    launch_spmv_csr!(c, A, b, α, β, backend)

Launch the SpMV kernel best suited to `A`'s rows, per [`subgroup_size`](@ref).

The sub-group size has to reach the kernel as a `Val` so that the reduction unrolls and the
local memory can be sized, hence the chain of branches over the handful of sizes
[`subgroup_size`](@ref) can return.
"""
function launch_spmv_csr!(c, A::GPUSparseMatrixCSR, b, α::Number, β::Number, backend)
    S = subgroup_size(A)
    if S >= 32
        launch_spmv_csr!(c, A, b, α, β, backend, Val(32))
    elseif S >= 16
        launch_spmv_csr!(c, A, b, α, β, backend, Val(16))
    elseif S >= 8
        launch_spmv_csr!(c, A, b, α, β, backend, Val(8))
    elseif S >= 4
        launch_spmv_csr!(c, A, b, α, β, backend, Val(4))
    elseif S >= 2
        launch_spmv_csr!(c, A, b, α, β, backend, Val(2))
    else
        kernel! = spmv_csr!(backend)
        kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β; ndrange = size(A, 1))
    end
    return nothing
end

function LinearAlgebra.mul!(
        c::V,
        A::GPUSparseMatrixCSR{T, Ti, V},
        b::V,
        α::Number,
        β::Number
    ) where {T <: Number, Ti, V <: DenseVector{T}}
    check_mul_dims(c, A, b)
    backend = common_backend(c, A, b)
    α_is_one = isone(α)
    β_is_zero = iszero(β)
    if α_is_one && β_is_zero
        launch_spmv_csr!(c, A, b, One(), Zero(), backend)
    elseif α_is_one
        launch_spmv_csr!(c, A, b, One(), β, backend)
    elseif β_is_zero
        launch_spmv_csr!(c, A, b, α, Zero(), backend)
    else
        launch_spmv_csr!(c, A, b, α, β, backend)
    end
    return c
end

"""
    spmm_csr!(c, A_rowptr, A_colval, A_nzval, b, α, β)

Batched counterpart of [`spmv_csr!`](@ref): one work item per (row, batch column) pair.
"""
@kernel function spmm_csr!(
        c::DenseMatrix{T},
        A_rowptr::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::DenseVector{T},
        b::DenseMatrix{T},
        α::Number,
        β::Number
    ) where {T, Ti}
    i, q = @index(Global, NTuple)
    s = zero(T)
    @inbounds for k in A_rowptr[i]:(A_rowptr[i + Ti(1)] - Ti(1))
        s += A_nzval[k] * b[A_colval[k], q]
    end
    @inbounds c[i, q] = α * s + β * c[i, q]
end

"""
    spmm_csr_vector!(c, A_rowptr, A_colval, A_nzval, b, α, β, Val(S))

Batched counterpart of [`spmv_csr_vector!`](@ref): `S` work items cooperate on one row of
one batch column. Launched over `(S * size(A, 1), size(c, 2))` with the cooperating lanes
contiguous in the first dimension, so that the workgroup is one-dimensional and its local
memory can be indexed by the linear local index.
"""
@kernel function spmm_csr_vector!(
        c::DenseMatrix{T},
        A_rowptr::DenseVector{Ti},
        A_colval::DenseVector{Ti},
        A_nzval::DenseVector{T},
        b::DenseMatrix{T},
        α::Number,
        β::Number,
        ::Val{S}
    ) where {T, Ti, S}
    gid, q = @index(Global, NTuple)
    lid = @index(Local, Linear)
    @uniform G = prod(@groupsize())
    @uniform nsteps = trailing_zeros(S)
    @uniform m = size(c, 1)
    tmp = @localmem T (G,)

    row = (gid - 1) ÷ S + 1
    lane = (gid - 1) % S
    s = zero(T)
    if row <= m
        @inbounds for k in (A_rowptr[row] + Ti(lane)):Ti(S):(A_rowptr[row + Ti(1)] - Ti(1))
            s += A_nzval[k] * b[A_colval[k], q]
        end
    end
    @inbounds tmp[lid] = s
    @synchronize

    for u in 1:nsteps
        lane = (gid - 1) % S
        span = S >> u
        @inbounds if lane < span
            tmp[lid] += tmp[lid + span]
        end
        @synchronize
    end

    row = (gid - 1) ÷ S + 1
    lane = (gid - 1) % S
    if lane == 0 && row <= m
        @inbounds c[row, q] = α * tmp[lid] + β * c[row, q]
    end
end

"""
    launch_spmm_csr!(c, A, b, α, β, backend, Val(S))

Launch [`spmm_csr_vector!`](@ref) with `S` work items per row.
"""
function launch_spmm_csr!(
        c, A::GPUSparseMatrixCSR, b, α::Number, β::Number, backend, ::Val{S}
    ) where {S}
    kernel! = spmm_csr_vector!(backend, (CSR_WORKGROUP, 1))
    rows = cld(size(A, 1) * S, CSR_WORKGROUP) * CSR_WORKGROUP
    kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β, Val(S); ndrange = (rows, size(c, 2)))
    return nothing
end

"""
    launch_spmm_csr!(c, A, b, α, β, backend)

Launch the SpMM kernel best suited to `A`'s rows, per [`subgroup_size`](@ref).

Sub-groups of two are not worth their barriers here, so anything below four work items per
row falls back to [`spmm_csr!`](@ref). Matrices that sparse would gain instead from having
one work item own a row and several batch columns, so that its row of `A` is fetched once
and reused, which is worth a further ~1.6x on them but is a loss on the skewed matrices
this kernel is for; it is not implemented.
"""
function launch_spmm_csr!(c, A::GPUSparseMatrixCSR, b, α::Number, β::Number, backend)
    S = subgroup_size(A)
    if S >= 32
        launch_spmm_csr!(c, A, b, α, β, backend, Val(32))
    elseif S >= 16
        launch_spmm_csr!(c, A, b, α, β, backend, Val(16))
    elseif S >= 8
        launch_spmm_csr!(c, A, b, α, β, backend, Val(8))
    elseif S >= 4
        launch_spmm_csr!(c, A, b, α, β, backend, Val(4))
    else
        kernel! = spmm_csr!(backend)
        kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β; ndrange = size(c))
    end
    return nothing
end

function LinearAlgebra.mul!(
        c::DenseMatrix{T},
        A::GPUSparseMatrixCSR{T},
        b::DenseMatrix{T},
        α::Number,
        β::Number
    ) where {T <: Number}
    check_mul_dims(c, A, b)
    backend = common_backend(c, A, b)
    α_is_one = isone(α)
    β_is_zero = iszero(β)
    if α_is_one && β_is_zero
        launch_spmm_csr!(c, A, b, One(), Zero(), backend)
    elseif α_is_one
        launch_spmm_csr!(c, A, b, One(), β, backend)
    elseif β_is_zero
        launch_spmm_csr!(c, A, b, α, Zero(), backend)
    else
        launch_spmm_csr!(c, A, b, α, β, backend)
    end
    return c
end
