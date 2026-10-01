module CoolPDLPReactantcuSPARSEExt

using CoolPDLP: CoolPDLP, GPUSparseMatrixCSR
using Reactant: Reactant
using cuSPARSE: CuSparseMatrixCSC, CuSparseMatrixCSR

const CuSparseMatrix{Tv, Ti} = Union{CuSparseMatrixCSC{Tv, Ti}, CuSparseMatrixCSR{Tv, Ti}}

"""
    as_csr(A)

Wrap the device buffers of a cuSPARSE matrix in the equivalent `GPUSparseMatrixCSR`, converting a CSC matrix to CSR on the device first.

A compiled program cannot call the cuSPARSE library, and the cuSPARSE types cannot hold traced
arrays because their field types are fixed to `CuVector`. Tracing therefore goes through the
CoolPDLP format, whose products Reactant compiles: inside `@compile`, a cuSPARSE matrix is
multiplied by CoolPDLP's own CSR kernel.
"""
as_csr(A::CuSparseMatrixCSR) = GPUSparseMatrixCSR(size(A)..., A.rowPtr, A.colVal, A.nzVal)
as_csr(A::CuSparseMatrixCSC) = as_csr(CuSparseMatrixCSR(A))

function csr_type(::Type{<:CuSparseMatrix{Tv, Ti}}) where {Tv, Ti}
    V = fieldtype(CuSparseMatrixCSR{Tv, Ti}, :nzVal)
    Vi = fieldtype(CuSparseMatrixCSR{Tv, Ti}, :rowPtr)
    return GPUSparseMatrixCSR{Tv, Ti, V, Vi}
end

function Reactant.traced_type_inner(
        @nospecialize(T::Type{<:CuSparseMatrix}),
        seen,
        mode::Reactant.TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(ndevices),
        @nospecialize(runtime),
    )
    return Reactant.traced_type_inner(
        csr_type(T), seen, mode, track_numbers, ndevices, runtime
    )
end

function Reactant.make_tracer(
        seen, @nospecialize(prev::CuSparseMatrix), @nospecialize(path), mode; kwargs...
    )
    return Reactant.make_tracer(seen, as_csr(prev), path, mode; kwargs...)
end

end
