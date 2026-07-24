using Adapt
using CoolPDLP: GPUSparseMatrixCSR, BatchedGPUSparseMatrixCSR
using GPUArraysCore
using JLArrays
using KernelAbstractions
using LinearAlgebra
using SparseArrays
using Test

function dense_from_csr(m, n, rowptr, colval, nzval)
    A = zeros(eltype(nzval), m, n)
    for i in 1:m
        for k in rowptr[i]:(rowptr[i + 1] - 1)
            A[i, colval[k]] = nzval[k]
        end
    end
    return A
end

@testset "spmm!" begin
    A = sprand(8, 6, 0.35)
    A_csr = GPUSparseMatrixCSR(A)
    rhs = rand(size(A, 2), 3)
    lhs = rand(size(A, 1), 3)
    α, β = rand(), rand()

    A_jl = adapt(JLBackend(), A_csr)
    rhs_jl = jl(rhs)
    lhs_jl = jl(lhs)
    expected = α * dense_from_csr(A_csr.m, A_csr.n, A_csr.rowptr, A_csr.colval, A_csr.nzval) * rhs + β * lhs

    @test mul!(copy(lhs_jl), A_jl, rhs_jl, α, β) ≈ expected
end

@testset "Batched CSR" begin
    A = sprand(8, 6, 0.35)
    A_csr = GPUSparseMatrixCSR(A)
    batches = 4
    nzval = rand(eltype(A_csr.nzval), length(A_csr.nzval), batches)
    rhs = rand(size(A, 2))
    lhs = rand(size(A, 1), batches)

    A_batched = adapt(JLBackend(), BatchedGPUSparseMatrixCSR(A_csr.m, A_csr.n, A_csr.rowptr, A_csr.colval, nzval))
    rhs_jl = jl(rhs)
    lhs_jl = jl(lhs)

    @test size(A_batched) == (A_csr.m, A_csr.n, batches)
    @test nnz(A_batched) == length(A_csr.nzval) * batches

    for k in 1:batches
        slice = view(A_batched, :, :, k)
        expected_slice = dense_from_csr(A_csr.m, A_csr.n, A_csr.rowptr, A_csr.colval, view(nzval, :, k))
        @test @allowscalar Matrix(slice) ≈ expected_slice
        @test @allowscalar slice[1, 1] == expected_slice[1, 1]
        @test @allowscalar A_batched[1, 1, k] == expected_slice[1, 1]
    end

    expected = similar(lhs)
    for k in 1:batches
        expected[:, k] .= dense_from_csr(A_csr.m, A_csr.n, A_csr.rowptr, A_csr.colval, view(nzval, :, k)) * rhs
    end

    @test mul!(copy(lhs_jl), A_batched, rhs_jl, 1.0, 0.0) ≈ expected
end
