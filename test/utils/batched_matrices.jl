using Adapt
using CoolPDLP: GPUSparseMatrixCSR, BatchedGPUSparseMatrixCSR, spectral_norm
using GPUArraysCore
using JLArrays
using KernelAbstractions
using LinearAlgebra
using Random: Xoshiro
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
    @test nonzeros(A_batched) === A_batched.nzval
    @test nonzeros(A_csr) === A_csr.nzval

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

@testset "Batched CSR slices on the CPU" begin
    batches = 4
    # how close the power method below lands depends on the matrix, so fix the draw
    rng = Xoshiro(0)
    pattern = sprand(rng, 8, 6, 0.35)
    As = map(1:batches) do _
        SparseMatrixCSC(pattern.m, pattern.n, copy(pattern.colptr), copy(pattern.rowval), rand(rng, nnz(pattern)))
    end
    function stack_csr(Ms)
        csrs = map(GPUSparseMatrixCSR, Ms)
        ref = first(csrs)
        return BatchedGPUSparseMatrixCSR(
            ref.m, ref.n, ref.rowptr, ref.colval, reduce(hcat, map(A -> A.nzval, csrs))
        )
    end
    A_batched = stack_csr(As)
    At_batched = stack_csr(map(A -> SparseMatrixCSC(transpose(A)), As))
    rhs = rand(rng, size(pattern, 2))

    # unlike a GPU slice, a CPU one is a `SubArray` rather than a `DenseVector`
    for k in 1:batches
        slice = view(A_batched, :, :, k)
        @test slice isa GPUSparseMatrixCSR
        @test SparseMatrixCSC(slice) ≈ As[k]
        @test mul!(zeros(size(pattern, 1)), slice, rhs) ≈ As[k] * rhs
        @test mul!(zeros(size(pattern, 1), 2), slice, repeat(rhs, 1, 2)) ≈ As[k] * repeat(rhs, 1, 2)
    end

    @test spectral_norm(A_batched, At_batched) ≈ map(A -> opnorm(Matrix(A)), As) rtol = 1.0e-2
end
