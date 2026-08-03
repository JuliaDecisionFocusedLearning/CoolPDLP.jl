using Adapt
using CoolPDLP: GPUSparseMatrixCSR, BatchedGPUSparseMatrixCSR, mynnz, sametype_transpose,
    spectral_norm
using GPUArraysCore
using JLArrays
using LinearAlgebra
using Random: Xoshiro
using SparseArrays
using Test

include("../fixtures.jl")

@testset "spmm!" begin
    A = sprand(8, 6, 0.35)
    A_jl = adapt(JLBackend(), GPUSparseMatrixCSR(A))
    rhs, lhs = rand(size(A, 2), 3), rand(size(A, 1), 3)
    α, β = rand(), rand()

    @test mul!(jl(copy(lhs)), A_jl, jl(rhs), α, β) ≈ α * A * rhs + β * lhs
    # a zero β is a strong zero, so a destination full of NaNs is never read
    nans() = jl(fill(NaN, size(lhs)))
    @test mul!(nans(), A_jl, jl(rhs), α, 0.0) ≈ α * A * rhs
    @test mul!(nans(), A_jl, jl(rhs), 1.0, 0.0) ≈ A * rhs
    @test mul!(jl(copy(lhs)), A_jl, jl(rhs), 1.0, β) ≈ A * rhs + β * lhs
end

@testset "Batched CSR" begin
    batches = 4
    rng = Xoshiro(0)
    pattern = sprand(rng, 8, 6, 0.35)
    As = [same_pattern(pattern, rng) for _ in 1:batches]
    A_csr = GPUSparseMatrixCSR(As[1])
    rhs, lhs = rand(rng, size(pattern, 2)), rand(rng, size(pattern, 1), batches)

    A_batched = adapt(JLBackend(), BatchedGPUSparseMatrixCSR(As))

    @test size(A_batched) == (size(pattern)..., batches)
    @test nnz(A_batched) == nnz(pattern) * batches
    @test mynnz(A_batched) == nnz(pattern)
    @test nonzeros(A_batched) === A_batched.nzval
    @test nonzeros(A_csr) === A_csr.nzval

    for k in 1:batches
        slice = view(A_batched, :, :, k)
        @test @allowscalar Matrix(slice) ≈ As[k]
        @test @allowscalar slice[1, 1] == As[k][1, 1]
        @test @allowscalar A_batched[1, 1, k] == As[k][1, 1]
    end

    @test mul!(jl(copy(lhs)), A_batched, jl(rhs), 1.0, 0.0) ≈ stack(A -> A * rhs, As)
end

@testset "Batched CSR slices on the CPU" begin
    batches = 4
    # how close the power method below lands depends on the matrix, so fix the draw
    rng = Xoshiro(0)
    pattern = sprand(rng, 8, 6, 0.35)
    As = [same_pattern(pattern, rng) for _ in 1:batches]
    A_batched = BatchedGPUSparseMatrixCSR(As)
    At_batched = sametype_transpose(A_batched)
    rhs = rand(rng, size(pattern, 2))

    @test size(At_batched) == (size(pattern, 2), size(pattern, 1), batches)

    # unlike a GPU slice, a CPU one is a `SubArray` rather than a `DenseVector`
    for k in 1:batches
        slice = view(A_batched, :, :, k)
        @test slice isa GPUSparseMatrixCSR
        @test SparseMatrixCSC(slice) ≈ As[k]
        @test SparseMatrixCSC(view(At_batched, :, :, k)) ≈ SparseMatrixCSC(transpose(As[k]))
        @test mul!(zeros(size(pattern, 1)), slice, rhs) ≈ As[k] * rhs
        @test mul!(zeros(size(pattern, 1), 2), slice, repeat(rhs, 1, 2)) ≈ As[k] * repeat(rhs, 1, 2)
        # the columns of a batched solution are views, not dense vectors
        dest = zeros(size(pattern, 1), 2)
        @test mul!(view(dest, :, 1), slice, view(repeat(rhs, 1, 2), :, 2)) ≈ As[k] * rhs
    end

    @test spectral_norm(A_batched, At_batched) ≈ map(A -> opnorm(Matrix(A)), As) rtol = 1.0e-2

    # the transpose stays on the backend of the matrix it comes from
    At_jl = sametype_transpose(adapt(JLBackend(), A_batched))
    @test nonzeros(At_jl) isa JLArray
    @test Array(nonzeros(At_jl)) ≈ nonzeros(At_batched)
end
