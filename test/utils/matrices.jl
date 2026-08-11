using Adapt
using CoolPDLP
using GPUArraysCore
using JLArrays
using KernelAbstractions
using LinearAlgebra
using SparseArrays
using Test

A_candidates = [
    sprand(m, n, p)
        for m in (10, 20, 30)
        for n in (10, 20, 30)
        for p in (0.01, 0.1, 0.2, 0.3)
];
b_candidates = [rand(size(A, 2)) for A in A_candidates];
c_candidates = [rand(size(A, 1)) for A in A_candidates];
α, β = rand(), rand()

function test_sparse_matrix(::Type{M}; A, b, c, α, β) where {M}
    A_jl = adapt(JLBackend(), M(A))
    At_jl = adapt(JLBackend(), M(sparse(transpose(A))))
    b_jl, c_jl = jl(b), jl(c)
    @test @allowscalar Matrix(A_jl) == A
    @test @allowscalar SparseMatrixCSC(A_jl) == A
    @test nnz(A_jl) == nnz(A)
    @test get_backend(A_jl) isa JLBackend
    @test mul!(copy(c_jl), A_jl, b_jl, α, β) ≈ mul!(copy(c), A, b, α, β)
    @test @allowscalar Matrix(CoolPDLP.sametype_transpose(A_jl)) == transpose(A)
    @test typeof(CoolPDLP.sametype_transpose(A_jl)) == typeof(At_jl)
    return nothing
end

@testset for M in (GPUSparseMatrixCOO, GPUSparseMatrixCSR, GPUSparseMatrixELL)
    for (A, b, c) in collect(zip(A_candidates, b_candidates, c_candidates))
        test_sparse_matrix(M; A, b, c, α, β)
        # test β is a strong zero, e.g. c should never be read since it may be uninitialized and contain NaNs
        c′ = similar(c)
        fill!(c′, NaN)
        test_sparse_matrix(M; A, b, c = c′, α, β = 0.0)
        copy!(c′, c)
        test_sparse_matrix(M; A, b, c = c′, α = 1.0, β)
        fill!(c′, NaN)
        test_sparse_matrix(M; A, b, c = c′, α = 1.0, β = 0.0)
    end
end

@testset "spmm! $M" for M in (GPUSparseMatrixCOO, GPUSparseMatrixCSR, GPUSparseMatrixELL)
    A = sprand(8, 6, 0.35)
    A_jl = adapt(JLBackend(), M(A))
    rhs, lhs = rand(size(A, 2), 3), rand(size(A, 1), 3)
    α, β = rand(), rand()

    @test mul!(jl(copy(lhs)), A_jl, jl(rhs), α, β) ≈ α * A * rhs + β * lhs
    # a zero β is a strong zero, so a destination full of NaNs is never read
    nans() = jl(fill(NaN, size(lhs)))
    @test mul!(nans(), A_jl, jl(rhs), α, 0.0) ≈ α * A * rhs
    @test mul!(nans(), A_jl, jl(rhs), 1.0, 0.0) ≈ A * rhs
    @test mul!(jl(copy(lhs)), A_jl, jl(rhs), 1.0, β) ≈ A * rhs + β * lhs
end
