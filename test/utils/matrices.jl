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
    @test mul!(copy(c_jl), A_jl, b_jl, α, β) ≈ α * A * b + β * c
    @test @allowscalar Matrix(CoolPDLP.sametype_transpose(A_jl)) == transpose(A)
    @test typeof(CoolPDLP.sametype_transpose(A_jl)) == typeof(At_jl)
    return nothing
end

@testset for M in (GPUSparseMatrixCOO, GPUSparseMatrixCSR, GPUSparseMatrixELL)
    for (A, b, c) in collect(zip(A_candidates, b_candidates, c_candidates))
        test_sparse_matrix(M; A, b, c, α, β)
    end
end
