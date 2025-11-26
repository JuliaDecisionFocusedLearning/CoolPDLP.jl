using Adapt
using CoolPDLP
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

@testset "COO" begin
    for (A, b, c) in zip(A_candidates, b_candidates, c_candidates)
        A_coo = GPUSparseMatrixCOO(A)
        A_coo_jl = adapt(JLBackend(), A_coo)
        b_jl, c_jl = jl(b), jl(c)
        @test Matrix(A_coo) == A
        @test get_backend(A_coo_jl) isa JLBackend
        @test A_coo_jl isa GPUSparseMatrixCOO{
            Float64, Int, JLVector{Float64}, JLVector{Int},
        }
        @test mul!(copy(c_jl), A_coo_jl, b_jl, α, β) ≈ α * A * b + β * c
    end
end

@testset "CSR" begin
    for (A, b, c) in zip(A_candidates, b_candidates, c_candidates)
        A_csr = GPUSparseMatrixCSR(A)
        A_csr_jl = adapt(JLBackend(), A_csr)
        b_jl, c_jl = jl(b), jl(c)
        @test Matrix(A_csr) == A
        @test get_backend(A_csr_jl) isa JLBackend
        @test A_csr_jl isa GPUSparseMatrixCSR{
            Float64, Int, JLVector{Float64}, JLVector{Int},
        }
        @test mul!(copy(c_jl), A_csr_jl, b_jl, α, β) ≈ α * A * b + β * c
    end
end

@testset "ELL" begin
    for (A, b, c) in zip(A_candidates, b_candidates, c_candidates)
        A_ell = GPUSparseMatrixELL(A)
        A_ell_jl = adapt(JLBackend(), A_ell)
        b_jl, c_jl = jl(b), jl(c)
        @test Matrix(A_ell) == A
        @test get_backend(A_ell_jl) isa JLBackend
        @test A_ell_jl isa GPUSparseMatrixELL{
            Float64, Int, JLMatrix{Float64}, JLMatrix{Int},
        }
        mul!(copy(c_jl), A_ell_jl, b_jl, α, β)
        # @test mul!(copy(c_jl), A_ell_jl, b_jl, α, β) ≈ α * A * b + β * c
    end
end
