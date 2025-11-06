using CoolPDLP
using LinearAlgebra
using SparseArrays
using Test

@testset "Symmetrized" begin
    A = randn(10, 20)
    S = CoolPDLP.Symmetrized(A, Matrix(transpose(A)))
    x = randn(20)
    y = zeros(20)
    mul!(y, S, x)
    @test y ≈ transpose(A) * A * x
end

@testset "Spectral norm" begin
    A = randn(10, 20)
    s1 = CoolPDLP.spectral_norm(A, Matrix(transpose(A)); tol = 1.0e-7)
    s1_ref = opnorm(A, 2)
    @test s1 ≈ s1_ref rtol = 1.0e-1
end

@testset "Sort columns" begin
    for m in (10, 20, 30), n in (10, 20, 30), p in (0.01, 0.1, 0.3)
        A = sprand(n, n, p)
        A_sorted, perm = sort_columns(A)
        @test A[:, perm] == A_sorted
    end
end
