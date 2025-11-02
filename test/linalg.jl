using CoolPDLP
using LinearAlgebra
using Test

@testset "Spectral norm" begin
    A = randn(10, 20)
    s1 = CoolPDLP.spectral_norm(A, Matrix(transpose(A)); tol = 1.0e-7)
    s1_ref = opnorm(A, 2)
    @test s1 ≈ s1_ref rtol = 1.0e-2
end
