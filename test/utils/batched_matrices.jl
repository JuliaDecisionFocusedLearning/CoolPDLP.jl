using Adapt
using CoolPDLP: GPUSparseMatrixCSR
using JLArrays
using LinearAlgebra
using SparseArrays
using Test

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
