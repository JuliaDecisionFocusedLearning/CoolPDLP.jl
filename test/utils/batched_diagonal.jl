using Adapt
using CoolPDLP: BatchedDiagonal, instance, nbinstances
using JLArrays
using KernelAbstractions
using LinearAlgebra
using Test

@testset "Batched diagonal" begin
    nbatch = 3
    d = rand(4, nbatch)
    D = BatchedDiagonal(d)
    x = rand(4, nbatch)

    @test size(D) == (4, 4, nbatch)
    @test size(D, 1) == size(D, 2) == 4
    @test size(D, 3) == nbatch
    @test eltype(D) == Float64
    @test diag(D) === d
    @test nbinstances(D) == nbatch

    @testset "instance $i" for i in 1:nbatch
        Di = instance(D, i)
        @test Di isa Diagonal
        @test Di ≈ Diagonal(d[:, i])
        @test (D * x)[:, i] ≈ Di * x[:, i]
        @test (D \ x)[:, i] ≈ Di \ x[:, i]
        @test instance(inv(D), i) ≈ inv(Di)
        @test instance(D * D, i) ≈ Di * Di
        @test instance(D * Diagonal(1:4), i) ≈ Di * Diagonal(1:4)
        @test instance(Diagonal(1:4) * D, i) ≈ Diagonal(1:4) * Di
    end

    # a shared vector is scaled into one column per instance
    @test D * rand(4) isa Matrix
    @test D ≈ BatchedDiagonal(copy(d))
    @test !isapprox(D, BatchedDiagonal(d .+ 1))

    D_jl = adapt(JLBackend(), D)
    @test diag(D_jl) isa JLArray
    @test get_backend(D_jl) == get_backend(diag(D_jl))
end
