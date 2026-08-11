using Adapt
using CoolPDLP
using JLArrays
using KernelAbstractions: CPU, get_backend
using Test

milp, sol = CoolPDLP.random_milp_and_sol(10, 20, 0.4)

@testset "Set types" begin
    milp_f32 = CoolPDLP.set_eltype(Float32, milp)
    @test milp_f32 isa MILP{Float32}
    @test milp_f32.A isa SparseMatrixCSC{Float32, Int}

    milp_i32 = CoolPDLP.set_indtype(Int32, milp)
    @test milp_i32 isa MILP{Float64}
    @test milp_i32.A isa SparseMatrixCSC{Float64, Int32}

    milp_dense = CoolPDLP.set_matrix_type(Matrix, milp)
    @test milp_dense isa MILP{Float64}
    @test milp_dense.A isa Matrix{Float64}

    sol_f32 = CoolPDLP.set_eltype(Float32, sol)
    @test sol_f32 isa PrimalDualSolution{Float32, Vector{Float32}}
end

@testset "Change backend" begin
    milp_flexible = CoolPDLP.set_matrix_type(GPUSparseMatrixCSR, milp)
    @test milp_flexible.A isa GPUSparseMatrixCSR{Float64, Int, Vector{Float64}, Vector{Int}}
    @test get_backend(milp_flexible) == CPU()

    milp_gpu = adapt(JLBackend(), milp_flexible)
    @test milp_gpu.A isa GPUSparseMatrixCSR{Float64, Int, JLVector{Float64}, JLVector{Int}}
    @test milp_gpu.c isa JLVector{Float64}
    @test milp_gpu.int_var isa JLVector{Bool}
    @test get_backend(milp_gpu) == JLBackend()

    sol_gpu = adapt(JLBackend(), sol)
    @test sol_gpu isa PrimalDualSolution{Float64, JLVector{Float64}}
end
