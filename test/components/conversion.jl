using CoolPDLP
using JLArrays
using Adapt: adapt
using KernelAbstractions: get_backend
using LinearAlgebra: mul!
using MathOptBenchmarkInstances: Netlib, read_instance
using SparseArrays
using Test

milp, sol = CoolPDLP.random_milp_and_sol(10, 20, 0.4)
params = CoolPDLP.ConversionParameters(Float32, Int32, GPUSparseMatrixCSR; backend = JLBackend())

milp_gpu = CoolPDLP.perform_conversion(milp, params)
@test milp_gpu isa MILP{Float32}
@test milp_gpu.A isa GPUSparseMatrixCSR{Float32, Int32, JLVector{Float32}, JLVector{Int32}}
@test get_backend(milp_gpu) == JLBackend()

sol_gpu = CoolPDLP.perform_conversion(sol, params)
@test sol_gpu isa PrimalDualSolution{Float32, JLVector{Float32}}

@testset "No conversion" begin
    params_none = CoolPDLP.ConversionParameters()
    @test params_none isa CoolPDLP.ConversionParameters{Nothing, Nothing, Nothing}
    @test isnothing(params_none.backend)
    # nothing is converted, so the arrays are passed through as they are
    @test CoolPDLP.perform_conversion(milp, params_none).A === milp.A
    @test CoolPDLP.perform_conversion(sol, params_none).x === sol.x
end

@testset "Mixed precision $M" for M in (
        SparseArrays.SparseMatrixCSC, GPUSparseMatrixCSR, GPUSparseMatrixCOO, GPUSparseMatrixELL,
    )
    backend = M === SparseArrays.SparseMatrixCSC ? CoolPDLP.CPU() : JLBackend()
    params_mixed = CoolPDLP.ConversionParameters(Float64, Int, M, Float32; backend)
    milp_mixed = CoolPDLP.perform_conversion(milp, params_mixed)
    @test milp_mixed isa MILP{Float64}
    @test eltype(milp_mixed.A) === eltype(milp_mixed.At) === Float32
    @test eltype(milp_mixed.c) === Float64
    @test occursin("matrices Float32", string(milp_mixed))
    # the products read `Float32` values but accumulate in `Float64`: they match a `Float64`
    # product by the rounded matrix, not one in `Float32`
    A32 = SparseArrays.SparseMatrixCSC{Float64}(SparseArrays.SparseMatrixCSC{Float32}(milp.A))
    x = randn(size(milp.A, 2))
    xb = randn(size(milp.A, 2), 3)
    @test Array(mul!(adapt(backend, zeros(size(milp.A, 1))), milp_mixed.A, adapt(backend, x))) ≈ A32 * x rtol = 1.0e-14
    @test Array(mul!(adapt(backend, zeros(size(milp.A, 1), 3)), milp_mixed.A, adapt(backend, xb))) ≈ A32 * xb rtol = 1.0e-14
    afiro = MILP(read_instance(Netlib, "afiro")[1])
    # the KKT errors are those of the `Float32` matrices: a tolerance below `eps(Float32)` would
    # be reported as met without being so on the original problem
    @test_throws ArgumentError PDLP(Float64, Int, M, Float32; backend, termination_reltol = 1.0e-8)
    algo = PDLP(Float64, Int, M, Float32; backend, termination_reltol = 1.0e-6)
    sol_mixed, stats = solve(afiro, algo)
    @test CoolPDLP.termination_status(stats) == CoolPDLP.MOI.OPTIMAL
    @test eltype(sol_mixed.x) === Float64
    @test objective_value(Array(sol_mixed.x), afiro) ≈ -464.75314286 rtol = 1.0e-5
end
