using CoolPDLP
using JLArrays
using KernelAbstractions: get_backend
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
