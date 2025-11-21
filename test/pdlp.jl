using Adapt
using CoolPDLP
using HiGHS: HiGHS
using JLArrays
using LinearAlgebra
using JuMP: JuMP, MOI
using Test

netlib = list_netlib_instances()
milp = read_netlib_instance(netlib[4])
params = PDLPParameters(; termination_reltol = 1.0e-6, max_kkt_passes = 10^7)
params_gpu = PDLPParameters(
    Float32,
    Int32,
    GPUSparseMatrixCSR,
    JLBackend();
    termination_reltol = 1.0e-3,
    max_kkt_passes = 10^7,
)

@testset "Comparison with JuMP" begin
    sol, state = pdlp(milp, params; show_progress = false)
    @test state.termination_reason == CoolPDLP.CONVERGENCE

    jump_model = JuMP.read_from_file(milp.path; format = MOI.FileFormats.FORMAT_MPS)
    JuMP.set_optimizer(jump_model, HiGHS.Optimizer)
    JuMP.set_silent(jump_model)
    JuMP.optimize!(jump_model)
    jump_x = JuMP.value.(JuMP.all_variables(jump_model))

    @test is_feasible(sol.x, milp; cons_tol = 1.0e-3)
    @test is_feasible(jump_x, milp)
    @test objective_value(jump_x, milp) ≈ objective_value(sol.x, milp) rtol = 1.0e-4
end

@testset "Running on GPU arrays" begin
    sol, state = pdlp(milp, params_gpu; show_progress = false)
    @test state.termination_reason == CoolPDLP.CONVERGENCE
end
