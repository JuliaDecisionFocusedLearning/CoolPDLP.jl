using CoolPDLP
using HiGHS: HiGHS
using JLArrays
using LinearAlgebra
using JuMP: JuMP, MOI
using Test

netlib = list_netlib_instances()
milp, path = read_netlib_instance(netlib[4])
params = PDHGParameters(; tol_termination = 1.0e-6, max_kkt_passes = 10^7)

@testset "Comparison with JuMP" begin
    state = pdhg(milp, params; show_progress = false)
    @test state.termination_reason == CoolPDLP.CONVERGENCE
    sol = state.x

    jump_model = JuMP.read_from_file(path; format = MOI.FileFormats.FORMAT_MPS)
    JuMP.set_optimizer(jump_model, HiGHS.Optimizer)
    JuMP.set_silent(jump_model)
    JuMP.optimize!(jump_model)
    jump_sol = JuMP.value.(JuMP.all_variables(jump_model))

    @test is_feasible(sol, milp; cons_tol = 1.0e-4)
    @test is_feasible(jump_sol, milp)
    @test objective_value(jump_sol, milp) ≈ objective_value(sol, milp) rtol = 1.0e-4
end

@testset "Running on GPU arrays" begin
    sad = SaddlePointProblem(milp)
    sad_jl = adapt(JLBackend(), change_matrix_type(DeviceSparseMatrixCSR, sad))
    state = pdhg(sad_jl, params; show_progress = false)
    @test state.termination_reason == CoolPDLP.CONVERGENCE
end
