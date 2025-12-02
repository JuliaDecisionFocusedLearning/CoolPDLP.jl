using Adapt
using CoolPDLP
using HiGHS: HiGHS
using JLArrays
using LinearAlgebra
using MathProgBenchmarks
using JuMP: JuMP, MOI
using Test

netlib = list_instances(Netlib)
qps, path = read_instance(Netlib, netlib[4]);
milp = MILP(qps)

begin
    jump_model = JuMP.read_from_file(path; format = MOI.FileFormats.FORMAT_MPS)
    JuMP.set_optimizer(jump_model, HiGHS.Optimizer)
    JuMP.set_silent(jump_model)
    JuMP.optimize!(jump_model)
    jump_x = JuMP.value.(JuMP.all_variables(jump_model))
end

@testset "CPU" begin
    params = PDHGParameters(; termination_reltol = 1.0e-6, max_kkt_passes = 10^7)
    sol, stats = solve(milp, params; show_progress = false)
    @test stats.termination_status == MOI.OPTIMAL
    @test is_feasible(sol.x, milp; cons_tol = 1.0e-4)
    @test is_feasible(jump_x, milp)
    @test objective_value(jump_x, milp) ≈ objective_value(sol.x, milp) rtol = 1.0e-4
end

@testset "GPU" begin
    params_gpu = PDHGParameters(
        Float32, Int32, GPUSparseMatrixCSR, JLBackend();
        termination_reltol = 1.0e-4, max_kkt_passes = 10^7
    )
    sol_gpu, stats_gpu = solve(milp, params_gpu; show_progress = false)
    @test stats_gpu.termination_status == MOI.OPTIMAL
    @test is_feasible(Array(sol_gpu.x), milp; cons_tol = 1.0e-3)
    @test objective_value(jump_x, milp) ≈ objective_value(Array(sol_gpu.x), milp) rtol = 1.0e-3
end
