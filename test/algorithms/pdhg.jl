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
    (x, y), stats = pdhg(milp, params; show_progress = false)
    @test stats.termination_status == MOI.OPTIMAL
    @test is_feasible(x, milp; cons_tol = 1.0e-4)
    @test is_feasible(jump_x, milp)
    @test objective_value(jump_x, milp) ≈ objective_value(x, milp) rtol = 1.0e-4
end

@testset "GPU" begin
    params_gpu = PDHGParameters(
        Float32, Int32, GPUSparseMatrixCSR, JLBackend();
        termination_reltol = 1.0e-4, max_kkt_passes = 10^7
    )
    (x_gpu, y_gpu), stats_gpu = pdhg(milp, params_gpu; show_progress = false)
    @test stats_gpu.termination_status == MOI.OPTIMAL
    @test is_feasible(Array(x_gpu), milp; cons_tol = 1.0e-4)
    @test objective_value(jump_x, milp) ≈ objective_value(Array(x_gpu), milp) rtol = 1.0e-4
end
