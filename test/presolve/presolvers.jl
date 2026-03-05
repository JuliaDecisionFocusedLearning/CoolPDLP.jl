using CoolPDLP
using PaPILO  # activates PaPILOExt
using PSLP    # activates PSLPExt
using HiGHS: HiGHS
using JuMP: JuMP, MOI
using MathOptBenchmarkInstances
using SparseArrays
using Test

const _fixed_var_milp = let
    A = sparse([0.0 1.0 1.0])
    MILP(; c = [1.0, 1.0, 1.0], lv = [1.0, 0.0, 0.0], uv = [1.0, Inf, Inf],
           A, lc = [2.0], uc = [Inf])
end

for presolver in [PaPILOPresolver, PSLPPresolver]
    @testset "$presolver reduces fixed variable" begin
        algo = PDLP(; termination_reltol = 1e-4, show_progress = false,
                      presolver = presolver())
        sol, stats = solve(_fixed_var_milp, algo)

        @test stats.termination_status == CoolPDLP.OPTIMAL
        @test is_feasible(sol.x, _fixed_var_milp; cons_tol = 1e-3)
        @test isapprox(sol.x[1], 1.0; atol = 1e-3)
        @test isapprox(objective_value(sol.x, _fixed_var_milp), 3.0; atol = 1e-2)
    end

    @testset "$presolver on Netlib/afiro" begin
        qps, path = read_instance(Netlib, "afiro")
        milp      = MILP(qps; dataset = Netlib, path)

        jump_model = JuMP.read_from_file(path; format = MOI.FileFormats.FORMAT_MPS)
        JuMP.set_optimizer(jump_model, HiGHS.Optimizer)
        JuMP.set_silent(jump_model)
        JuMP.optimize!(jump_model)
        ref_obj = JuMP.objective_value(jump_model)

        algo = PDLP(; termination_reltol = 1e-4, show_progress = false,
                      presolver = presolver())
        sol, stats = solve(milp, algo)

        @test stats.termination_status == CoolPDLP.OPTIMAL
        @test isapprox(objective_value(sol.x, milp), ref_obj; rtol = 1e-2)
    end
end
