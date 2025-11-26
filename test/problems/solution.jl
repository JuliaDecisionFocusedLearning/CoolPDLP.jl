using CoolPDLP
using HiGHS: HiGHS
using JLArrays
using LinearAlgebra
using JuMP: JuMP, MOI
using Test

@testset "Cube MILP" begin
    c = [1.0, 2.0]
    lv = zeros(2)
    uv = 2 .* ones(2)
    A = [1.0 1.0]
    lc = [1.0]
    uc = [1.0]
    int_var = [true, false]

    milp = MILP(; c, lv, uv, A, lc, uc, int_var)
    @test is_feasible([1.0, 0.0], milp)
    @test @test_warn "Integrality not satisfied" !is_feasible([0.5, 0.5], milp)
    @test @test_warn "Constraints not satisfied" !is_feasible([0.0, 0.0], milp)
    @test @test_warn "Variable bounds not satisfied" !is_feasible([2.0, -1.0], milp)
    @test objective_value([1.0, 1.0], milp) == 3
end

@testset "Comparison with JuMP" begin
    name = "afiro"
    qps, path = read_instance(Netlib, name)
    milp = MILP(qps; path, name, dataset = "Netlib")

    jump_model = JuMP.read_from_file(milp.path; format = MOI.FileFormats.FORMAT_MPS)
    JuMP.set_optimizer(jump_model, HiGHS.Optimizer)
    JuMP.set_silent(jump_model)
    JuMP.optimize!(jump_model)
    jump_x = JuMP.value.(JuMP.all_variables(jump_model))
    jump_obj = JuMP.objective_value(jump_model)

    @test is_feasible(jump_x, milp)
    @test objective_value(jump_x, milp) ≈ jump_obj
end
