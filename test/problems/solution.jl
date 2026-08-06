using CoolPDLP
using CoolPDLP: instance, isbatched, nbinstances
using HiGHS: HiGHS
using JLArrays
using LinearAlgebra
using JuMP: JuMP, MOI
using SparseArrays
using Test

"""
    batched_variants(nbatch; c, lv, uv, A, lc, uc)

Return the MILP keyword sets which batch one field group each, as swept by `test/algorithms/batching.jl`.
"""
function batched_variants(nbatch; c, lv, uv, A, lc, uc)
    batch(v) = repeat(v, 1, nbatch)
    return (
        "objective" => (; c = batch(c), lv, uv, A, lc, uc),
        "variable bounds" => (; c, lv = batch(lv), uv = batch(uv), A, lc, uc),
        "constraint bounds" => (; c, lv, uv, A, lc = batch(lc), uc = batch(uc)),
    )
end

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

@testset "Zero solution of a batch" begin
    nbatch = 3
    A = sprandn(4, 6, 0.5)
    int_var = zeros(Bool, 6)
    single = (c = randn(6), lv = -ones(6), uv = ones(6), lc = -ones(4), uc = ones(4))

    # the batch dimension must be picked up from whichever fields carry it
    @testset "$name" for (name, kw) in batched_variants(nbatch; single..., A)
        milp = MILP(; kw..., int_var)
        @test isbatched(milp)
        @test nbinstances(milp) == nbatch
        # the number of instances is only known at run time, but its shape must be inferrable
        sol = @inferred PrimalDualSolution(milp)
        @test sol isa PrimalDualSolution{Float64, Matrix{Float64}}
        @test nbinstances(sol) == nbatch
        @test size(sol.x) == (6, nbatch)
        @test size(sol.y) == (4, nbatch)
        @test iszero(sol.x) && iszero(sol.y)
    end

    @testset "unbatched" begin
        milp = MILP(; single..., A, int_var)
        @test !isbatched(milp)
        @test nbinstances(milp) == 1
        sol = @inferred PrimalDualSolution(milp)
        @test sol isa PrimalDualSolution{Float64, Vector{Float64}}
        @test size(sol.x) == (6,)
        @test size(sol.y) == (4,)
    end

    @testset "solve keeps the batch" begin
        milp = MILP(; single..., c = repeat(single.c, 1, nbatch), A, int_var)
        @test size(solve(milp, PDLP(; max_kkt_passes = 100))[1].x) == (6, nbatch)
    end
end

@testset "Feasibility of a batch" begin
    nbatch = 3
    c, lv, uv = [1.0, 2.0], zeros(2), 2 .* ones(2)
    A = sparse([1.0 1.0])
    lc, uc = [1.0], [1.0]
    int_var = [true, false]

    # one feasible column, then a bound violation, then a constraint violation
    x = [1.0 2.0 0.0; 0.0 -1.0 0.0]

    @testset "$name" for (name, kw) in batched_variants(nbatch; c, lv, uv, A, lc, uc)
        milp = MILP(; kw..., int_var)
        feas = is_feasible(x, milp; verbose = false)
        @test feas isa Vector{Bool}
        @test feas == [true, false, false]
        for i in 1:nbatch
            @test feas[i] == is_feasible(x[:, i], instance(milp, i); verbose = false)
        end
    end

    @testset "warnings" begin
        milp = MILP(; c = repeat(c, 1, nbatch), lv, uv, A, lc, uc, int_var)
        @test @test_warn "Variable bounds not satisfied" is_feasible(x, milp) ==
            [true, false, false]
    end
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
