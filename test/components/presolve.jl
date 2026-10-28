using CoolPDLP
using CoolPDLP: GPUSparseMatrixCSR, KKTErrors, perform_conversion, Scratch, kkt_errors!, PrimalDualSolution, relative
using JLArrays: JLBackend
using JuMP: JuMP
using KernelAbstractions: CPU
using MathOptBenchmarkInstances
using MathOptInterface: MathOptInterface as MOI
using PaPILO: PaPILO
using SparseArrays
using Test

function kkt_error(sol::PrimalDualSolution, milp::MILP)
    err = KKTErrors(sol)
    kkt_errors!(err, Scratch(sol), sol, milp)
    return relative(err)
end

# a 2-variable LP padded with a fixed variable, a free unused variable and an empty row
function padded_milp()
    c = [1.0, 1.0, 0.0, 0.0]
    lv = [0.0, 0.0, 3.0, -Inf]
    uv = [10.0, 10.0, 3.0, Inf]
    A = sparse([1.0 0 0 0; 0 1.0 0 0; 0 0 0 0])
    return MILP(; c, lv, uv, A, lc = [4.0, 4.0, -Inf], uc = [4.0, 4.0, Inf])
end

function afiro()
    qps, path = read_instance(Netlib, "afiro")
    return MILP(qps; path, name = "afiro")
end

@testset "Presolve and postsolve" begin
    presolver = PaPILOPresolver()
    milp = padded_milp()
    milp_reduced, presolve_info = @inferred presolve(presolver, milp)
    @test nbvar(milp_reduced) < nbvar(milp)
    @test nbcons(milp_reduced) < nbcons(milp)
    sol = @inferred postsolve(presolver, presolve_info, PrimalDualSolution(milp_reduced))
    @test length(sol.x) == nbvar(milp)
    @test length(sol.y) == nbcons(milp)
    # postsolving twice gives the same result
    @test postsolve(presolver, presolve_info, PrimalDualSolution(milp_reduced)).x == sol.x
end

@testset "Dual postsolve" begin
    presolver = PaPILOPresolver()
    milp = afiro()
    milp_reduced, presolve_info = presolve(presolver, milp)
    algo = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), termination_reltol = 1.0e-9, show_progress = false)
    sol_reduced, _ = solve(milp_reduced, PrimalDualSolution(milp_reduced), algo)
    sol = postsolve(presolver, presolve_info, sol_reduced)
    @test kkt_error(sol, milp) <= 1.0e-6
end

@testset "Integer problems are presolved as their relaxation" begin
    milp = padded_milp()
    milp_int = MILP(; milp.c, milp.lv, milp.uv, milp.A, milp.lc, milp.uc, int_var = [true, false, false, false])
    milp_reduced, _ = presolve(PaPILOPresolver(), milp_int)
    @test nbvar_int(milp_reduced) == 0
end

@testset "Infeasible problem" begin
    milp = MILP(;
        c = [1.0, 1.0], lv = zeros(2), uv = fill(10.0, 2),
        A = sparse([1.0 1.0; 1.0 1.0]), lc = [2.0, -Inf], uc = [Inf, 1.0],
    )
    @test_throws "infeasible or unbounded" presolve(PaPILOPresolver(), milp)
end

@testset "Solve with presolve" begin
    milp = padded_milp()
    @testset for (T, Ti, M, backend) in [
            (Float64, Int, SparseMatrixCSC, CPU()),
            (Float32, Int32, GPUSparseMatrixCSR, JLBackend()),
        ]
        algo = PDLP(T, Ti, M; backend, termination_reltol = T(1.0e-5), show_progress = false, presolver = PaPILOPresolver())
        sol, stats = @inferred solve(milp, algo)
        @test stats.termination_status == MOI.OPTIMAL
        @test eltype(sol.x) === T
        @test objective_value(Float64.(Array(sol.x)), milp) ≈ 8.0 atol = 1.0e-3
        @test relative(stats.err) ≈ kkt_error(sol, perform_conversion(milp, algo.conversion))
    end
end

@testset "Postsolved solution missing the tolerance is almost optimal" begin
    milp = afiro()
    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(), termination_reltol = 1.0e-5,
        max_kkt_passes = 10^7, show_progress = false, presolver = PaPILOPresolver(),
    )
    sol, stats = solve(milp, algo)
    @test stats.termination_status == MOI.ALMOST_OPTIMAL
    @test relative(stats.err) ≈ kkt_error(sol, milp)
end

@testset "Solve from a starting point never presolves" begin
    milp = padded_milp()
    algo = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), show_progress = false, presolver = PaPILOPresolver())
    _, stats = solve(milp, PrimalDualSolution(milp), algo)
    @test stats.termination_status == MOI.OPTIMAL
end

@testset "Batched problems are rejected" begin
    milp = padded_milp()
    milp_batch = MILP(; c = repeat(milp.c, 1, 3), milp.lv, milp.uv, milp.A, milp.lc, milp.uc)
    algo = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), presolver = PaPILOPresolver())
    @test_throws ArgumentError solve(milp_batch, algo)
end

@testset "JuMP" begin
    model = JuMP.Model(CoolPDLP.Optimizer)
    JuMP.set_silent(model)
    JuMP.set_attribute(model, "presolver", PaPILOPresolver())
    JuMP.@variable(model, 0 <= x[1:2] <= 10)
    JuMP.@constraint(model, x[1] + x[2] >= 4)
    JuMP.@objective(model, Min, x[1] + 2 * x[2])
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.objective_value(model) ≈ 4.0 atol = 1.0e-3
end
