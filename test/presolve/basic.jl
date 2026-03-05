using CoolPDLP
using SparseArrays
using Test

function _milp(; c, lv, uv, A, lc, uc)
    return MILP(; c = Float64.(c), lv = Float64.(lv), uv = Float64.(uv),
                  A = SparseMatrixCSC{Float64, Int}(A),
                  lc = Float64.(lc), uc = Float64.(uc))
end

@testset "show" begin
    io = IOBuffer()
    show(io, BasicPresolver())
    out = String(take!(io))
    @test contains(out, "fixed_variables")
    @test contains(out, "empty_rows")
    @test contains(out, "empty_columns")

    show(io, BasicPresolver(fixed_variables = false))
    out = String(take!(io))
    @test !contains(out, "fixed_variables")
    @test contains(out, "empty_rows")
end

@testset "unchanged" begin
    milp = _milp(; c = [1.0, 1.0], lv = [0.0, 0.0], uv = [Inf, Inf],
                   A = sparse([1.0 1.0]), lc = [1.0], uc = [Inf])
    status, res = CoolPDLP.apply_presolve(BasicPresolver(), milp)
    @test status == CoolPDLP.PresolveUnchanged
    @test res isa CoolPDLP.BasicPresolveResult
    @test nbvar(res.milp_to_solve) == 2
    @test nbcons(res.milp_to_solve) == 1
end

@testset "fixed variable not in constraint" begin
    milp = _milp(; c = [1.0, 1.0, 1.0],
                   lv = [3.0, 0.0, 0.0], uv = [3.0, Inf, Inf],
                   A = sparse([0.0 1.0 1.0]),
                   lc = [2.0], uc = [Inf])
    status, res = CoolPDLP.apply_presolve(BasicPresolver(), milp)
    red = res.milp_to_solve
    @test status == CoolPDLP.PresolveReduced
    @test res isa CoolPDLP.BasicPresolveResult
    @test nbvar(red) == 2
    @test nbcons(red) == 1
    @test red.lc[1] == 2.0
    @test red.uc[1] == Inf
    @test res.var_map == [2, 3]
    @test res.con_map == [1]
    @test res.fixed_var_idx == [1]
    @test res.fixed_var_val ≈ [3.0]
    @test res.n_orig == 3
    @test res.m_orig == 1
end

@testset "fixed variable in constraint" begin
    milp = _milp(; c = [1.0, 1.0],
                   lv = [2.0, 0.0], uv = [2.0, Inf],
                   A = sparse([1.0 1.0]),
                   lc = [5.0], uc = [5.0])
    status, res = CoolPDLP.apply_presolve(BasicPresolver(), milp)
    red = res.milp_to_solve
    @test status == CoolPDLP.PresolveReduced
    @test nbvar(red) == 1
    @test red.lc[1] ≈ 3.0
    @test red.uc[1] ≈ 3.0
end

@testset "empty row feasible" begin
    A = SparseMatrixCSC{Float64, Int}(sparse([1.0 1.0; 0.0 0.0]))
    milp = _milp(; c = [1.0, 1.0], lv = [0.0, 0.0], uv = [Inf, Inf],
                   A, lc = [1.0, 0.0], uc = [Inf, 0.0])
    status, res = CoolPDLP.apply_presolve(BasicPresolver(), milp)
    red = res.milp_to_solve
    @test status == CoolPDLP.PresolveReduced
    @test nbcons(red) == 1
    @test res.con_map == [1]
end

@testset "empty row infeasible lower" begin
    milp = _milp(; c = [1.0], lv = [0.0], uv = [Inf],
                   A = spzeros(Float64, 1, 1),
                   lc = [5.0], uc = [Inf])
    status, res = CoolPDLP.apply_presolve(BasicPresolver(), milp)
    @test status == CoolPDLP.PresolveInfeasible
    @test res isa CoolPDLP.BasicPresolveResult
end

@testset "empty row infeasible upper" begin
    milp = _milp(; c = [1.0], lv = [0.0], uv = [Inf],
                   A = spzeros(Float64, 1, 1),
                   lc = [-Inf], uc = [-3.0])
    status, _ = CoolPDLP.apply_presolve(BasicPresolver(), milp)
    @test status == CoolPDLP.PresolveInfeasible
end

@testset "empty column c > 0, fixed to lower" begin
    milp = _milp(; c = [1.0, 2.0], lv = [0.0, 0.0], uv = [Inf, Inf],
                   A = sparse([1.0 0.0]), lc = [1.0], uc = [Inf])
    status, res = CoolPDLP.apply_presolve(BasicPresolver(), milp)
    red = res.milp_to_solve
    @test status == CoolPDLP.PresolveReduced
    @test nbvar(red) == 1
    @test res.fixed_var_idx == [2]
    @test res.fixed_var_val ≈ [0.0]
end

@testset "empty column c < 0, fixed to upper" begin
    milp = _milp(; c = [1.0, -2.0], lv = [0.0, 0.0], uv = [Inf, 5.0],
                   A = sparse([1.0 0.0]), lc = [1.0], uc = [Inf])
    status, res = CoolPDLP.apply_presolve(BasicPresolver(), milp)
    @test status == CoolPDLP.PresolveReduced
    @test res.fixed_var_idx == [2]
    @test res.fixed_var_val ≈ [5.0]
end

@testset "empty column c = 0, fixed to clamped" begin
    milp = _milp(; c = [1.0, 0.0], lv = [0.0, 1.0], uv = [Inf, 3.0],
                   A = sparse([1.0 0.0]), lc = [1.0], uc = [Inf])
    status, res = CoolPDLP.apply_presolve(BasicPresolver(), milp)
    @test status == CoolPDLP.PresolveReduced
    @test res.fixed_var_val ≈ [1.0]
end

@testset "empty column c < 0, uv = Inf" begin
    milp = _milp(; c = [1.0, -1.0], lv = [0.0, 0.0], uv = [Inf, Inf],
                   A = sparse([1.0 0.0]), lc = [1.0], uc = [Inf])
    status, _ = CoolPDLP.apply_presolve(BasicPresolver(), milp)
    @test status == CoolPDLP.PresolveUnbounded
end

@testset "empty column c > 0, lv = -Inf" begin
    milp = _milp(; c = [1.0, 1.0], lv = [0.0, -Inf], uv = [Inf, Inf],
                   A = sparse([1.0 0.0]), lc = [1.0], uc = [Inf])
    status, _ = CoolPDLP.apply_presolve(BasicPresolver(), milp)
    @test status == CoolPDLP.PresolveUnbounded
end

@testset "disable fixed_variables" begin
    milp = _milp(; c = [1.0, 1.0], lv = [2.0, 0.0], uv = [2.0, Inf],
                   A = sparse([1.0 1.0]), lc = [1.0], uc = [Inf])
    status, _ = CoolPDLP.apply_presolve(BasicPresolver(; fixed_variables = false), milp)
    @test status == CoolPDLP.PresolveUnchanged
end

@testset "disable empty_rows" begin
    milp = _milp(; c = [1.0], lv = [0.0], uv = [Inf],
                   A = sparse([0.0; 1.0]), lc = [0.0, 1.0], uc = [0.0, Inf])
    status, _ = CoolPDLP.apply_presolve(BasicPresolver(; empty_rows = false), milp)
    @test status == CoolPDLP.PresolveUnchanged
end

@testset "disable empty_columns" begin
    milp = _milp(; c = [1.0, 2.0], lv = [0.0, 0.0], uv = [Inf, Inf],
                   A = sparse([1.0 0.0]), lc = [1.0], uc = [Inf])
    status, _ = CoolPDLP.apply_presolve(BasicPresolver(; empty_columns = false), milp)
    @test status == CoolPDLP.PresolveUnchanged
end

@testset "recover_solution fixed var placed, free vars from reduced" begin
    milp = _milp(; c = [1.0, 1.0, 1.0],
                   lv = [3.0, 0.0, 0.0], uv = [3.0, Inf, Inf],
                   A = sparse([0.0 1.0 1.0]), lc = [2.0], uc = [Inf])
    _, res = CoolPDLP.apply_presolve(BasicPresolver(), milp)

    reduced_sol = PrimalDualSolution([1.5, 0.5], [0.9])
    orig_sol    = CoolPDLP.recover_solution(res, reduced_sol)

    @test length(orig_sol.x) == 3
    @test length(orig_sol.y) == 1
    @test orig_sol.x[1] ≈ 3.0
    @test orig_sol.x[2] ≈ 1.5
    @test orig_sol.x[3] ≈ 0.5
    @test orig_sol.y[1] ≈ 0.9
end

@testset "recover_solution removed row gets dual = 0" begin
    A = SparseMatrixCSC{Float64, Int}(sparse([1.0 1.0; 0.0 0.0]))
    milp = _milp(; c = [1.0, 1.0], lv = [0.0, 0.0], uv = [Inf, Inf],
                   A, lc = [1.0, 0.0], uc = [Inf, 0.0])
    _, res = CoolPDLP.apply_presolve(BasicPresolver(), milp)

    reduced_sol = PrimalDualSolution([0.5, 0.5], [1.2])
    orig_sol    = CoolPDLP.recover_solution(res, reduced_sol)

    @test length(orig_sol.y) == 2
    @test orig_sol.y[1] ≈ 1.2
    @test orig_sol.y[2] ≈ 0.0
end

@testset "CoolPDLP.map_warmstart projects to reduced space" begin
    milp = _milp(; c = [1.0, 1.0, 1.0],
                   lv = [3.0, 0.0, 0.0], uv = [3.0, Inf, Inf],
                   A = sparse([0.0 1.0 1.0]), lc = [2.0], uc = [Inf])
    _, res = CoolPDLP.apply_presolve(BasicPresolver(), milp)

    orig_sol = PrimalDualSolution([0.1, 0.2, 0.3], [0.5])
    ws       = CoolPDLP.map_warmstart(res, orig_sol)

    @test ws.x ≈ [0.2, 0.3]
    @test ws.y ≈ [0.5]
end

@testset "fixed var + empty row + empty col" begin
    milp = _milp(; c = [1.0, 1.0],
                   lv = [1.0, 0.0], uv = [1.0, Inf],
                   A = sparse([1.0 0.0]),
                   lc = [1.0], uc = [1.0])

    status, res = CoolPDLP.apply_presolve(BasicPresolver(), milp)
    red = res.milp_to_solve

    @test status == CoolPDLP.PresolveReduced
    @test nbvar(red) == 0
    @test nbcons(red) == 0
    @test 1 ∈ res.fixed_var_idx
    @test 2 ∈ res.fixed_var_idx

    orig_sol = CoolPDLP.recover_solution(res, PrimalDualSolution(Float64[], Float64[]))
    @test orig_sol.x ≈ [1.0, 0.0]
    @test orig_sol.y ≈ [0.0]
end

@testset "fixed variable full solve" begin
    milp = MILP(; c = [1.0, 1.0, 1.0],
                  lv = [1.0, 0.0, 0.0], uv = [1.0, Inf, Inf],
                  A = sparse([0.0 1.0 1.0]),
                  lc = [2.0], uc = [Inf])
    algo = PDLP(; termination_reltol = 1e-4, time_limit = 30.0,
                  show_progress = false, presolver = BasicPresolver())
    sol, stats = solve(milp, algo)

    @test stats.termination_status == CoolPDLP.OPTIMAL
    @test length(sol.x) == 3
    @test sol.x[1] ≈ 1.0  atol = 1e-3
    @test is_feasible(sol.x, milp; cons_tol = 1e-3)
    @test objective_value(sol.x, milp) ≈ 3.0  rtol = 1e-2
end

@testset "empty column full solve" begin
    milp = MILP(; c = [1.0, 2.0], lv = [0.0, 0.0], uv = [Inf, Inf],
                  A = sparse([1.0 0.0]), lc = [1.0], uc = [Inf])
    algo = PDLP(; termination_reltol = 1e-4, time_limit = 30.0,
                  show_progress = false, presolver = BasicPresolver())
    sol, stats = solve(milp, algo)

    @test stats.termination_status == CoolPDLP.OPTIMAL
    @test sol.x[2] ≈ 0.0  atol = 1e-3
    @test is_feasible(sol.x, milp; cons_tol = 1e-3)
    @test objective_value(sol.x, milp) ≈ 1.0  rtol = 1e-2
end

@testset "infeasibility detection" begin
    milp = MILP(; c = [1.0], lv = [0.0], uv = [Inf],
                  A = spzeros(Float64, 1, 1), lc = [5.0], uc = [Inf])
    _, stats = solve(milp, PDLP(; presolver = BasicPresolver(), show_progress = false))
    @test stats.termination_status == CoolPDLP.INFEASIBLE
end

@testset "unboundedness detection" begin
    milp = MILP(; c = [0.0, -1.0], lv = [0.0, 0.0], uv = [Inf, Inf],
                  A = sparse([1.0 0.0]), lc = [1.0], uc = [Inf])
    _, stats = solve(milp, PDLP(; presolver = BasicPresolver(), show_progress = false))
    @test stats.termination_status == CoolPDLP.UNBOUNDED
end
