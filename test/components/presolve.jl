using CoolPDLP
using CoolPDLP:
    milp_to_mps, mps_to_milp, presolve_milp, postsolve_solution, postsolve_or_passthrough,
    PresolveParameters, PrimalDualSolution
using KernelAbstractions: CPU
using MathOptBenchmarkInstances
using MathOptInterface: MathOptInterface as MOI
using SparseArrays
using Test

@testset "PresolveParameters" begin
    p = PresolveParameters()
    @test !p.enabled
    @test !p.verbose
    p2 = PresolveParameters(; enabled = true, verbose = true)
    @test p2.enabled
    @test p2.verbose
    @test occursin("enabled=true", string(p2))
    @test occursin("verbose=true", string(p2))
end

@testset "Algorithm propagation" begin
    algo_default = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU())
    @test !algo_default.presolve.enabled
    @test !algo_default.presolve.verbose

    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(),
        presolve_enabled = true, presolve_verbose = true,
    )
    @test algo.presolve.enabled
    @test algo.presolve.verbose
    @test occursin("PresolveParameters", string(algo))
end

@testset "MPS round trip: mixed bounds and integrality" begin
    # every row keeps strictly finite, unequal bounds so it stays an `Interval` constraint in
    # JuMP/MOI: rows of a single (F, S) constraint type round trip in creation order, which lets
    # us compare `A`, `lc`, `uc` element-wise instead of merely checking feasibility
    c = [1.0, 2.0, -1.0, 0.0]
    lv = [0.0, -Inf, -3.0, -Inf]
    uv = [4.0, Inf, 3.0, Inf]
    A = sparse(
        [
            1.0 1.0 0.0 0.0
            0.0 1.0 1.0 0.0
            1.0 0.0 0.0 1.0
        ]
    )
    lc = [-1.0, -2.0, -3.0]
    uc = [1.0, 5.0, 4.0]
    int_var = [false, false, true, false]
    var_names = ["alpha", "beta", "gamma", "delta"]
    milp = MILP(; c, lv, uv, A, lc, uc, int_var, var_names)

    path = tempname() * ".mps"
    milp_to_mps(milp, path)
    milp2 = mps_to_milp(path)
    rm(path; force = true)

    @test nbvar(milp2) == nbvar(milp)
    @test nbcons(milp2) == nbcons(milp)
    @test milp2.var_names == milp.var_names
    @test milp2.c == milp.c
    @test milp2.lv == milp.lv
    @test milp2.uv == milp.uv
    @test milp2.int_var == milp.int_var
    @test Matrix(milp2.A) == Matrix(milp.A)
    @test milp2.lc == milp.lc
    @test milp2.uc == milp.uc
end

@testset "MPS round trip: fixed variable, free variable, free row" begin
    c = [1.0, -1.0, 2.0]
    lv = [2.0, -Inf, -Inf]
    uv = [2.0, Inf, Inf]
    A = sparse([1.0 1.0 0.0; 0.0 0.0 1.0])
    lc = [-Inf, 0.0]
    uc = [Inf, 0.0]
    milp = MILP(; c, lv, uv, A, lc, uc)

    path = tempname() * ".mps"
    milp_to_mps(milp, path)
    milp2 = mps_to_milp(path)
    rm(path; force = true)

    @test milp2.lv == milp.lv  # in particular, the fixed and free variables keep their bounds
    @test milp2.uv == milp.uv
    @test nbvar(milp2) == nbvar(milp)
    @test nbcons(milp2) == nbcons(milp)
end

@testset "MPS round trip: batched or GPU MILPs are converted to CPU sparse matrices first" begin
    c = [1.0, 1.0]
    lv = [0.0, 0.0]
    uv = [1.0, 1.0]
    A = sparse([1.0 1.0])
    lc, uc = [0.5], [0.5]
    milp = MILP(; c, lv, uv, A, lc, uc)
    path = tempname() * ".mps"
    @test_nowarn milp_to_mps(milp, path)
    rm(path; force = true)
end

function _core_padded_milp()
    # a tiny 2-variable, 2-constraint "core" LP, padded with redundant structure that a
    # presolver should strip entirely: a fixed variable, a variable absent from every
    # constraint (and absent from the objective, so it stays bounded), and an empty
    # (all-zero) row
    c = [1.0, 1.0, 0.0, 0.0]
    lv = [0.0, 0.0, 3.0, -Inf]
    uv = [10.0, 10.0, 3.0, Inf]
    A = sparse(
        [
            1.0 0.0 0.0 0.0
            0.0 1.0 0.0 0.0
            0.0 0.0 0.0 0.0
        ]
    )
    lc = [4.0, 4.0, -Inf]
    uc = [4.0, 4.0, Inf]
    return MILP(; c, lv, uv, A, lc, uc)
end

@testset "presolve_milp strips redundant structure" begin
    milp = _core_padded_milp()
    params = PresolveParameters(; enabled = true)
    milp_reduced, result = presolve_milp(milp, params)

    @test !isnothing(result)
    @test nbvar(milp_reduced) < nbvar(milp)
    @test nbcons(milp_reduced) < nbcons(milp)
end

@testset "postsolve_solution recovers a feasible, optimal reduced solution" begin
    milp = _core_padded_milp()
    params = PresolveParameters(; enabled = true)
    milp_reduced, result = presolve_milp(milp, params)

    # the only genuine degree of freedom left after presolve should be a single variable fixed
    # by the RHS (x1 == x2 == 4), so any value compatible with its own bounds is optimal
    x_reduced = copy(milp_reduced.lv)
    x_reduced[.!isfinite.(x_reduced)] .= 0.0
    sol_reduced = PrimalDualSolution(x_reduced, zeros(nbcons(milp_reduced)))

    x_orig = postsolve_solution(result, sol_reduced, params)
    @test is_feasible(x_orig, milp)
    @test isapprox(objective_value(x_orig, milp), 8.0; atol = 1.0e-6)
end

@testset "postsolve_solution does not launder an infeasible reduced solution" begin
    # `_core_padded_milp` is small enough that PaPILO's presolve solves it outright, leaving no
    # variable to corrupt; use a real (non-trivial, still feasible and bounded) instance instead
    qps, path = read_instance(Netlib, "afiro")
    milp = MILP(qps; path, name = "afiro")
    params = PresolveParameters(; enabled = true)
    milp_reduced, result = presolve_milp(milp, params)
    @test nbvar(milp_reduced) > 0

    # push every surviving reduced variable far out of its bounds
    x_reduced = min.(milp_reduced.uv, 1.0e6) .+ 1000.0
    sol_reduced = PrimalDualSolution(x_reduced, zeros(nbcons(milp_reduced)))

    x_orig = postsolve_solution(result, sol_reduced, params)
    @test !is_feasible(x_orig, milp; verbose = false)
end

@testset "postsolve_or_passthrough matches the identity when presolve made no reduction" begin
    milp, sol = CoolPDLP.random_milp_and_sol(5, 8, 0.6)
    params = PresolveParameters(; enabled = true)
    x, y = postsolve_or_passthrough(nothing, sol, milp, params)
    @test x == Array(sol.x)
    @test y == Array(sol.y)
end

@testset "presolve_milp falls back gracefully when PaPILO fails" begin
    milp, _ = CoolPDLP.random_milp_and_sol(3, 4, 0.6)
    params = PresolveParameters(; enabled = true)
    bogus_dir = joinpath(tempdir(), "coolpdlp-does-not-exist-$(rand(UInt64))")
    milp_reduced, result = withenv("TMPDIR" => bogus_dir) do
        @test_logs (:warn, r"Presolve failed") match_mode = :any presolve_milp(milp, params)
    end
    @test isnothing(result)
    @test milp_reduced === milp
end

@testset "Full solve with presolve enabled matches a direct solve on a reducible problem" begin
    milp = _core_padded_milp()
    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(),
        termination_reltol = 1.0e-8, show_progress = false, presolve_enabled = true,
    )
    sol, stats = solve(milp, algo)
    @test stats.termination_status == MOI.OPTIMAL
    @test is_feasible(Array(sol.x), milp)
    @test isapprox(objective_value(Array(sol.x), milp), 8.0; atol = 1.0e-4)
    # the dual is not postsolved and comes back as zero, but must still have the right shape
    @test length(sol.y) == nbcons(milp)
    @test all(iszero, sol.y)
end

@testset "Presolve does not support batched MILPs" begin
    milp, _ = CoolPDLP.random_milp_and_sol(4, 6, 0.5)
    milp_batch = MILP(; c = repeat(milp.c, 1, 3), milp.lv, milp.uv, milp.A, milp.lc, milp.uc)
    algo = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), presolve_enabled = true, show_progress = false)
    @test_throws ArgumentError solve(milp_batch, algo)
end

@testset "Presolve is a strict speedup on a heavily padded problem" begin
    # pad the core problem with many redundant fixed variables and empty rows: a presolver
    # should strip all of that away, so far fewer KKT passes are needed to reach the same
    # tolerance than when solving the padded problem directly
    npad = 200
    c = vcat([1.0, 1.0], zeros(npad))
    lv = vcat([0.0, 0.0], fill(3.0, npad))
    uv = vcat([10.0, 10.0], fill(3.0, npad))
    A = spzeros(2 + npad, 2 + npad)
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    for k in 1:npad
        A[2 + k, 2 + k] = 1.0
    end
    lc = vcat([4.0, 4.0], fill(3.0, npad))
    uc = vcat([4.0, 4.0], fill(3.0, npad))
    milp = MILP(; c, lv, uv, A, lc, uc)

    common_opts = (; termination_reltol = 1.0e-8, max_kkt_passes = 10^6, show_progress = false)
    algo_np = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), common_opts..., presolve_enabled = false)
    algo_p = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), common_opts..., presolve_enabled = true)

    sol_np, stats_np = solve(milp, algo_np)
    sol_p, stats_p = solve(milp, algo_p)

    @test stats_np.termination_status == MOI.OPTIMAL
    @test stats_p.termination_status == MOI.OPTIMAL
    @test is_feasible(Array(sol_np.x), milp)
    @test is_feasible(Array(sol_p.x), milp)
    @test isapprox(objective_value(Array(sol_np.x), milp), 8.0; atol = 1.0e-4)
    @test isapprox(objective_value(Array(sol_p.x), milp), 8.0; atol = 1.0e-4)
    @test stats_p.kkt_passes < stats_np.kkt_passes
end
