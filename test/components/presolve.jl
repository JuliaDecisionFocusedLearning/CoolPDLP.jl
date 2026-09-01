using CoolPDLP
using CoolPDLP:
    milp_to_mps, mps_to_milp, PresolveParameters, presolve_enabled, PrimalDualSolution,
    write_sol_file, read_sol_file, PaPILOPresolveState
using KernelAbstractions: CPU
using MathOptBenchmarkInstances
using MathOptInterface: MathOptInterface as MOI
using PaPILO: PaPILO  # loads the `CoolPDLPPaPILOExt` extension that implements presolve/postsolve
using SCIP: SCIP
using SparseArrays
using Test

@testset "presolve errors informatively when PaPILO is not loaded" begin
    # spawn a fresh process that never `using`s PaPILO, so `CoolPDLPPaPILOExt` never loads and
    # `presolve`/`postsolve` on a `PaPILOPresolver` stay at their stub definitions
    script = """
    using CoolPDLP
    milp = CoolPDLP.MILP(; c = [1.0], lv = [0.0], uv = [1.0], A = zeros(0, 1), lc = Float64[], uc = Float64[])
    try
        CoolPDLP.presolve(CoolPDLP.PaPILOPresolver(), milp)
        println("NO_ERROR")
    catch e
        println("ERROR: ", sprint(showerror, e))
    end
    """
    out = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $script`, String)
    @test occursin("Presolve requires PaPILO.jl to be loaded", out)
end

@testset "PresolveParameters" begin
    p = PresolveParameters()
    @test !presolve_enabled(p)
    @test !p.strict
    @test isnothing(p.presolver)

    p2 = PresolveParameters(; presolver = CoolPDLP.PaPILOPresolver(; verbose = true), strict = true)
    @test presolve_enabled(p2)
    @test p2.strict
    @test p2.presolver.verbose
    @test occursin("PaPILOPresolver", string(p2))
    @test occursin("strict=true", string(p2))
end

@testset "Algorithm propagation" begin
    algo_default = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU())
    @test !presolve_enabled(algo_default.presolve)
    @test isnothing(algo_default.presolve.presolver)
    @test !algo_default.presolve.strict

    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(),
        presolve = CoolPDLP.PaPILOPresolver(; verbose = true), presolve_strict = true,
    )
    @test presolve_enabled(algo.presolve)
    @test algo.presolve.presolver isa CoolPDLP.PaPILOPresolver
    @test algo.presolve.presolver.verbose
    @test algo.presolve.strict
    @test occursin("PresolveParameters", string(algo))
    # the presolver type is baked into the type of `algo`, so it should be inferred as a constant
    val_presolve_enabled(a) = Val(presolve_enabled(a.presolve))
    @test @inferred(val_presolve_enabled(algo)) === Val(true)
    @test @inferred(val_presolve_enabled(algo_default)) === Val(false)
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

@testset "write_sol_file/read_sol_file round-trip with SCIP" begin
    # a non-degenerate LP with a unique optimum, so there is a single right answer to check
    # against: minimize x1 + 2*x2 - x3 s.t. x1+x2+x3 == 6, all in [0, 10]
    c = [1.0, 2.0, -1.0]
    lv = [0.0, 0.0, 0.0]
    uv = [10.0, 10.0, 10.0]
    A = sparse([1.0 1.0 1.0])
    lc, uc = [6.0], [6.0]
    var_names = ["x1", "x2", "x3"]
    milp = MILP(; c, lv, uv, A, lc, uc, var_names)
    path = tempname() * ".mps"
    milp_to_mps(milp, path)

    # SCIP solves and writes its own .sol file: does `read_sol_file` parse it correctly?
    scip = SCIP.Optimizer()
    SCIP.LibSCIP.SCIPreadProb(scip, path, C_NULL)
    SCIP.LibSCIP.SCIPsolve(scip)
    scip_sol_file = tempname() * ".sol"
    open(scip_sol_file, "w") do f
        SCIP.LibSCIP.SCIPprintBestSol(scip, Libc.FILE(f), 0)
    end
    @test read_sol_file(scip_sol_file, var_names) ≈ [0.0, 0.0, 6.0]

    # `write_sol_file` writes a solution: can SCIP itself read it back correctly?
    x_known = [1.0, 2.0, 3.0]
    my_sol_file = tempname() * ".sol"
    write_sol_file(my_sol_file, x_known, var_names)
    scip2 = SCIP.Optimizer()
    SCIP.LibSCIP.SCIPreadProb(scip2, path, C_NULL)
    @test SCIP.LibSCIP.SCIPreadSol(scip2, my_sol_file) == SCIP.LibSCIP.SCIP_OKAY
    sol = SCIP.LibSCIP.SCIPgetBestSol(scip2)
    nvars = SCIP.LibSCIP.SCIPgetNVars(scip2)
    vars_ptr = SCIP.LibSCIP.SCIPgetVars(scip2)
    scip_values = Dict(
        unsafe_string(SCIP.LibSCIP.SCIPvarGetName(v)) => SCIP.LibSCIP.SCIPgetSolVal(scip2, sol, v)
            for v in unsafe_wrap(Array, vars_ptr, Int(nvars))
    )
    @test [scip_values[name] for name in var_names] ≈ x_known

    rm(path; force = true)
    rm(scip_sol_file; force = true)
    rm(my_sol_file; force = true)
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

@testset "presolve(::PaPILOPresolver, ...) strips redundant structure" begin
    milp = _core_padded_milp()
    milp_reduced, state = presolve(CoolPDLP.PaPILOPresolver(), milp)

    @test state isa PaPILOPresolveState
    @test nbvar(milp_reduced) < nbvar(milp)
    @test nbcons(milp_reduced) < nbcons(milp)
end

@testset "postsolve(::PaPILOPresolver, ...) recovers a feasible, optimal reduced solution" begin
    presolver = CoolPDLP.PaPILOPresolver()
    milp = _core_padded_milp()
    milp_reduced, state = presolve(presolver, milp)

    # the only genuine degree of freedom left after presolve should be a single variable fixed
    # by the RHS (x1 == x2 == 4), so any value compatible with its own bounds is optimal
    x_reduced = copy(milp_reduced.lv)
    x_reduced[.!isfinite.(x_reduced)] .= 0.0
    sol_reduced = PrimalDualSolution(x_reduced, zeros(nbcons(milp_reduced)))

    sol_orig = postsolve(presolver, state, sol_reduced)
    @test is_feasible(sol_orig.x, milp)
    @test isapprox(objective_value(sol_orig.x, milp), 8.0; atol = 1.0e-6)
    @test all(isnan, sol_orig.y)  # PaPILO's file-based interface does not round-trip duals
end

@testset "postsolve(::PaPILOPresolver, ...) does not launder an infeasible reduced solution" begin
    # `_core_padded_milp` is small enough that PaPILO's presolve solves it outright, leaving no
    # variable to corrupt; use a real (non-trivial, still feasible and bounded) instance instead
    presolver = CoolPDLP.PaPILOPresolver()
    qps, path = read_instance(Netlib, "afiro")
    milp = MILP(qps; path, name = "afiro")
    milp_reduced, state = presolve(presolver, milp)
    @test nbvar(milp_reduced) > 0

    # push every surviving reduced variable far out of its bounds
    x_reduced = min.(milp_reduced.uv, 1.0e6) .+ 1000.0
    sol_reduced = PrimalDualSolution(x_reduced, zeros(nbcons(milp_reduced)))

    sol_orig = postsolve(presolver, state, sol_reduced)
    @test !is_feasible(sol_orig.x, milp; verbose = false)
end

@testset "solve falls back gracefully when presolve fails and strict = false" begin
    milp, _ = CoolPDLP.random_milp_and_sol(3, 4, 0.6)
    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(),
        presolve = CoolPDLP.PaPILOPresolver(), presolve_strict = false, show_progress = false,
    )
    bogus_dir = joinpath(tempdir(), "coolpdlp-does-not-exist-$(rand(UInt64))")
    sol, stats = withenv("TMPDIR" => bogus_dir) do
        @test_logs (:warn, r"Presolve failed") match_mode = :any solve(milp, algo)
    end
    @test length(sol.x) == nbvar(milp)  # solved the *original* (non-reduced) problem
end

@testset "solve errors instead of falling back when presolve fails and strict = true" begin
    milp, _ = CoolPDLP.random_milp_and_sol(3, 4, 0.6)
    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(),
        presolve = CoolPDLP.PaPILOPresolver(), presolve_strict = true, show_progress = false,
    )
    bogus_dir = joinpath(tempdir(), "coolpdlp-does-not-exist-$(rand(UInt64))")
    withenv("TMPDIR" => bogus_dir) do
        @test_throws ArgumentError solve(milp, algo)
    end
end

@testset "Full solve with presolve enabled matches a direct solve on a reducible problem" begin
    milp = _core_padded_milp()
    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(),
        termination_reltol = 1.0e-8, show_progress = false, presolve = CoolPDLP.PaPILOPresolver(),
    )
    sol, stats = solve(milp, algo)
    @test stats.termination_status == MOI.OPTIMAL
    @test is_feasible(Array(sol.x), milp)
    @test isapprox(objective_value(Array(sol.x), milp), 8.0; atol = 1.0e-4)
    # the dual is not postsolved and comes back as NaN, but must still have the right shape
    @test length(sol.y) == nbcons(milp)
    @test all(isnan, sol.y)
end

@testset "Presolve does not support batched MILPs" begin
    milp, _ = CoolPDLP.random_milp_and_sol(4, 6, 0.5)
    milp_batch = MILP(; c = repeat(milp.c, 1, 3), milp.lv, milp.uv, milp.A, milp.lc, milp.uc)
    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(),
        presolve = CoolPDLP.PaPILOPresolver(), show_progress = false,
    )
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
    algo_np = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), common_opts...)
    algo_p = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), common_opts..., presolve = CoolPDLP.PaPILOPresolver())

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
