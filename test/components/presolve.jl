using Adapt: adapt
using CoolPDLP
using CoolPDLP:
    ConversionParameters, GPUSparseMatrixCSR, KKTErrors, Scratch, kkt_errors!, milp_to_mps,
    mps_to_milp, perform_conversion, PrimalDualSolution, relative
using JLArrays: JLBackend
using JuMP: JuMP
using KernelAbstractions: CPU
using MathOptBenchmarkInstances
using MathOptInterface: MathOptInterface as MOI
using PaPILO: PaPILO  # loads the `CoolPDLPPaPILOExt` extension that implements presolve/postsolve
using SparseArrays
using Test

const PaPILOExt = Base.get_extension(CoolPDLP, :CoolPDLPPaPILOExt)
const GPU_CONV = ConversionParameters(Float32, Int32, GPUSparseMatrixCSR; backend = JLBackend())

"""
    relative_kkt_error(sol, milp)

Largest relative KKT error of `sol` on `milp`, the quantity `solve` compares to its tolerance.
Unlike `is_feasible` it also grades the dual half of `sol`.
"""
function relative_kkt_error(sol::PrimalDualSolution, milp::MILP)
    err = KKTErrors(sol)
    kkt_errors!(err, Scratch(sol), sol, milp)
    return relative(err)
end

@testset "presolve throws a MethodError (with a hint) when PaPILO is not loaded" begin
    # spawn a fresh process that never `using`s PaPILO, so `CoolPDLPPaPILOExt` never loads and
    # `presolve`/`postsolve` have no method for a `PaPILOPresolver` at all
    script = """
    using CoolPDLP
    milp = CoolPDLP.MILP(; c = [1.0], lv = [0.0], uv = [1.0], A = zeros(0, 1), lc = Float64[], uc = Float64[])
    try
        CoolPDLP.presolve(CoolPDLP.PaPILOPresolver(), milp)
        println("NO_ERROR")
    catch e
        println("ERROR: ", e isa MethodError, " ", sprint(showerror, e))
    end
    """
    out = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $script`, String)
    @test occursin("ERROR: true", out)
    @test occursin("run `using PaPILO`", out)
end

@testset "Algorithm propagation" begin
    algo_default = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU())
    @test isnothing(algo_default.presolver)

    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(),
        presolver = CoolPDLP.PaPILOPresolver(; verbose = true),
    )
    @test algo.presolver isa CoolPDLP.PaPILOPresolver
    @test algo.presolver.verbose
    @test algo.presolver.dual_postsolve  # duals are recovered unless the user opts out
    @test !CoolPDLP.PaPILOPresolver(; dual_postsolve = false).dual_postsolve
    @test occursin("PaPILOPresolver", string(algo))
    # the presolver type is baked into the type of `algo`, so it should be inferred as a constant
    uses_presolve(a) = Val(!isnothing(a.presolver))
    @test @inferred(uses_presolve(algo)) === Val(true)
    @test @inferred(uses_presolve(algo_default)) === Val(false)
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
    @test milp2.con_names == milp.con_names  # PaPILO matches its solution files up by name
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

@testset "MPS round trip: a Float32 GPU MILP comes back as a host Float64 MILP" begin
    # MPS is a host, `Float64` format, so `milp_to_mps` must accept a MILP living anywhere and
    # `mps_to_milp` always hands back a host problem — `solve` converts it afterwards anyway
    qps, path_afiro = read_instance(Netlib, "afiro")
    milp_cpu = MILP(qps; path = path_afiro, name = "afiro")
    milp_gpu = perform_conversion(milp_cpu, GPU_CONV)

    path = tempname() * ".mps"
    milp_to_mps(milp_gpu, path)
    milp2 = mps_to_milp(path)
    rm(path; force = true)

    @test typeof(milp2) === typeof(milp_cpu)
    @test nbvar(milp2) == nbvar(milp_gpu)
    @test nbcons(milp2) == nbcons(milp_gpu)
    @test milp2.c ≈ Float64.(Array(milp_gpu.c))
    @test milp2.lv ≈ Float64.(Array(milp_gpu.lv))
    @test milp2.uv ≈ Float64.(Array(milp_gpu.uv))
end

@testset "MPS conversion rejects batched MILPs and non-linear models" begin
    milp, _ = CoolPDLP.random_milp_and_sol(4, 6, 0.5)
    milp_batch = MILP(; c = repeat(milp.c, 1, 3), milp.lv, milp.uv, milp.A, milp.lc, milp.uc)
    @test_throws ArgumentError milp_to_mps(milp_batch, tempname() * ".mps")

    quad_model = JuMP.Model()
    JuMP.@variable(quad_model, 0 <= q[1:2] <= 1)
    JuMP.@constraint(quad_model, q[1] * q[2] <= 1)
    JuMP.@objective(quad_model, Min, sum(q))
    quad_path = tempname() * ".mps"
    JuMP.write_to_file(quad_model, quad_path; format = MOI.FileFormats.FORMAT_MPS)
    @test_throws ArgumentError mps_to_milp(quad_path)
    rm(quad_path; force = true)
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

    @test state isa PaPILOExt.PaPILOPresolveState
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
    @test !any(isnan, sol_orig.y)  # the dual travels back with the primal
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

@testset "postsolve types its result like the reduced solution it is given" begin
    # `presolve` may return whatever it likes (here a host `Float64` problem, since PaPILO reads
    # it back from an MPS file), but `postsolve` must hand back a solution the caller can use as
    # is: the shape of the original problem, with `sol_reduced`'s element and array types
    presolver = CoolPDLP.PaPILOPresolver()
    qps, path = read_instance(Netlib, "afiro")
    milp_cpu = MILP(qps; path, name = "afiro")
    milp_gpu = perform_conversion(milp_cpu, GPU_CONV)

    milp_reduced, state = presolve(presolver, milp_cpu)
    @test typeof(milp_reduced) === typeof(milp_cpu)
    @test nbvar(milp_reduced) < nbvar(milp_cpu)

    sol_reduced = perform_conversion(PrimalDualSolution(milp_reduced), GPU_CONV)
    sol_orig = postsolve(presolver, state, sol_reduced)
    @test typeof(sol_orig) === typeof(PrimalDualSolution(milp_gpu))
    @test length(sol_orig.x) == nbvar(milp_cpu)
    @test length(sol_orig.y) == nbcons(milp_cpu)
end

@testset "postsolve recovers a dual solution that solves the original problem" begin
    # PaPILO maps the reduced problem's dual back onto the rows of the original one, so a
    # solution that is optimal for the reduced problem must come back optimal for the original
    # problem *including its dual half*, which `is_feasible` alone would not catch
    presolver = CoolPDLP.PaPILOPresolver()
    qps, path = read_instance(Netlib, "afiro")
    milp = MILP(qps; path, name = "afiro")
    milp_reduced, state = presolve(presolver, milp)

    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(),
        termination_reltol = 1.0e-9, show_progress = false,
    )
    sol_reduced, stats_reduced = solve(milp_reduced, PrimalDualSolution(milp_reduced), algo)
    @test stats_reduced.termination_status == MOI.OPTIMAL
    @test relative_kkt_error(sol_reduced, milp_reduced) <= 1.0e-9

    sol = postsolve(presolver, state, sol_reduced)
    @test !any(isnan, sol.y)
    @test relative_kkt_error(sol, milp) <= 1.0e-6
end

@testset "dual_postsolve = false gives back a NaN dual, and a reduction PaPILO cannot dualize" begin
    # switching the duals off lets PaPILO use its full arsenal, which reduces afiro further
    qps, path = read_instance(Netlib, "afiro")
    milp = MILP(qps; path, name = "afiro")
    milp_dual, _ = presolve(CoolPDLP.PaPILOPresolver(), milp)
    milp_primal, state = presolve(CoolPDLP.PaPILOPresolver(; dual_postsolve = false), milp)
    @test nbvar(milp_primal) < nbvar(milp_dual)

    sol_reduced = PrimalDualSolution(milp_primal)
    sol = postsolve(CoolPDLP.PaPILOPresolver(; dual_postsolve = false), state, sol_reduced)
    @test length(sol.x) == nbvar(milp)
    @test all(isnan, sol.y)
end

@testset "presolve refuses to drop the dual of a problem with integer variables" begin
    # PaPILO never records dual information for an integer problem, so asking for both is an
    # error rather than a silently primal-only answer
    milp = _core_padded_milp()
    milp_int = MILP(;
        milp.c, milp.lv, milp.uv, milp.A, milp.lc, milp.uc,
        int_var = [true, false, false, false],
    )
    @test_throws ArgumentError presolve(CoolPDLP.PaPILOPresolver(), milp_int)
    milp_reduced, _ = presolve(CoolPDLP.PaPILOPresolver(; dual_postsolve = false), milp_int)
    @test nbvar(milp_reduced) <= nbvar(milp_int)
end

@testset "Full solve with presolve on Float32 JLArrays" begin
    milp = _core_padded_milp()
    algo = PDLP(
        Float32, Int32, GPUSparseMatrixCSR; backend = JLBackend(),
        termination_reltol = 1.0f-5, show_progress = false, presolver = CoolPDLP.PaPILOPresolver(),
    )
    sol, stats = solve(milp, algo)
    @test stats.termination_status == MOI.OPTIMAL
    @test eltype(sol.x) === Float32
    @test length(sol.x) == nbvar(milp)
    @test isapprox(objective_value(Float64.(Array(sol.x)), milp), 8.0; atol = 1.0e-3)
end

@testset "solve lets a presolve failure propagate" begin
    milp, _ = CoolPDLP.random_milp_and_sol(3, 4, 0.6)
    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(),
        presolver = CoolPDLP.PaPILOPresolver(), show_progress = false,
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
        termination_reltol = 1.0e-8, show_progress = false, presolver = CoolPDLP.PaPILOPresolver(),
    )
    sol, stats = solve(milp, algo)
    @test stats.termination_status == MOI.OPTIMAL
    @test is_feasible(Array(sol.x), milp)
    @test isapprox(objective_value(Array(sol.x), milp), 8.0; atol = 1.0e-4)
    # the two `x1 == 4`, `x2 == 4` rows each cost 1 per unit, the padding row is free
    @test length(sol.y) == nbcons(milp)
    @test sol.y ≈ [1.0, 1.0, 0.0]
end

@testset "solve grades the solution it returns on the problem it was given" begin
    # the stats describe the solution of the original problem that the caller gets back, not the
    # reduced problem the algorithm iterated on
    milp = _core_padded_milp()
    common_opts = (; termination_reltol = 1.0e-8, show_progress = false)
    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(), common_opts...,
        presolver = CoolPDLP.PaPILOPresolver(),
    )
    sol, stats = solve(milp, algo)
    @test stats.termination_status == MOI.OPTIMAL
    @test CoolPDLP.relative(stats.err) ≈ relative_kkt_error(sol, milp)
    @test stats.time_elapsed >= 0.0  # presolve and postsolve are inside the reported time
end

@testset "a postsolved solution that misses the tolerance is polished, not advertised" begin
    # PaPILO reconstructs the dual of a removed row from the reduced solution it is handed, and
    # gets it badly wrong when that solution is only approximate: afiro at the default tolerance
    # comes back with one dual off by an order of magnitude. The polish is what repairs it
    qps, path = read_instance(Netlib, "afiro")
    milp = MILP(qps; path, name = "afiro")
    presolver = CoolPDLP.PaPILOPresolver()
    common_opts = (; termination_reltol = 1.0e-5, max_kkt_passes = 10^7, show_progress = false)

    milp_reduced, state = presolve(presolver, milp)
    algo_reduced = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), common_opts...)
    sol_reduced, stats_reduced = solve(milp_reduced, PrimalDualSolution(milp_reduced), algo_reduced)
    @test stats_reduced.termination_status == MOI.OPTIMAL
    sol_postsolved = postsolve(presolver, state, sol_reduced)
    @test relative_kkt_error(sol_postsolved, milp) > 1.0e-5  # nowhere near what was asked for

    algo = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), common_opts..., presolver)
    sol, stats = solve(milp, algo)
    @test stats.termination_status == MOI.OPTIMAL
    @test relative_kkt_error(sol, milp) <= 1.0e-5
    @test CoolPDLP.relative(stats.err) ≈ relative_kkt_error(sol, milp)
    # the polish is charged to the same budget as the solve of the reduced problem
    @test stats.kkt_passes > stats_reduced.kkt_passes
end

@testset "a `NaN` dual is dropped from the warm start rather than propagated" begin
    # `dual_postsolve = false` gives back a `NaN` dual: it must not reach the polish iterations,
    # while the primal half of the postsolved solution is still worth starting from
    milp = _core_padded_milp()
    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(),
        termination_reltol = 1.0e-8, show_progress = false,
        presolver = CoolPDLP.PaPILOPresolver(; dual_postsolve = false),
    )
    sol, stats = solve(milp, algo)
    @test !any(isnan, sol.x)
    @test !any(isnan, sol.y)
    @test stats.termination_status == MOI.OPTIMAL
    @test isapprox(objective_value(Array(sol.x), milp), 8.0; atol = 1.0e-4)
end

@testset "the postsolved solution is returned as is when the budget is spent" begin
    # with no KKT passes left there is nothing to polish with, so the honest thing left to do is
    # to hand back the postsolved solution and refuse to call it optimal
    qps, path = read_instance(Netlib, "afiro")
    milp = MILP(qps; path, name = "afiro")
    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(),
        termination_reltol = 1.0e-5, max_kkt_passes = 1, show_progress = false,
        presolver = CoolPDLP.PaPILOPresolver(),
    )
    _, stats = solve(milp, algo)
    @test stats.termination_status != MOI.OPTIMAL
end

@testset "Presolve does not support batched MILPs" begin
    milp, _ = CoolPDLP.random_milp_and_sol(4, 6, 0.5)
    milp_batch = MILP(; c = repeat(milp.c, 1, 3), milp.lv, milp.uv, milp.A, milp.lc, milp.uc)
    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(),
        presolver = CoolPDLP.PaPILOPresolver(), show_progress = false,
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
    algo_p = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), common_opts..., presolver = CoolPDLP.PaPILOPresolver())

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
