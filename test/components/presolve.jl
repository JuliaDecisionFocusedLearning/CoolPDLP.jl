using Adapt: adapt
using CoolPDLP
using CoolPDLP:
    ConversionParameters, GPUSparseMatrixCSR, KKTErrors, Scratch, kkt_errors!,
    perform_conversion, PrimalDualSolution, relative
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
    milp_reduced, presolve_info = presolve(CoolPDLP.PaPILOPresolver(), milp)

    @test presolve_info isa PaPILOExt.PaPILOPresolveInfo
    @test nbvar(milp_reduced) < nbvar(milp)
    @test nbcons(milp_reduced) < nbcons(milp)
end

@testset "postsolve recovers a dual solution that solves the original problem" begin
    # PaPILO maps the reduced problem's dual back onto the rows of the original one, so a
    # solution that is optimal for the reduced problem must come back optimal for the original
    # problem *including its dual half*, which `is_feasible` alone would not catch
    presolver = CoolPDLP.PaPILOPresolver()
    qps, path = read_instance(Netlib, "afiro")
    milp = MILP(qps; path, name = "afiro")
    milp_reduced, presolve_info = presolve(presolver, milp)

    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(),
        termination_reltol = 1.0e-9, show_progress = false,
    )
    sol_reduced, stats_reduced = solve(milp_reduced, PrimalDualSolution(milp_reduced), algo)
    @test stats_reduced.termination_status == MOI.OPTIMAL
    @test relative_kkt_error(sol_reduced, milp_reduced) <= 1.0e-9

    sol = postsolve(presolver, presolve_info, sol_reduced)
    @test !any(isnan, sol.y)
    @test relative_kkt_error(sol, milp) <= 1.0e-6
end

@testset "dual_postsolve = false gives back a NaN dual, and a reduction PaPILO cannot dualize" begin
    # switching the duals off lets PaPILO use its full arsenal, which reduces afiro further
    qps, path = read_instance(Netlib, "afiro")
    milp = MILP(qps; path, name = "afiro")
    milp_dual, _ = presolve(CoolPDLP.PaPILOPresolver(), milp)
    milp_primal, presolve_info = presolve(CoolPDLP.PaPILOPresolver(; dual_postsolve = false), milp)
    @test nbvar(milp_primal) < nbvar(milp_dual)

    sol_reduced = PrimalDualSolution(milp_primal)
    sol = postsolve(CoolPDLP.PaPILOPresolver(; dual_postsolve = false), presolve_info, sol_reduced)
    @test length(sol.x) == nbvar(milp)
    @test all(isnan, sol.y)
end

@testset "presolve hands PaPILO the continuous relaxation of an integer problem" begin
    # `solve` only ever tackles the relaxation, so that is what gets presolved: an integer
    # problem reduces exactly like its relaxation, and its dual comes back — which it could not
    # if PaPILO saw the integrality, since it records no dual information for such a problem
    presolver = CoolPDLP.PaPILOPresolver()
    milp = _core_padded_milp()
    milp_int = MILP(;
        milp.c, milp.lv, milp.uv, milp.A, milp.lc, milp.uc,
        int_var = [true, false, false, false],
    )

    milp_reduced, _ = presolve(presolver, milp)
    milp_reduced_int, presolve_info = presolve(presolver, milp_int)
    @test nbvar(milp_reduced_int) == nbvar(milp_reduced)
    @test nbcons(milp_reduced_int) == nbcons(milp_reduced)
    @test nbvar_int(milp_reduced_int) == 0

    sol = postsolve(presolver, presolve_info, PrimalDualSolution(milp_reduced_int))
    @test length(sol.x) == nbvar(milp_int)
    @test !any(isnan, sol.y)
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

@testset "presolve throws when PaPILO proves the problem infeasible or unbounded" begin
    presolver = CoolPDLP.PaPILOPresolver()
    # x1 + x2 >= 2 and x1 + x2 <= 1
    milp_infeasible = MILP(;
        c = [1.0, 1.0], lv = zeros(2), uv = fill(10.0, 2),
        A = sparse([1.0 1.0; 1.0 1.0]), lc = [2.0, -Inf], uc = [Inf, 1.0],
    )
    # x1 = t + 1, x2 = t is feasible for every t >= 0, with objective -2t - 1
    milp_unbounded = MILP(;
        c = [-1.0, -1.0], lv = zeros(2), uv = fill(Inf, 2),
        A = sparse([1.0 -1.0; 1.0 -2.0]), lc = fill(-Inf, 2), uc = [1.0, 3.0],
    )
    @test_throws "infeasible or unbounded" presolve(presolver, milp_infeasible)
    @test_throws "infeasible or unbounded" presolve(presolver, milp_unbounded)

    algo = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), presolver, show_progress = false)
    @test_throws ErrorException solve(milp_infeasible, algo)
end

@testset "presolve and postsolve do not depend on the names of the MILP" begin
    # names with spaces or duplicates do not survive an MPS file, and PaPILO matches its
    # solution files up by name: a mismatch used to come back as a silently wrong solution
    presolver = CoolPDLP.PaPILOPresolver()
    qps, path = read_instance(Netlib, "afiro")
    milp = MILP(qps; path, name = "afiro")
    function postsolved_ones(var_names, con_names)
        milp_named = MILP(;
            milp.c, milp.lv, milp.uv, milp.A, milp.lc, milp.uc, var_names, con_names,
        )
        milp_reduced, presolve_info = presolve(presolver, milp_named)
        sol_reduced = PrimalDualSolution(ones(nbvar(milp_reduced)), ones(nbcons(milp_reduced)))
        return postsolve(presolver, presolve_info, sol_reduced)
    end
    n, m = nbvar(milp), nbcons(milp)
    sol = postsolved_ones(milp.var_names, milp.con_names)
    @test !iszero(sol.x)
    @test !iszero(sol.y)
    sol_spaces = postsolved_ones(["x $j" for j in 1:n], ["c $i" for i in 1:m])
    @test sol_spaces.x == sol.x
    @test sol_spaces.y == sol.y
    sol_duplicates = postsolved_ones(fill("x", n), fill("c", m))
    @test sol_duplicates.x == sol.x
    @test sol_duplicates.y == sol.y
end

@testset "presolve_info can be postsolved several times, and owns no file" begin
    presolver = CoolPDLP.PaPILOPresolver()
    qps, path = read_instance(Netlib, "afiro")
    milp = MILP(qps; path, name = "afiro")
    dir = mktempdir()
    withenv("TMPDIR" => dir) do
        milp_reduced, presolve_info = presolve(presolver, milp)
        sol_reduced = PrimalDualSolution(ones(nbvar(milp_reduced)), ones(nbcons(milp_reduced)))
        sol1 = postsolve(presolver, presolve_info, sol_reduced)
        sol2 = postsolve(presolver, presolve_info, sol_reduced)
        @test sol1.x == sol2.x
        @test sol1.y == sol2.y
    end
    # the only leftovers are the settings files that PaPILO.jl itself writes
    @test all(endswith(".set"), readdir(dir))
end

@testset "postsolve rejects a solution file with unknown names" begin
    values = Dict("x1" => 1.0, "x3" => 3.0)
    @test PaPILOExt.gather_values(values, ["x1", "x2", "x3"]) == [1.0, 0.0, 3.0]
    @test_throws ErrorException PaPILOExt.gather_values(values, ["x1", "x2"])
end

@testset "Full solve with presolve on a reducible problem" begin
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
    @test sol.y ≈ [1.0, 1.0, 0.0]
    # the stats describe the solution of the original problem that the caller gets back, not the
    # reduced problem the algorithm iterated on
    @test CoolPDLP.relative(stats.err) ≈ relative_kkt_error(sol, milp)
end

@testset "a postsolved solution that misses the tolerance is not advertised as optimal" begin
    # PaPILO reconstructs the dual of a removed row from the reduced solution it is handed, and
    # gets it badly wrong when that solution is only approximate: afiro at this tolerance comes
    # back with one dual off by an order of magnitude
    qps, path = read_instance(Netlib, "afiro")
    milp = MILP(qps; path, name = "afiro")
    presolver = CoolPDLP.PaPILOPresolver()
    common_opts = (; termination_reltol = 1.0e-5, max_kkt_passes = 10^7, show_progress = false)

    milp_reduced, presolve_info = presolve(presolver, milp)
    algo_reduced = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), common_opts...)
    sol_reduced, stats_reduced = solve(milp_reduced, PrimalDualSolution(milp_reduced), algo_reduced)
    @test stats_reduced.termination_status == MOI.OPTIMAL
    sol_postsolved = postsolve(presolver, presolve_info, sol_reduced)
    @test relative_kkt_error(sol_postsolved, milp) > 1.0e-5  # nowhere near what was asked for

    algo = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), common_opts..., presolver)
    sol, stats = solve(milp, algo)
    @test stats.termination_status == MOI.ALMOST_OPTIMAL
    @test CoolPDLP.relative(stats.err) ≈ relative_kkt_error(sol, milp)
    @test relative_kkt_error(sol, milp) > 1.0e-5
    # the iterations all happened on the reduced problem
    @test stats.kkt_passes == stats_reduced.kkt_passes
end

@testset "a `NaN` dual is never advertised as optimal" begin
    # without dual postsolve the dual comes back as `NaN`, which no tolerance can vouch for,
    # including in the algorithm's own element and array types
    milp = _core_padded_milp()
    algo = PDLP(
        Float32, Int32, GPUSparseMatrixCSR; backend = JLBackend(),
        termination_reltol = 1.0f-5, show_progress = false,
        presolver = CoolPDLP.PaPILOPresolver(; dual_postsolve = false),
    )
    sol, stats = solve(milp, algo)
    @test stats.termination_status == MOI.ALMOST_OPTIMAL
    @test typeof(sol) === typeof(PrimalDualSolution(perform_conversion(milp, GPU_CONV)))
    @test all(isnan, Array(sol.y))
    @test isapprox(objective_value(Float64.(Array(sol.x)), milp), 8.0; atol = 1.0e-3)
end

@testset "a limit reached on the reduced problem is reported as such" begin
    qps, path = read_instance(Netlib, "afiro")
    milp = MILP(qps; path, name = "afiro")
    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(),
        termination_reltol = 1.0e-5, max_kkt_passes = 1, show_progress = false,
        presolver = CoolPDLP.PaPILOPresolver(),
    )
    sol, stats = solve(milp, algo)
    @test stats.termination_status == MOI.ITERATION_LIMIT
    @test length(sol.x) == nbvar(milp)
end

struct FailingPresolver <: CoolPDLP.AbstractPresolver end
CoolPDLP.presolve(::FailingPresolver, ::MILP) = error("presolve was called")

@testset "solve from a starting point never presolves" begin
    milp = _core_padded_milp()
    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(),
        termination_reltol = 1.0e-8, show_progress = false, presolver = FailingPresolver(),
    )
    @test_throws "presolve was called" solve(milp, algo)
    sol, stats = solve(milp, PrimalDualSolution(milp), algo)
    @test stats.termination_status == MOI.OPTIMAL
    @test isapprox(objective_value(sol.x, milp), 8.0; atol = 1.0e-4)
end

@testset "presolve through JuMP" begin
    model = JuMP.Model(CoolPDLP.Optimizer)
    JuMP.set_silent(model)
    JuMP.set_attribute(model, "presolver", CoolPDLP.PaPILOPresolver())
    JuMP.@variable(model, 0 <= x[1:2] <= 10)
    JuMP.@constraint(model, x[1] + x[2] >= 4)
    JuMP.@objective(model, Min, x[1] + 2 * x[2])
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test isapprox(JuMP.objective_value(model), 4.0; atol = 1.0e-3)
end

@testset "`nothing` is the identity presolver" begin
    # `presolver = nothing` is not a special case in `solve`, it is the identity step, so the
    # pipeline is the same either way
    milp = _core_padded_milp()
    milp_reduced, presolve_info = presolve(nothing, milp)
    @test milp_reduced === milp
    @test isnothing(presolve_info)
    sol = PrimalDualSolution(milp)
    @test postsolve(nothing, presolve_info, sol) === sol
end

@testset "presolve, postsolve and solve are type-stable" begin
    # `solve` runs `presolve`/`postsolve` as ordinary steps of its pipeline, next to `preprocess`
    # and `initialize`, so it inherits their return types: a presolver is expected to be
    # inferrable, and `solve` must come out just as concrete with one as without
    presolver = CoolPDLP.PaPILOPresolver()
    milp = _core_padded_milp()
    milp_reduced, presolve_info = @inferred presolve(presolver, milp)
    @inferred postsolve(presolver, presolve_info, PrimalDualSolution(milp_reduced))

    common_opts = (; termination_reltol = 1.0e-8, show_progress = false)
    algo = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), common_opts..., presolver)
    algo_plain = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), common_opts...)
    @inferred solve(milp, algo)
    # `Base.return_types` rather than `Base.infer_return_type`, which needs Julia >= 1.11
    solve_type(a) = only(Base.return_types(solve, Tuple{typeof(milp), typeof(a)}))
    @test isconcretetype(solve_type(algo))
    @test solve_type(algo) === solve_type(algo_plain)
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
