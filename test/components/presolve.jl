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

const PaPILOExt = Base.get_extension(CoolPDLP, :CoolPDLPPaPILOExt)

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

solved_reduced(milp_reduced; tol = 1.0e-9) = first(
    solve(
        milp_reduced, PrimalDualSolution(milp_reduced),
        PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), termination_reltol = tol, show_progress = false),
    )
)

@testset "Missing PaPILO gives a hint" begin
    script = """
    using CoolPDLP
    milp = MILP(; c = [1.0], lv = [0.0], uv = [1.0], A = zeros(0, 1), lc = Float64[], uc = Float64[])
    try
        presolve(PaPILOPresolver(), milp)
    catch e
        print(e isa MethodError, " ", sprint(showerror, e))
    end
    """
    out = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $script`, String)
    @test startswith(out, "true")
    @test occursin("run `using PaPILO` first", out)
end

@testset "Presolve and postsolve" begin
    presolver = PaPILOPresolver()
    milp = padded_milp()
    milp_reduced, presolve_info = @inferred presolve(presolver, milp)
    @test nbvar(milp_reduced) < nbvar(milp)
    @test nbcons(milp_reduced) < nbcons(milp)
    sol_reduced = PrimalDualSolution(ones(nbvar(milp_reduced)), ones(nbcons(milp_reduced)))
    sol = @inferred postsolve(presolver, presolve_info, sol_reduced)
    @test length(sol.x) == nbvar(milp)
    @test length(sol.y) == nbcons(milp)
    # presolve_info can be reused
    sol2 = postsolve(presolver, presolve_info, sol_reduced)
    @test sol2.x == sol.x
    @test sol2.y == sol.y
end

@testset "Dual postsolve" begin
    presolver = PaPILOPresolver()
    milp = afiro()
    milp_reduced, presolve_info = presolve(presolver, milp)
    sol = postsolve(presolver, presolve_info, solved_reduced(milp_reduced))
    @test kkt_error(sol, milp) <= 1.0e-6
end

@testset "Objective constant of the reduced problem" begin
    presolver = PaPILOPresolver()
    # x1 and x2 are fixed by their rows, so the reduced objective is a constant
    milp = MILP(;
        c = [1.0, 1.0], c0 = 7.0, lv = zeros(2), uv = fill(10.0, 2),
        A = sparse([1.0 0; 0 1.0]), lc = [2.0, 3.0], uc = [2.0, 3.0],
    )
    milp_reduced, _ = presolve(presolver, milp)
    @test milp_reduced.c0 == 12
    # the reduced and original objectives agree at corresponding solutions
    milp = afiro()
    milp_reduced, presolve_info = presolve(presolver, milp)
    @test !iszero(milp_reduced.c0)
    sol_reduced = solved_reduced(milp_reduced)
    sol = postsolve(presolver, presolve_info, sol_reduced)
    @test objective_value(sol_reduced.x, milp_reduced) ≈ objective_value(sol.x, milp)
end

@testset "Problem removed entirely by presolve" begin
    milp = MILP(; c = [1.0, -1.0], lv = zeros(2), uv = [1.0, 2.0], A = spzeros(0, 2), lc = Float64[], uc = Float64[])
    milp_reduced, _ = presolve(PaPILOPresolver(), milp)
    @test nbvar(milp_reduced) == 0
    algo = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), show_progress = false, presolver = PaPILOPresolver())
    sol, stats = solve(milp, algo)
    @test stats.termination_status == MOI.OPTIMAL
    @test sol.x ≈ [0.0, 2.0]
end

@testset "Names of the MILP do not matter" begin
    # names with spaces or duplicates do not survive an MPS file
    presolver = PaPILOPresolver()
    milp = afiro()
    function postsolved_ones(var_names, con_names)
        milp_named = MILP(; milp.c, milp.lv, milp.uv, milp.A, milp.lc, milp.uc, var_names, con_names)
        milp_reduced, presolve_info = presolve(presolver, milp_named)
        sol_reduced = PrimalDualSolution(ones(nbvar(milp_reduced)), ones(nbcons(milp_reduced)))
        return postsolve(presolver, presolve_info, sol_reduced)
    end
    n, m = nbvar(milp), nbcons(milp)
    sol = postsolved_ones(milp.var_names, milp.con_names)
    @test !iszero(sol.x)
    @test !iszero(sol.y)
    for (var_names, con_names) in [(["x $j" for j in 1:n], ["c $i" for i in 1:m]), (fill("x", n), fill("c", m))]
        sol_renamed = postsolved_ones(var_names, con_names)
        @test sol_renamed.x == sol.x
        @test sol_renamed.y == sol.y
    end
    @test PaPILOExt.gather_values(Dict("x1" => 1.0, "x3" => 3.0), ["x1", "x2", "x3"]) == [1.0, 0.0, 3.0]
    @test_throws "unknown name `x3`" PaPILOExt.gather_values(Dict("x3" => 3.0), ["x1", "x2"])
end

@testset "No leftover files" begin
    presolver = PaPILOPresolver()
    dir = mktempdir()
    withenv("TMPDIR" => dir) do
        milp_reduced, presolve_info = presolve(presolver, afiro())
        postsolve(presolver, presolve_info, PrimalDualSolution(milp_reduced))
    end
    # except the settings files that PaPILO.jl itself writes
    @test all(endswith(".set"), readdir(dir))
end

@testset "Integer problems are presolved as their relaxation" begin
    milp = padded_milp()
    milp_int = MILP(; milp.c, milp.lv, milp.uv, milp.A, milp.lc, milp.uc, int_var = [true, false, false, false])
    milp_reduced, presolve_info = presolve(PaPILOPresolver(), milp_int)
    @test nbvar_int(milp_reduced) == 0
    @test nbvar(milp_reduced) == nbvar(first(presolve(PaPILOPresolver(), milp)))
    # the dual is recovered, which PaPILO refuses to do for integer problems
    sol = postsolve(PaPILOPresolver(), presolve_info, PrimalDualSolution(milp_reduced))
    @test !any(isnan, sol.y)
end

@testset "Infeasible or unbounded problems" begin
    milp_infeasible = MILP(;
        c = [1.0, 1.0], lv = zeros(2), uv = fill(10.0, 2),
        A = sparse([1.0 1.0; 1.0 1.0]), lc = [2.0, -Inf], uc = [Inf, 1.0],
    )
    milp_unbounded = MILP(;
        c = [-1.0, -1.0], lv = zeros(2), uv = fill(Inf, 2),
        A = sparse([1.0 -1.0; 1.0 -2.0]), lc = fill(-Inf, 2), uc = [1.0, 3.0],
    )
    algo = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), show_progress = false, presolver = PaPILOPresolver())
    for milp in (milp_infeasible, milp_unbounded)
        @test_throws "infeasible or unbounded" presolve(PaPILOPresolver(), milp)
        @test_throws "infeasible or unbounded" solve(milp, algo)
    end
end

@testset "Solve with presolve" begin
    milp = padded_milp()
    @testset for (T, Ti, M, backend) in [
            (Float64, Int, SparseMatrixCSC, CPU()),
            (Float32, Int32, GPUSparseMatrixCSR, JLBackend()),
        ]
        algo = PDLP(T, Ti, M; backend, termination_reltol = T(1.0e-5), show_progress = false, presolver = PaPILOPresolver())
        sol, stats = @inferred solve(milp, algo)
        milp_conv = perform_conversion(milp, algo.conversion)
        @test typeof(sol) === typeof(PrimalDualSolution(milp_conv))
        @test stats.termination_status == MOI.OPTIMAL
        @test objective_value(Float64.(Array(sol.x)), milp) ≈ 8.0 atol = 1.0e-3
        @test Array(sol.y) ≈ [1.0, 1.0, 0.0] atol = 1.0e-3
        @test relative(stats.err) ≈ kkt_error(sol, milp_conv)
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
    @test relative(stats.err) > 1.0e-5
end

@testset "Limits are reported" begin
    algo = PDLP(
        Float64, Int, SparseMatrixCSC; backend = CPU(), max_kkt_passes = 1,
        show_progress = false, presolver = PaPILOPresolver(),
    )
    sol, stats = solve(afiro(), algo)
    @test stats.termination_status == MOI.ITERATION_LIMIT
    @test length(sol.x) == nbvar(afiro())
end

struct FailingPresolver <: CoolPDLP.AbstractPresolver end
CoolPDLP.presolve(::FailingPresolver, ::MILP) = error("presolve was called")

@testset "Solve from a starting point never presolves" begin
    milp = padded_milp()
    algo = PDLP(Float64, Int, SparseMatrixCSC; backend = CPU(), show_progress = false, presolver = FailingPresolver())
    @test_throws "presolve was called" solve(milp, algo)
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
    JuMP.@objective(model, Max, -x[1] - 2 * x[2] + 3)
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.objective_value(model) ≈ -1.0 atol = 1.0e-3
end
