using CoolPDLP
using CoolPDLP: termination_status
import MathOptInterface as MOI
using MathOptBenchmarkInstances
using Random
using SparseArrays
using Test

@testset "Termination check not skipped" begin
    c = [1.0, 1.0]
    lc, A, uc = [1.0], sparse([1.0 1.0]), [Inf]
    lv, uv = [0.0, 0.0], [Inf, Inf]

    milp = CoolPDLP.MILP(; c, lv, uv, A, lc, uc)
    algo = CoolPDLP.PDLP()
    sol, stats = CoolPDLP.solve(milp, algo)
    @test termination_status(stats) == MOI.OPTIMAL
end

@testset "Termination statuses" begin
    Random.seed!(0)
    milp, _ = CoolPDLP.random_milp_and_sol(20, 30, 0.4)

    @testset "$alg" for alg in (PDHG, PDLP)
        _, stats = solve(milp, alg(; termination_reltol = 0.0, max_kkt_passes = 200))
        @test termination_status(stats) == MOI.ITERATION_LIMIT
        @test stats.kkt_passes >= 200

        _, stats = solve(milp, alg(; termination_reltol = 0.0, time_limit = 0.0))
        @test termination_status(stats) == MOI.TIME_LIMIT
        @test stats.time_elapsed >= 0
    end
end

@testset "No time limit by default" begin
    Random.seed!(0)
    milp, _ = CoolPDLP.random_milp_and_sol(20, 30, 0.4)
    @testset "$alg" for alg in (PDHG, PDLP)
        algo = alg(; termination_reltol = 0.0, max_kkt_passes = 200)
        @test algo.generic.time_limit == Inf
        _, stats = solve(milp, algo)
        @test termination_status(stats) == MOI.ITERATION_LIMIT
        # the elapsed time is still measured outside of Reactant
        @test stats.time_elapsed > 0
    end
end

@testset "Empty problem" begin
    # no constraint and no objective: the solution is read off the variable bounds
    lv, uv = [-1.0, 0.0, 2.0], [1.0, 3.0, 4.0]
    milp = MILP(;
        c = zeros(3), lv, uv,
        A = spzeros(0, 3), lc = Float64[], uc = Float64[],
    )
    @test nbcons(milp) == 0

    @testset "$alg" for alg in (PDHG, PDLP)
        sol, stats = solve(milp, alg())
        @test termination_status(stats) == MOI.OPTIMAL
        @test sol.x == clamp.(0.0, lv, uv)
        @test is_feasible(sol.x, milp)
        @test objective_value(sol.x, milp) == 0
    end
end

@testset "Constraint-free problem with nonzero objective" begin
    # no constraint rows, but a nonzero objective: the fixed step size used to be
    # `0.9 / spectral_norm(A) == 0.9 / 0 == Inf`, corrupting the very first primal step
    # (`Inf * 0 == NaN` for zero-coefficient variables, see #96)
    c = [1.0, -1.0, 0.0]
    lv, uv = [0.0, 0.0, 0.0], [5.0, 5.0, 5.0]
    milp = MILP(; c, lv, uv, A = spzeros(0, 3), lc = Float64[], uc = Float64[])
    @test nbcons(milp) == 0

    @testset "$alg" for alg in (PDHG, PDLP)
        sol, stats = solve(milp, alg())
        @test termination_status(stats) == MOI.OPTIMAL
        @test !any(isnan, sol.x)
        @test sol.x == [0.0, 5.0, 0.0]
        @test objective_value(sol.x, milp) == -5.0
        # the early exit must still populate the stats, not leave them at their NaN/0.0 defaults
        @test stats.err.gap == 0
        @test stats.time_elapsed > 0
    end
end

@testset "Constraint-free problem with infeasible or unbounded box" begin
    # this package has no dedicated infeasible/unbounded status: falling through to the
    # general iteration loop (which no longer blows up thanks to the `fixed_stepsize` fix)
    # is the same "no detection, just don't converge" behavior as any other bad problem,
    # whereas silently claiming OPTIMAL from the early exit would be actively wrong
    algo = PDLP(; max_kkt_passes = 20, show_progress = false)

    @testset "infeasible box (lv > uv)" begin
        milp = MILP(;
            c = [1.0], lv = [5.0], uv = [2.0], A = spzeros(0, 1), lc = Float64[], uc = Float64[],
        )
        sol, stats = solve(milp, algo)
        @test !any(isnan, sol.x) && !any(isinf, sol.x)
        @test termination_status(stats) != MOI.OPTIMAL
    end

    @testset "unbounded direction (c[1] > 0, lv[1] == -Inf)" begin
        milp = MILP(;
            c = [1.0], lv = [-Inf], uv = [Inf], A = spzeros(0, 1), lc = Float64[], uc = Float64[],
        )
        sol, stats = solve(milp, algo)
        @test !any(isnan, sol.x) && !any(isinf, sol.x)
        @test termination_status(stats) != MOI.OPTIMAL
    end

    @testset "unbounded direction (c[1] < 0, uv[1] == Inf)" begin
        milp = MILP(;
            c = [-1.0], lv = [-Inf], uv = [Inf], A = spzeros(0, 1), lc = Float64[], uc = Float64[],
        )
        sol, stats = solve(milp, algo)
        @test !any(isnan, sol.x) && !any(isinf, sol.x)
        @test termination_status(stats) != MOI.OPTIMAL
    end
end

@testset "Error history" begin
    Random.seed!(0)
    milp, _ = CoolPDLP.random_milp_and_sol(20, 30, 0.4)
    check_every, max_kkt_passes = 10, 100

    @testset "$alg" for alg in (PDHG, PDLP)
        algo = alg(;
            termination_reltol = 0.0, check_every, max_kkt_passes,
            record_error_history = true,
        )
        _, stats = solve(milp, algo)
        history = stats.error_history

        # the history is actually recorded, not left at its single seed entry
        @test length(history) > 1
        @test length(history) == 1 + div(max_kkt_passes, check_every)

        passes, errors = first.(history), last.(history)

        # it is indexed by the number of KKT passes, starting at the initial point
        @test first(passes) == 0
        @test issorted(passes)
        @test last(passes) == stats.kkt_passes

        recorded = map(CoolPDLP.relative, errors)

        # the seed is the initial point, whose errors must be filled in rather than left NaN
        @test all(isfinite, recorded)
        # the history must not be the same value repeated: it tracks an evolving quantity
        @test first(recorded) != last(recorded)
        # the last entry reflects the errors the algorithm actually terminated on
        @test last(recorded) == CoolPDLP.relative(stats.err)

        # every entry is an independent snapshot: mutating the live errors afterwards, as
        # `kkt_errors!` does on every check, must not rewrite what was already recorded
        @test all(err -> err !== stats.err, errors)
        @test allunique(map(objectid, errors))
        stats.err.primal += 1
        @test map(CoolPDLP.relative, last.(stats.error_history)) == recorded
    end

    @testset "disabled: $alg" for alg in (PDHG, PDLP)
        algo = alg(;
            termination_reltol = 0.0, check_every, max_kkt_passes,
            record_error_history = false,
        )
        _, stats = solve(milp, algo)
        # only the seed entry survives, and it is still a usable snapshot
        @test length(stats.error_history) == 1
        @test first(first(stats.error_history)) == 0
        @test isfinite(CoolPDLP.relative(last(first(stats.error_history))))
    end
end

@testset "Absolute and relative tolerances" begin
    using CoolPDLP: KKTErrors, TerminationParameters, ConvergenceStats, set_termination_status!!
    # raw norms ‖b‖ = 99, ‖c‖ = 9, |pobj| + |dobj| = 0: the stored scales are `1 + norm`
    err = KKTErrors(;
        primal = 0.5, primal_scale = 100.0, dual = 0.04, dual_scale = 10.0,
        gap = 1.0e-3, gap_scale = 1.0,
    )
    function converged(; termination_reltol, termination_abstol = termination_reltol)
        params = TerminationParameters(; termination_reltol, termination_abstol, max_kkt_passes = 10^9)
        stats = ConvergenceStats(copy(err))
        stop = set_termination_status!!(stats, 0.0, params, Inf)
        @test stop == (termination_status(stats) == MOI.OPTIMAL)
        return stop
    end
    # relative only (the default): primal 0.5 / 100 = 5e-3, dual 4e-3, gap 1e-3
    @test converged(; termination_reltol = 5.01e-3)
    @test !converged(; termination_reltol = 4.9e-3)
    # absolute only: the largest raw error is 0.5
    @test converged(; termination_reltol = 0.0, termination_abstol = 0.5)
    @test !converged(; termination_reltol = 0.0, termination_abstol = 0.49)
    # mixed, with ϵₐ = 1e-3: 0.5 ≤ ϵₐ + ϵᵣ * 99 needs ϵᵣ ≥ 5.04e-3,
    # 0.04 ≤ ϵₐ + ϵᵣ * 9 needs ϵᵣ ≥ 4.33e-3, and the gap (zero objective scale) needs ϵₐ ≥ 1e-3
    @test converged(; termination_reltol = 0.0051, termination_abstol = 1.0e-3)
    @test !converged(; termination_reltol = 0.0051, termination_abstol = 0.9e-3)
    @test !converged(; termination_reltol = 0.005, termination_abstol = 1.0e-3)
    # the default absolute tolerance is the relative one
    @test PDLP(; termination_reltol = 1.0e-3).termination.termination_abstol == 1.0e-3
    # NaN errors never count as converged
    err.primal = NaN
    @test !converged(; termination_reltol = 1.0)
    @testset "Batched" begin
        errb = KKTErrors(;
            primal = [0.5, 0.05], primal_scale = [100.0, 100.0], dual = [0.0, 0.0],
            dual_scale = [1.0, 1.0], gap = [0.0, 0.0], gap_scale = [1.0, 1.0],
        )
        dest = zeros(2)
        margin = CoolPDLP.termination_margin!!(
            dest, errb,
            TerminationParameters(;
                termination_reltol = 0.0, termination_abstol = 0.1,
                max_kkt_passes = 1,
            ),
        )
        @test margin === dest
        @test margin ≈ [0.4, -0.05]
        stats = ConvergenceStats(errb)
        params = TerminationParameters(;
            termination_reltol = 0.0, termination_abstol = 0.1,
            max_kkt_passes = 10^9,
        )
        # one instance is not converged yet, so the batch goes on
        @test !set_termination_status!!(stats, dest, params, Inf)
        errb.primal[1] = 0.1
        @test set_termination_status!!(stats, dest, params, Inf)
    end
end

@testset "Termination norm" begin
    using LinearAlgebra: norm
    Random.seed!(1)
    milp, sol = CoolPDLP.random_milp_and_sol(20, 30, 0.4)
    sol = PrimalDualSolution(randn(nbvar(milp)), randn(nbcons(milp)))
    scratch = CoolPDLP.Scratch(sol)
    (; c, A, lc, uc) = milp
    err2 = CoolPDLP.kkt_errors!(CoolPDLP.KKTErrors(sol), scratch, sol, milp, L2Norm())
    errinf = CoolPDLP.kkt_errors!(CoolPDLP.KKTErrors(sol), scratch, sol, milp, LInfNorm())
    Ax = A * sol.x
    @test err2.primal ≈ norm(Ax - clamp.(Ax, lc, uc))
    @test errinf.primal ≈ norm(Ax - clamp.(Ax, lc, uc), Inf)
    @test errinf.primal_scale ≈ 1 + norm(CoolPDLP.combine.(lc, uc), Inf)
    @test errinf.dual_scale ≈ 1 + norm(c, Inf)
    @test errinf.primal < err2.primal
    @test errinf.dual <= err2.dual
    @test errinf.gap == err2.gap

    # batched columns are normed one by one
    solb = PrimalDualSolution(stack([sol.x, 2 .* sol.x]), stack([sol.y, 2 .* sol.y]))
    milpb = MILP(; c = stack([c, c]), milp.lv, milp.uv, A, lc, uc)
    errb = CoolPDLP.kkt_errors!(CoolPDLP.KKTErrors(solb), CoolPDLP.Scratch(solb), solb, milpb, LInfNorm())
    @test errb.primal[1] ≈ errinf.primal
    @test errb.dual[1] ≈ errinf.dual
    @test CoolPDLP.colnorm!!(zeros(2), zeros(0, 2), LInfNorm()) == [0.0, 0.0]

    netlib = MILP(read_instance(Netlib, "afiro")[1])
    @testset "$alg" for alg in (PDHG, PDLP)
        sol, stats = solve(netlib, alg(; termination_reltol = 1.0e-4, termination_norm = LInfNorm()))
        @test termination_status(stats) == MOI.OPTIMAL
        # the reported errors, which stopped the solve, are the L∞ errors of the solution
        err = CoolPDLP.kkt_errors!(CoolPDLP.KKTErrors(sol), CoolPDLP.Scratch(sol), sol, netlib, LInfNorm())
        @test err.primal ≈ stats.err.primal
        @test err.dual ≈ stats.err.dual
        @test err.primal <= 1.0e-4 * err.primal_scale
    end

    @test_throws MethodError PDLP(; termination_norm = 1)
end
