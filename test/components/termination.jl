using CoolPDLP
import MathOptInterface as MOI
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
    @test stats.termination_status == MOI.OPTIMAL
end

@testset "Termination statuses" begin
    Random.seed!(0)
    milp, _ = CoolPDLP.random_milp_and_sol(20, 30, 0.4)

    @testset "$alg" for alg in (PDHG, PDLP)
        _, stats = solve(milp, alg(; termination_reltol = 0.0, max_kkt_passes = 200))
        @test stats.termination_status == MOI.ITERATION_LIMIT
        @test stats.kkt_passes >= 200

        _, stats = solve(milp, alg(; termination_reltol = 0.0, time_limit = 0.0))
        @test stats.termination_status == MOI.TIME_LIMIT
        @test stats.time_elapsed >= 0
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
        @test stats.termination_status == MOI.OPTIMAL
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
        @test stats.termination_status == MOI.OPTIMAL
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
        @test stats.termination_status != MOI.OPTIMAL
    end

    @testset "unbounded direction (c[1] > 0, lv[1] == -Inf)" begin
        milp = MILP(;
            c = [1.0], lv = [-Inf], uv = [Inf], A = spzeros(0, 1), lc = Float64[], uc = Float64[],
        )
        sol, stats = solve(milp, algo)
        @test !any(isnan, sol.x) && !any(isinf, sol.x)
        @test stats.termination_status != MOI.OPTIMAL
    end

    @testset "unbounded direction (c[1] < 0, uv[1] == Inf)" begin
        milp = MILP(;
            c = [-1.0], lv = [-Inf], uv = [Inf], A = spzeros(0, 1), lc = Float64[], uc = Float64[],
        )
        sol, stats = solve(milp, algo)
        @test !any(isnan, sol.x) && !any(isinf, sol.x)
        @test stats.termination_status != MOI.OPTIMAL
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
