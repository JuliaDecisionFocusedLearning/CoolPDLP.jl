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
        @test stats.termination_status == OPTIMAL
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
        @test stats.termination_status != OPTIMAL
    end

    @testset "unbounded direction (c[1] > 0, lv[1] == -Inf)" begin
        milp = MILP(;
            c = [1.0], lv = [-Inf], uv = [Inf], A = spzeros(0, 1), lc = Float64[], uc = Float64[],
        )
        sol, stats = solve(milp, algo)
        @test !any(isnan, sol.x) && !any(isinf, sol.x)
        @test stats.termination_status != OPTIMAL
    end

    @testset "unbounded direction (c[1] < 0, uv[1] == Inf)" begin
        milp = MILP(;
            c = [-1.0], lv = [-Inf], uv = [Inf], A = spzeros(0, 1), lc = Float64[], uc = Float64[],
        )
        sol, stats = solve(milp, algo)
        @test !any(isnan, sol.x) && !any(isinf, sol.x)
        @test stats.termination_status != OPTIMAL
    end
end
