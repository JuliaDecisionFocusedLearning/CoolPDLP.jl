using CoolPDLP
using CoolPDLP: ITERATION_LIMIT, OPTIMAL, TIME_LIMIT
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
    @test stats.termination_status == OPTIMAL
end

@testset "Termination statuses" begin
    Random.seed!(0)
    milp, _ = CoolPDLP.random_milp_and_sol(20, 30, 0.4)

    @testset "$alg" for alg in (PDHG, PDLP)
        _, stats = solve(milp, alg(; termination_reltol = 0.0, max_kkt_passes = 200))
        @test stats.termination_status == ITERATION_LIMIT
        @test stats.kkt_passes >= 200

        _, stats = solve(milp, alg(; termination_reltol = 0.0, time_limit = 0.0))
        @test stats.termination_status == TIME_LIMIT
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
        @test stats.termination_status == OPTIMAL
        @test sol.x == clamp.(0.0, lv, uv)
        @test is_feasible(sol.x, milp)
        @test objective_value(sol.x, milp) == 0
    end
end
