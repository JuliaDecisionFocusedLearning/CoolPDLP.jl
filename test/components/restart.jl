using CoolPDLP
using CoolPDLP: IterationCounter, KKTErrors, RestartParameters, RestartStats, Scratch,
    absolute!!, batched_mean, batched_similar, initialize, kkt_errors!, nbinstances,
    restart_check!, should_restart, step!
using Random
using Test

@testset "Recorded errors match their names" begin
    Random.seed!(0)
    milps, milp_batch = random_milp_batch(20, 30, 0.4, 3)
    algo = PDLP()

    @testset "batch size $(nbinstances(milp))" for milp in (milps[1], milp_batch)
        state = initialize(milp, PrimalDualSolution(milp), algo; starting_time = time())
        for _ in 1:20
            step!(state, milp)
        end
        restart_check!(state, milp, algo)

        (; restart_stats) = state
        fresh(sol) = kkt_errors!(KKTErrors(sol), Scratch(sol), sol, milp)
        @test restart_stats.err_current ≈ fresh(state.sol)
        @test restart_stats.err_avg ≈ fresh(state.sol_avg)
        @test restart_stats.err_last ≈ fresh(state.sol_last)
        @test restart_stats.err_avg_last ≈ fresh(state.sol_avg_last)
        @test restart_stats.err_restart ≈ fresh(state.sol_restart)

        (; ω) = state.step_sizes
        abs_err(err) = absolute!!(batched_similar(restart_stats.abs_candidate), err, ω)
        @test restart_stats.abs_candidate ≈
            min.(abs_err(restart_stats.err_current), abs_err(restart_stats.err_avg))
        @test restart_stats.abs_restart ≈ abs_err(restart_stats.err_restart)
    end
end

@testset "Parametrizable batch aggregation" begin
    sol = PrimalDualSolution(zeros(6, 3), zeros(4, 3))
    stats = RestartStats(sol)
    stats.abs_candidate .= [0.1, 0.1, 1.0]
    stats.abs_candidate_last .= 2.0
    stats.abs_restart .= 1.0
    iteration = IterationCounter(0, 1, 10)

    params(f) = RestartParameters(;
        sufficient_decay = 0.5, necessary_decay = 0.8, artificial_decay = 1.0,
        batch_aggregation = f,
    )
    # the mean of the candidate errors has decayed enough, their maximum has not
    @test should_restart(stats, iteration, params(batched_mean))
    @test !should_restart(stats, iteration, params(maximum))

    # the aggregation reaches the restart parameters through the algorithm constructor
    @test PDLP(; batch_aggregation = maximum).restart.batch_aggregation === maximum
end
