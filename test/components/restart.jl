using CoolPDLP
using CoolPDLP: KKTErrors, Scratch, absolute!!, batched_similar, initialize,
    kkt_errors!, restart_check!, step!
using Random
using Test

Random.seed!(0)

@testset "Recorded errors match their names" begin
    algo = PDLP()
    @testset "batch size $nbatch" for nbatch in (1, 3)
        milps = [CoolPDLP.random_milp_and_sol(20, 30, 0.4)[1] for _ in 1:nbatch]
        milp = MILP(;
            c = nbatch == 1 ? milps[1].c : stack(m -> m.c, milps),
            milps[1].lv, milps[1].uv, milps[1].A, milps[1].lc, milps[1].uc,
            milps[1].int_var,
        )
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
