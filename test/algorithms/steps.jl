using CoolPDLP
using CoolPDLP: initialize, nbinstances, restart!, restart_check!, save_last!, step!,
    termination_check!
using Random
using Test

@testset "In-place step matches the PDHG update" begin
    Random.seed!(0)
    milps, milp_batch = random_milp_batch(20, 30, 0.4, 3)
    @testset "$alg, batch size $(nbinstances(milp))" for alg in (PDHG, PDLP),
            milp in (milps[1], milp_batch)
        algo = alg()
        state = initialize(milp, PrimalDualSolution(milp), algo; starting_time = time())
        (; c, lv, uv, A, At, lc, uc) = milp
        for _ in 1:10
            # out-of-place update from a copy of the iterates, which `step!` overwrites
            (; x, y) = copy(state.sol)
            (; η, ω) = state.step_sizes
            τ, σ = transpose(η ./ ω), transpose(η .* ω)
            x_new = @. clamp(x - τ * (c - $(At * y)), lv, uv)
            A_xdiff = A * (2 .* x_new .- x)
            y_new = @. y - σ * A_xdiff - σ * clamp(inv(σ) * y - A_xdiff, -uc, -lc)
            step!(state, milp)
            # guard against a vacuous comparison: the iterates must move
            @test !(x_new ≈ x)
            @test state.sol.x ≈ x_new
            @test state.sol.y ≈ y_new
        end
    end
end

"""
    solve_saving_every_step!(state, milp, algo)

Run the PDLP loop of `solve!`, but save the last iterates before every step rather than only
before the last step of each inner loop.
"""
function solve_saving_every_step!(state, milp, algo)
    must_terminate = false
    while !must_terminate
        for _ in 1:algo.generic.check_every
            save_last!(state)
            step!(state, milp)
        end
        must_terminate = termination_check!(state, milp, algo)
        if !must_terminate & restart_check!(state, milp, algo)
            restart!(state, milp, algo)
        end
    end
    return state
end

@testset "Last iterates are only saved before a check" begin
    Random.seed!(1)
    milps, milp_batch = random_milp_batch(20, 30, 0.4, 3)
    same(sol1, sol2) = sol1.x == sol2.x && sol1.y == sol2.y
    @testset "check_every = $check_every, batch size $(nbinstances(milp))" for check_every in (1, 2, 7),
            milp in (milps[1], milp_batch)
        algo = PDLP(;
            check_every, termination_reltol = 1.0e-12, max_kkt_passes = 500,
            record_error_history = false,
        )
        init() = initialize(milp, PrimalDualSolution(milp), algo; starting_time = time())
        state = solve!(init(), milp, algo)
        state_ref = solve_saving_every_step!(init(), milp, algo)
        # guard against a vacuous comparison: the restart check must have triggered
        @test state.iteration.outer > 0
        @test state.iteration.outer == state_ref.iteration.outer
        @test state.stats.kkt_passes == state_ref.stats.kkt_passes
        @test same(state.sol, state_ref.sol)
        @test same(state.sol_avg, state_ref.sol_avg)
        @test same(state.sol_last, state_ref.sol_last)
        @test same(state.sol_avg_last, state_ref.sol_avg_last)
    end
end
