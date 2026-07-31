using CoolPDLP
using CoolPDLP: initialize, primal_weight_update!!
using Random
using Test

include("../fixtures.jl")

@testset "Primal weight left alone without movement" begin
    Random.seed!(0)
    milps, milp_batch = random_milp_batch(20, 30, 0.4, 3)
    algo = PDLP()
    @testset "batch size $(CoolPDLP.nbinstances(milp))" for milp in (milps[1], milp_batch)
        state = initialize(milp, PrimalDualSolution(milp), algo; starting_time = time())
        ω = copy(state.step_sizes.ω)
        # the candidate sits exactly on the restart point, so there is nothing to learn from it
        updated = primal_weight_update!!(
            state.scratch, state.step_sizes, state.sol, state.sol, algo.step_size
        )
        @test updated == ω
    end
end
