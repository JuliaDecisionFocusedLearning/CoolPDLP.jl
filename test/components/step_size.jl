using CoolPDLP
using CoolPDLP: fixed_stepsize, initialize, primal_weight_update!!
using LinearAlgebra
using SparseArrays
using Random
using Test

@testset "Fixed step size stays finite for a zero spectral norm" begin
    # no constraint rows: spectral_norm(A) == 0, so the old `invnorm_scaling / norm` would
    # divide by zero and produce Inf, corrupting the very first primal step (see #96)
    milp_no_rows = MILP(;
        c = [1.0, -1.0], lv = [0.0, 0.0], uv = [1.0, 1.0],
        A = spzeros(0, 2), lc = Float64[], uc = Float64[],
    )
    @test isfinite(fixed_stepsize(milp_no_rows, PDLP().step_size))

    # an all-zero `A` with actual rows has the same zero spectral norm
    milp_zero_A = MILP(;
        c = [1.0, -1.0], lv = [0.0, 0.0], uv = [1.0, 1.0],
        A = spzeros(2, 2), lc = [-Inf, -Inf], uc = [Inf, Inf],
    )
    @test isfinite(fixed_stepsize(milp_zero_A, PDLP().step_size))
end

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
