using CoolPDLP
import MathOptInterface as MOI
using SparseArrays
using Test

"""Liu & Lu (2025), Example 6. arXiv:2409.14715, eq. (3)."""
function paper_example_6()
    # min 2 x1 + 3 x2  s.t.  x1 + 2 x2 = 1,  x >= 0  →  x* = (0, 1/2), y* = 3/2
    return MILP(;
        c = [2.0, 3.0],
        lv = [0.0, 0.0],
        uv = [Inf, Inf],
        A = sparse([1.0 2.0]),
        lc = [1.0],
        uc = [1.0],
    )
end

@testset "Liu & Lu Example 6: 3235 PDHG steps" begin
    # Vanilla PDHG on the raw LP: η = 0.05, ε = 1e-8, x0 = (1, 2), y0 = 2.
    # `PDHG` already skips Ruiz. `initialize` skips Chambolle–Pock, which
    # `solve` would apply and which changes the step count.
    # This checks the PDHG iteration, not the crossover.
    milp = paper_example_6()
    algo = PDHG(;
        termination_reltol = 1.0e-8,
        check_every = 1,
        max_kkt_passes = 20_000,
        time_limit = 60.0,
        show_progress = false,
        record_error_history = false,
    )
    state = initialize(
        milp, PrimalDualSolution([1.0, 2.0], [2.0]), algo; starting_time = time(),
    )
    state.step_sizes.η = 0.05
    solve!(state, milp, algo)
    @test state.stats.termination_status == MOI.OPTIMAL
    @test state.stats.kkt_passes == 3235
    @test Array(state.sol.x) ≈ [0.0, 0.5] atol = 1.0e-6
    @test Array(state.sol.y) ≈ [1.5] atol = 1.0e-6
end
