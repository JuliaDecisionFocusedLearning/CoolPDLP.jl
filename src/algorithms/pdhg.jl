"""
    PDHG(args...; kwargs...)

Shortcut for [`Algorithm{:PDHG}`](@ref) with some defaults disabled.
"""
function PDHG(args...; kwargs...)
    return Algorithm{:PDHG}(
        args...;
        ruiz_iter = 0,
        primal_weight_damping = NaN,
        sufficient_decay = NaN,
        necessary_decay = NaN,
        artificial_decay = NaN,
        kwargs...
    )
end

"""
    PDHGState

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct PDHGState{
        T <: Number, V <: AbstractVecOrMat{T},
        SS <: StepSizes, Sc <: Scratch, CS <: ConvergenceStats,
    } <: AbstractState{T, V}
    "current solution"
    sol::PrimalDualSolution{T, V}
    "last solution"
    sol_last::PrimalDualSolution{T, V}
    "step sizes"
    step_sizes::SS
    "scratch space"
    scratch::Sc
    "convergence stats"
    stats::CS
end

nbinstances((; sol)::PDHGState) = nbinstances(sol)
function instance(state::PDHGState, i::Int)
    return PDHGState(
        instance(state.sol, i),
        instance(state.sol_last, i),
        instance(state.step_sizes, i),
        instance(state.scratch, i),
        instance(state.stats, i),
    )
end

function initialize(
        milp::MILP{T},
        sol::PrimalDualSolution{T, V},
        algo::Algorithm{:PDHG};
        starting_time::Float64
    ) where {T, V}
    sol_last = zero(sol)
    η = batched_expand(sol.x, fixed_stepsize(milp, algo.step_size))
    ω = batched_expand(sol.x, one(T))
    step_sizes = StepSizes(; η, ω)
    scratch = Scratch(sol)
    # the error history is seeded with the starting point, so its errors must be filled already
    err = KKTErrors(sol)
    kkt_errors!(err, scratch, sol, milp)
    stats = ConvergenceStats(err; starting_time)
    state = PDHGState(; sol, sol_last, step_sizes, scratch, stats)
    return state
end

function solve!(
        state::PDHGState,
        milp::MILP,
        algo::Algorithm{:PDHG}
    )
    prog = init_progress("PDHG iterations:", algo.generic.show_progress)
    must_terminate = false
    @trace while !must_terminate
        @trace for _ in 1:algo.generic.check_every
            step!(state, milp)
            next_progress!(prog, state)
        end
        must_terminate = termination_check!(state, milp, algo)
    end
    finish_progress!(prog)
    return state
end

function step!(
        state::PDHGState{T, V},
        milp::MILP{T},
    ) where {T, V}
    # switch pointers
    state.sol, state.sol_last = state.sol_last, state.sol

    (; sol, sol_last, step_sizes, scratch) = state
    (; x, y) = sol_last
    (; η, ω) = step_sizes
    (; c, lv, uv, A, At, lc, uc) = milp

    τ = transpose(broadcast!!(/, scratch.b1, η, ω))
    σ = transpose(broadcast!!(*, scratch.b2, η, ω))

    # xp = clamp.(x - τ * (c - At * y), lv, uv)
    At_y = mul!(scratch.x, At, y)
    @. sol.x = clamp(x - τ * (c - At_y), lv, uv)
    xdiff = @. scratch.x = 2sol.x - x

    # yp = y - σ * A * (2xp - x) - σ * clamp.(inv(σ) * y - A * (2xp - x), -uc, -lc)
    A_xdiff = mul!(scratch.y, A, xdiff)
    @. sol.y = y - σ * A_xdiff - σ * clamp(inv(σ) * y - A_xdiff, -uc, -lc)

    # other updates
    state.stats.kkt_passes += 1
    return nothing
end
