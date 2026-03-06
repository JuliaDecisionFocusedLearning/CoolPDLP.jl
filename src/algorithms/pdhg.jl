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
        T <: Number, V <: DenseVector{T},
    } <: AbstractState{T, V}
    "current solution"
    sol::PrimalDualSolution{T, V}
    "last solution"
    sol_last::PrimalDualSolution{T, V}
    "step sizes"
    step_sizes::StepSizes{T}
    "step size parameters"
    step_size_params::StepSizeParameters{T}
    "scratch space"
    scratch::Scratch{T, V}
    "convergence stats"
    stats::ConvergenceStats{T}
end

function initialize(
        milp::AbstractProgram{T, V},
        sol::PrimalDualSolution{T, V},
        algo::Algorithm{:PDHG, T};
        starting_time::Float64
    ) where {T, V}
    sol_last = zero(sol)
    η, ω, norm_A, norm_Q = init_stepsize(milp, algo.step_size)
    step_sizes = StepSizes(; η, ω, norm_A, norm_Q)
    step_size_params = algo.step_size
    scratch = Scratch(sol, milp)
    stats = ConvergenceStats(T; starting_time)
    state = PDHGState(; sol, sol_last, step_sizes, step_size_params, scratch, stats)
    return state
end

function solve!(
        state::PDHGState,
        milp::AbstractProgram,
        algo::Algorithm{:PDHG}
    )
    progress = ProgressUnknown(desc = "PDHG iterations:", enabled = algo.generic.show_progress)
    while true
        yield()
        for _ in 1:algo.generic.check_every
            step!(state, milp)
            next!(progress; showvalues = prog_showvalues(state))
        end
        if termination_check!(state, milp, algo)
            break
        end
    end
    finish!(progress)
    return state
end

function step!(
        state::PDHGState,
        milp::AbstractProgram,
    )
    state.sol, state.sol_last = state.sol_last, state.sol
    (; sol, sol_last, step_sizes, step_size_params, scratch) = state
    (; x, y) = sol_last
    (; lv, uv, A, At, lc, uc) = milp

    η, ω = update_step_size!(step_sizes, milp, step_size_params)

    τ, σ = η / ω, η * ω

    # xp = proj_box.(x - τ * (grad - At * y), lv, uv)
    g = grad_x(scratch, x, milp)
    At_y = mul!(scratch.x, At, y)
    @. sol.x = proj_box(x - τ * (g - At_y), lv, uv)
    xdiff = @. scratch.x = 2sol.x - x

    # yp = y - σ * A * (2xp - x) - σ * proj_box.(inv(σ) * y - A * (2xp - x), -uc, -lc)
    A_xdiff = mul!(scratch.y, A, xdiff)
    @. sol.y = y - σ * A_xdiff - σ * proj_box(inv(σ) * y - A_xdiff, -uc, -lc)

    # other updates
    state.stats.kkt_passes += 1
    return nothing
end
