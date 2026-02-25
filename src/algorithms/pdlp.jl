"""
    PDLP(args...; kwargs...)

Shortcut for [`Algorithm{:PDLP}`](@ref).
"""
function PDLP(args...; kwargs...)
    return Algorithm{:PDLP}(
        args...;
        kwargs...
    )
end

"""
    PDHGState

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct PDLPState{
        T <: Number, V <: DenseVector{T},
    } <: AbstractState{T, V}
    "current solution"
    sol::PrimalDualSolution{T, V}
    "last solution"
    sol_last::PrimalDualSolution{T, V}
    "current average solution"
    sol_avg::PrimalDualSolution{T, V}
    "last average solution"
    sol_avg_last::PrimalDualSolution{T, V}
    "solution from last restart"
    sol_restart::PrimalDualSolution{T, V}
    "step sizes"
    step_sizes::StepSizes{T}
    "scratch space"
    scratch::Scratch{T, V}
    "iteration counter"
    iteration::IterationCounter
    "restart stats"
    restart_stats::RestartStats{T}
    "convergence stats"
    stats::ConvergenceStats{T}
end

function initialize(
        milp::MILP{T, V},
        sol::PrimalDualSolution{T, V},
        algo::Algorithm{:PDLP, T};
        starting_time::Float64
    ) where {T, V}
    sol_last = zero(sol)
    sol_avg = copy(sol)
    sol_avg_last = zero(sol)
    sol_restart = copy(sol)
    η = fixed_stepsize(milp, algo.step_size)
    ω = primal_weight_init(milp, algo.step_size)
    step_sizes = StepSizes(; η, ω)
    scratch = Scratch(sol)
    iteration = IterationCounter(0, 0, 0)
    restart_stats = RestartStats(T)
    stats = ConvergenceStats(T; starting_time)
    state = PDLPState(;
        sol, sol_last, sol_avg, sol_avg_last, sol_restart,
        step_sizes, scratch, iteration, restart_stats, stats
    )
    return state
end

function solve!(
        state::PDLPState,
        milp::MILP,
        algo::Algorithm{:PDLP}
    )
    prog = ProgressUnknown(desc = "PDLP iterations:", enabled = algo.generic.show_progress)
    while true
        while true
            yield()
            for _ in 1:algo.generic.check_every
                step!(state, milp)
                next!(prog; showvalues = prog_showvalues(state))
            end
            if termination_check!(state, milp, algo)
                break
            elseif restart_check!(state, milp, algo)
                restart!(state, algo)
            end
        end
        if termination_check!(state, milp, algo)
            break
        end
    end
    finish!(prog)
    return state
end

function step!(
        state::PDLPState{T, V},
        milp::MILP{T, V},
    ) where {T, V}
    # switch pointers
    state.sol, state.sol_last = state.sol_last, state.sol

    (; sol, sol_last, step_sizes, scratch) = state
    (; x, y) = sol_last
    (; η, ω) = step_sizes
    (; c, lv, uv, A, At, lc, uc) = milp

    τ, σ = η / ω, η * ω

    # xp = proj_box.(x - τ * (c - At * y), lv, uv)
    At_y = mul!(scratch.x, At, y)
    @. sol.x = proj_box(x - τ * (c - At_y), lv, uv)
    xdiff = @. scratch.x = 2sol.x - x

    # yp = y - σ * A * (2xp - x) - σ * proj_box.(inv(σ) * y - A * (2xp - x), -uc, -lc)
    A_xdiff = mul!(scratch.y, A, xdiff)
    @. sol.y = y - σ * A_xdiff - σ * proj_box(inv(σ) * y - A_xdiff, -uc, -lc)

    # other updates
    state.stats.kkt_passes += 1
    update_average!(state)
    add_inner!(state.iteration)
    return nothing
end

function update_average!(state::PDLPState)
    (; sol, sol_avg, sol_avg_last, step_sizes) = state
    (; η, η_sum) = step_sizes
    copy!(sol_avg_last, sol_avg)
    axpby!(
        η / (η + η_sum), sol,
        η_sum / (η + η_sum), sol_avg
    )
    step_sizes.η_sum += η
    return nothing
end

function restart_check!(
        state::PDLPState,
        milp::MILP,
        algo::Algorithm{:PDLP}
    )
    (;
        sol, sol_last, sol_avg, sol_avg_last, sol_restart,
        step_sizes, scratch, iteration, restart_stats,
    ) = state
    (; ω) = step_sizes

    err = kkt_errors!(scratch, sol, milp)
    err_avg = kkt_errors!(scratch, sol_avg, milp)
    if absolute(err, ω) < absolute(err_avg, ω)
        restart_stats.restart_from_avg = false
        restart_stats.err_candidate = err
    else
        restart_stats.restart_from_avg = true
        restart_stats.err_candidate = err_avg
    end

    err_last = kkt_errors!(scratch, sol_last, milp)
    err_avg_last = kkt_errors!(scratch, sol_avg_last, milp)
    if absolute(err_last, ω) < absolute(err_avg_last, ω)
        restart_stats.err_candidate_last = err_last
    else
        restart_stats.err_candidate_last = err_avg_last
    end

    restart_stats.err_restart = kkt_errors!(scratch, sol_restart, milp)

    return should_restart(restart_stats, step_sizes, iteration, algo.restart)
end

function restart!(state::PDLPState{T}, algo::Algorithm{:PDLP}) where {T}
    (;
        sol, sol_avg, sol_restart,
        step_sizes, iteration, scratch, restart_stats,
    ) = state

    # identify candidate
    if restart_stats.restart_from_avg
        sol_cand = sol_avg
    else
        sol_cand = sol
    end
    # update step sizes (must be done before losing previous restart)
    step_sizes.η_sum = zero(T)
    step_sizes.ω = primal_weight_update!(
        scratch, step_sizes, sol_cand, sol_restart, algo.step_size
    )
    # update solutions
    copy!(sol, sol_cand)
    zero!(sol_avg)
    copy!(sol_restart, sol)
    # update counters
    add_outer!(iteration)
    return nothing
end
