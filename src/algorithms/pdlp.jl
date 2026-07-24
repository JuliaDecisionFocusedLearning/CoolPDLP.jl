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
    PDLPState

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct PDLPState{
        T <: Number, V <: StridedVecOrMat{T}, S <: BatchedNumber, B,
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
    step_sizes::StepSizes{S}
    "scratch space"
    scratch::Scratch{T, V, S}
    "iteration counter"
    iteration::IterationCounter
    "restart stats"
    restart_stats::RestartStats{S, B}
    "convergence stats"
    stats::ConvergenceStats{S}
end

batch_size((; sol)::PDLPState) = batch_size(sol)
function batch(state::PDLPState, i::Int)
    return PDLPState(
        batch(state.sol, i),
        batch(state.sol_last, i),
        batch(state.sol_avg, i),
        batch(state.sol_avg_last, i),
        batch(state.sol_restart, i),
        batch(state.step_sizes, i),
        batch(state.scratch, i),
        state.iteration,
        batch(state.restart_stats, i),
        batch(state.stats, i),
    )
end

function initialize(
        milp::MILP{T},
        sol::PrimalDualSolution{T, V},
        algo::Algorithm{:PDLP, T};
        starting_time::Float64
    ) where {T, V}
    sol_last = zero(sol)
    sol_avg = copy(sol)
    sol_avg_last = zero(sol)
    sol_restart = copy(sol)
    η = batch_expand(sol.x, fixed_stepsize(milp, algo.step_size))
    ω = batch_expand(sol.x, primal_weight_init(milp, algo.step_size))
    step_sizes = StepSizes(; η, ω)
    scratch = Scratch(sol)
    iteration = IterationCounter(0, 0, 0)
    restart_stats = RestartStats(sol)
    stats = ConvergenceStats(KKTErrors(sol); starting_time)
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
        yield()
        for _ in 1:algo.generic.check_every
            step!(state, milp)
            next!(prog; showvalues = () -> prog_showvalues(state))
        end
        if termination_check!(state, milp, algo)
            break
        elseif restart_check!(state, milp, algo)
            restart!(state, algo)
        end
    end
    finish!(prog)
    return state
end

function step!(
        state::PDLPState{T, V},
        milp::MILP{T},
    ) where {T, V}
    # switch pointers
    state.sol, state.sol_last = state.sol_last, state.sol

    (; sol, sol_last, step_sizes, scratch) = state
    (; x, y) = sol_last
    (; η, ω) = step_sizes
    (; c, lv, uv, A, At, lc, uc) = milp

    τ = rowvec(batch_apply!(/, scratch.b1, η, ω))
    σ = rowvec(batch_apply!(*, scratch.b2, η, ω))

    # xp = clamp.(x - τ * (c - At * y), lv, uv)
    At_y = mul!(scratch.x, At, y)
    @. sol.x = clamp(x - τ * (c - At_y), lv, uv)
    xdiff = @. scratch.x = 2sol.x - x

    # yp = y - σ * A * (2xp - x) - σ * clamp.(inv(σ) * y - A * (2xp - x), -uc, -lc)
    A_xdiff = mul!(scratch.y, A, xdiff)
    @. sol.y = y - σ * A_xdiff - σ * clamp(inv(σ) * y - A_xdiff, -uc, -lc)

    # other updates
    state.stats.kkt_passes += 1
    update_average!(state)
    add_inner!(state.iteration)
    return nothing
end

function update_average!(state::PDLPState)
    (; sol, sol_avg, sol_avg_last, step_sizes, scratch) = state
    (; η, η_sum) = step_sizes
    copy!(sol_avg_last, sol_avg)
    weight_new = batch_apply!((a, b) -> a / (a + b), scratch.b1, η, η_sum)
    weight_avg = batch_apply!((a, b) -> b / (a + b), scratch.b2, η, η_sum)
    axpby!(weight_new, sol, weight_avg, sol_avg)
    add_stepsize!(step_sizes)
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
    from_avg = absolute(err_avg, ω) .<= absolute(err, ω)
    restart_stats.restart_from_avg = from_avg
    restart_stats.err_candidate = batch_select(from_avg, err_avg, err)

    err_last = kkt_errors!(scratch, sol_last, milp)
    err_avg_last = kkt_errors!(scratch, sol_avg_last, milp)
    from_avg_last = absolute(err_avg_last, ω) .<= absolute(err_last, ω)
    restart_stats.err_candidate_last = batch_select(from_avg_last, err_avg_last, err_last)

    restart_stats.err_restart = kkt_errors!(scratch, sol_restart, milp)

    return should_restart(restart_stats, step_sizes, iteration, algo.restart)
end

function restart!(state::PDLPState{T}, algo::Algorithm{:PDLP}) where {T}
    (;
        sol, sol_avg, sol_restart,
        step_sizes, iteration, scratch, restart_stats,
    ) = state

    # identify candidate, column by column
    batch_select!(sol, restart_stats.restart_from_avg, sol_avg)
    # update step sizes (must be done before losing previous restart)
    reset_stepsize!(step_sizes)
    step_sizes.ω = primal_weight_update!(
        scratch, step_sizes, sol, sol_restart, algo.step_size
    )
    # update solutions
    zero!(sol_avg)
    copy!(sol_restart, sol)
    # update counters
    add_outer!(iteration)
    return nothing
end
