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
        T <: Number, V <: AbstractVecOrMat{T}, S <: BatchedNumber,
        B <: BatchedNumber, Sc <: Scratch{T, V, S},
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
    scratch::Sc
    "iteration counter"
    iteration::IterationCounter
    "restart stats"
    restart_stats::RestartStats{S, B}
    "convergence stats"
    stats::ConvergenceStats{S}
end

nbinstances((; sol)::PDLPState) = nbinstances(sol)
function instance(state::PDLPState, i::Int)
    return PDLPState(
        instance(state.sol, i),
        instance(state.sol_last, i),
        instance(state.sol_avg, i),
        instance(state.sol_avg_last, i),
        instance(state.sol_restart, i),
        instance(state.step_sizes, i),
        instance(state.scratch, i),
        state.iteration,
        instance(state.restart_stats, i),
        instance(state.stats, i),
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
    η = batched_expand(sol.x, fixed_stepsize(milp, algo.step_size))
    ω = batched_expand(sol.x, primal_weight_init(milp, algo.step_size))
    step_sizes = StepSizes(; η, ω)
    scratch = Scratch(sol)
    iteration = IterationCounter(0, 0, 0)
    restart_stats = RestartStats(sol)
    stats = ConvergenceStats(KKTErrors(sol); starting_time)
    state = PDLPState(;
        sol, sol_last, sol_avg, sol_avg_last, sol_restart,
        step_sizes, scratch, iteration, restart_stats, stats
    )
    # `restart_check!` reads the errors of the restart point without recomputing them, so
    # they must be filled for the starting point already
    kkt_errors!(restart_stats.err_restart, scratch, sol, milp)
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
    update_average!(state)
    add_inner!(state.iteration)
    return nothing
end

function update_average!(state::PDLPState)
    (; sol, sol_avg, sol_avg_last, step_sizes, scratch) = state
    (; η, η_sum) = step_sizes
    copy!(sol_avg_last, sol_avg)
    weight_new = broadcast!!((a, b) -> a / (a + b), scratch.b1, η, η_sum)
    weight_avg = broadcast!!((a, b) -> b / (a + b), scratch.b2, η, η_sum)
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
        sol, sol_last, sol_avg, sol_avg_last,
        step_sizes, scratch, iteration, restart_stats,
    ) = state
    (; ω) = step_sizes
    (; err_current, err_avg, err_last, err_avg_last, err_restart) = restart_stats

    restart_stats.abs_candidate, abs1, abs2 = best_error!!(
        restart_stats.abs_candidate, err_current, err_avg, scratch, sol, sol_avg, milp, ω,
    )
    restart_stats.restart_from_avg = broadcast!!(
        <=, restart_stats.restart_from_avg, abs2, abs1,
    )
    restart_stats.abs_candidate_last, _, _ = best_error!!(
        restart_stats.abs_candidate_last, err_last, err_avg_last,
        scratch, sol_last, sol_avg_last, milp, ω,
    )

    # `err_restart` stays valid between restarts, only the primal weight `ω` moves
    restart_stats.abs_restart = absolute!!(restart_stats.abs_restart, err_restart, ω)

    return should_restart(restart_stats, iteration, algo.restart)
end

"""
    best_error!!(abs_err, err1, err2, scratch, sol1, sol2, milp, ω)

Fill `err1` and `err2` with the KKT errors of `sol1` and `sol2`, then keep the smaller of their absolute errors, column by column, in `abs_err`.

Return `abs_err` together with both absolute errors, which live in the scratch space and stay valid only until the next call.
"""
function best_error!!(
        abs_err::BatchedNumber, err1::KKTErrors, err2::KKTErrors,
        scratch::Scratch, sol1::PrimalDualSolution, sol2::PrimalDualSolution,
        milp::MILP, ω::BatchedNumber,
    )
    # `kkt_errors!` writes into `scratch.b1` and `b2`, so both errors come first
    kkt_errors!(err1, scratch, sol1, milp)
    kkt_errors!(err2, scratch, sol2, milp)
    abs1 = absolute!!(scratch.b1, err1, ω)
    abs2 = absolute!!(scratch.b2, err2, ω)
    return broadcast!!(min, abs_err, abs1, abs2), abs1, abs2
end

function restart!(state::PDLPState{T}, algo::Algorithm{:PDLP}) where {T}
    (;
        sol, sol_avg, sol_restart,
        step_sizes, iteration, scratch, restart_stats,
    ) = state

    # identify candidate, column by column
    batched_select!(sol, restart_stats.restart_from_avg, sol_avg)
    # the restart point is the candidate just selected, so its errors are already known
    select_errors!!(
        restart_stats.err_restart, restart_stats.restart_from_avg,
        restart_stats.err_avg, restart_stats.err_current,
    )
    # update step sizes (must be done before losing previous restart)
    reset_stepsize!(step_sizes)
    step_sizes.ω = primal_weight_update!!(
        scratch, step_sizes, sol, sol_restart, algo.step_size
    )
    # update solutions
    zero!(sol_avg)
    copy!(sol_restart, sol)
    # update counters
    add_outer!(iteration)
    return nothing
end
