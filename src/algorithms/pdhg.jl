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
        L <: Union{Nothing, PrimalDualSolution{T, V}}, IW <: Union{Nothing, InfeasibilityWorkspace},
    } <: AbstractState{T, V}
    "current solution"
    sol::PrimalDualSolution{T, V}
    "solution before the last step, saved by [`save_last!`](@ref) for the infeasibility check (`nothing` when it is disabled)"
    sol_last::L
    "step sizes"
    step_sizes::SS
    "scratch space"
    scratch::Sc
    "convergence stats"
    stats::CS
    "infeasibility detection buffers"
    infeas::IW
end

nbinstances((; sol)::PDHGState) = nbinstances(sol)
function instance(state::PDHGState, i::Int)
    return PDHGState(
        instance(state.sol, i),
        instance(state.sol_last, i),
        instance(state.step_sizes, i),
        instance(state.scratch, i),
        instance(state.stats, i),
        instance(state.infeas, i),
    )
end

function initialize(
        milp::MILP{T},
        sol::PrimalDualSolution{T, V},
        algo::Algorithm{:PDHG};
        starting_time::Float64
    ) where {T, V}
    η = batched_expand(sol.x, fixed_stepsize(milp, algo.step_size))
    ω = batched_expand(sol.x, one(T))
    step_sizes = StepSizes(; η, ω)
    scratch = Scratch(sol)
    # the error history is seeded with the starting point, so its errors must be filled already
    err = KKTErrors(sol)
    kkt_errors!(err, scratch, sol, milp, algo.generic.termination_norm)  # TODO: count this KKT pass
    tol = algo.termination.infeasibility_tol
    stats = ConvergenceStats(err; ray = ray_buffer(sol, tol), starting_time)
    infeas = InfeasibilityWorkspace(sol, tol)
    # only the infeasibility check needs the previous iterate
    sol_last = ray_buffer(sol, tol)
    state = PDHGState(; sol, sol_last, step_sizes, scratch, stats, infeas)
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
        @trace for _ in 1:(algo.generic.check_every - 1)
            step!(state, milp)
            next_progress!(prog, state)
        end
        # the infeasibility check compares with the iterate before the last step, so only that
        # step needs it saved
        save_last!(state)
        step!(state, milp)
        next_progress!(prog, state)
        must_terminate = termination_check!(state, milp, algo)
    end
    export_certificate!(state.stats, state.infeas, milp)
    finish_progress!(prog)
    return state
end

"""
    save_last!(state::PDHGState)

Save the current iterate of `state` into `sol_last`, which [`infeasibility_check!`](@ref)
compares with. Does nothing when the infeasibility detection is disabled.
"""
function save_last!(state::PDHGState)
    save_last!(state.sol_last, state.sol)
    return nothing
end

save_last!(sol_last::PrimalDualSolution, sol::PrimalDualSolution) = (copy!(sol_last, sol); nothing)
save_last!(::Nothing, ::PrimalDualSolution) = nothing

"""
    ray_candidates(state)

For PDHG, the candidate directions are the last step and the current iterate itself (the
displacement since the starting point, up to a constant).
"""
ray_candidates(state::PDHGState) = ((state.sol, state.sol_last), (state.sol, nothing))

function step!(
        state::PDHGState{T, V},
        milp::MILP{T},
    ) where {T, V}
    (; sol, step_sizes, scratch) = state
    pdhg_update!(sol, scratch, step_sizes, milp, active_mask(state.stats))
    state.stats.kkt_passes += 1
    return nothing
end

"""
    dual_update(y, σ, Ax, lc, uc)

Dual PDHG step `y - σ Ax - σ clamp(y / σ - Ax, -uc, -lc)` of one entry.
"""
@inline dual_update(y, σ, Ax, lc, uc) = y - σ * Ax - σ * clamp(inv(σ) * y - Ax, -uc, -lc)

"""
    pdhg_update!(sol, scratch, step_sizes, milp, active)

Apply one PDHG update to the iterates `sol`, in place, for the instances flagged in `active`
only (see [`active_mask`](@ref)).

The new primal iterate is formed in scratch space, used for the extrapolation `2xp - x`, then
copied into `x`. Writing it into a second buffer and swapping the two would avoid that copy in
plain Julia, but under Reactant both buffers would be carried through the compiled loop and trade
places at each step, so XLA would copy them in full every time.
"""
function pdhg_update!(
        sol::PrimalDualSolution, scratch::Scratch, step_sizes::StepSizes, milp::MILP,
        active,
    )
    (; x, y) = sol
    (; η, ω) = step_sizes
    (; c, lv, uv, A, At, lc, uc) = milp

    τ = transpose(batched_apply!!(/, scratch.b1, η, ω))
    σ = transpose(batched_apply!!(*, scratch.b2, η, ω))

    # xp = clamp.(x - τ * (c - At * y), lv, uv)
    At_y = mul!(scratch.z, At, y)
    xp = @. scratch.x = clamp(x - τ * (c - At_y), lv, uv)
    xdiff = @. scratch.z = 2xp - x
    masked_copyto!(x, active, xp)

    # yp = y - σ * A * (2xp - x) - σ * clamp.(inv(σ) * y - A * (2xp - x), -uc, -lc)
    A_xdiff = mul!(scratch.y, A, xdiff)
    masked_broadcast!(dual_update, y, active, y, σ, A_xdiff, lc, uc)
    return sol
end
