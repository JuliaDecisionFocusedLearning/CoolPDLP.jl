"""
    PDLP(args...; kwargs...)

Shortcut for [`Algorithm{:PDLP}`](@ref).

The preconditioner starts with `curtis_reid_iter = 20` Curtis-Reid iterations, as in HPR-LP,
before the Ruiz passes and the Pock-Chambolle step. Pass `curtis_reid_iter = 0` to go back to
the Ruiz and Pock-Chambolle scaling of the original PDLP (and of `Algorithm{:PDLP}` itself).
"""
function PDLP(args...; kwargs...)
    return Algorithm{:PDLP}(
        args...;
        curtis_reid_iter = 20,
        kwargs...
    )
end

"""
    PDLPState

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct PDLPState{
        T <: Number,
        V <: AbstractVecOrMat{T},
        SS <: StepSizes,
        Sc <: Scratch,
        It <: IterationCounter,
        RS <: RestartStats,
        CS <: ConvergenceStats,
        IW <: Union{Nothing, InfeasibilityWorkspace},
    } <: AbstractState{T, V}
    "current solution"
    sol::PrimalDualSolution{T, V}
    "solution before the last step, saved by [`save_last!`](@ref) for the restart check"
    sol_last::PrimalDualSolution{T, V}
    "current average solution"
    sol_avg::PrimalDualSolution{T, V}
    "average solution before the last step, saved by [`save_last!`](@ref) for the restart check"
    sol_avg_last::PrimalDualSolution{T, V}
    "solution from last restart"
    sol_restart::PrimalDualSolution{T, V}
    "step sizes"
    step_sizes::SS
    "scratch space"
    scratch::Sc
    "iteration counter"
    iteration::It
    "restart stats"
    restart_stats::RS
    "convergence stats"
    stats::CS
    "infeasibility detection buffers"
    infeas::IW
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
        instance(state.infeas, i),
    )
end

function initialize(
        milp::MILP{T},
        sol::PrimalDualSolution{T, V},
        algo::Algorithm{:PDLP};
        starting_time::Float64
    ) where {T, V}
    sol_last = zero(sol)
    sol_avg = copy(sol)
    sol_avg_last = zero(sol)
    sol_restart = copy(sol)
    # after bound and objective rescaling, both norms are below one and cuPDLPx starts from ω = 1
    ω_init = if algo.preconditioning.bound_objective_rescaling
        one(T)
    else
        primal_weight_init(milp, algo.step_size)
    end
    ω = batched_expand(sol.x, ω_init)
    η, norm_A, norm_Q = initial_stepsize(milp, ω, algo.step_size)
    step_sizes = StepSizes(; η, ω, norm_A, norm_Q)
    scratch = Scratch(sol)
    iteration = IterationCounter(0, 0, 0)
    restart_stats = RestartStats(sol)
    # The error history is seeded with the starting point, and `restart_check!` reads the errors
    # of the restart point without recomputing them, so both must be filled already.
    err = KKTErrors(sol)
    kkt_errors!(err, scratch, sol, milp, :L2)
    # The restart errors stay in the L2 norm whatever `termination_norm` is: the restart
    # criteria (sufficient and necessary decay in `should_restart`) and the choice between the
    # current and averaged candidates compare these errors over time, with PDLP's default decay
    # factors, which assume L2. Following the termination norm would let a change of the
    # stopping test alone (e.g. to L∞) change the restart schedule, and therefore the iterates.
    restart_stats.err_restart = copy(err)
    norm = algo.generic.termination_norm
    if norm !== :L2
        kkt_errors!(err, scratch, sol, milp, norm)
    end
    tol = algo.termination.infeasibility_tol
    # each evaluation of the initial KKT errors costs one product by `A` and one by `Aᵀ`
    extra_products = norm === :L2 ? 2 : 4
    stats = ConvergenceStats(err; ray = ray_buffer(sol, tol), starting_time, extra_products)
    infeas = InfeasibilityWorkspace(sol, tol)
    state = PDLPState(;
        sol, sol_last, sol_avg, sol_avg_last, sol_restart,
        step_sizes, scratch, iteration, restart_stats, stats, infeas
    )
    return state
end

function solve!(
        state::PDLPState,
        milp::MILP,
        algo::Algorithm{:PDLP}
    )
    prog = init_progress("PDLP iterations:", algo.generic.show_progress)
    must_terminate = false
    @trace while !must_terminate
        @trace for _ in 1:(algo.generic.check_every - 1)
            step!(state, milp, algo)
            next_progress!(prog, state)
        end
        # the restart check compares with the iterates before the last step, so only that step
        # needs them saved
        save_last!(state)
        step!(state, milp, algo)
        next_progress!(prog, state)
        must_terminate = termination_check!(state, milp, algo)
        @trace if !must_terminate & restart_check!(state, milp, algo)
            restart!(state, milp, algo)
        end
    end
    export_certificate!(state.stats, state.infeas, milp)
    finish_progress!(prog)
    return state
end

"""
    ray_candidates(state)

Return the candidate directions for the infeasibility certificates, as `(to, from)` pairs
standing for `to - from` (or `to` itself if `from` is `nothing`).

For PDLP, these are the last step and the displacement since the last restart.
"""
ray_candidates(state::PDLPState) = ((state.sol, state.sol_last), (state.sol, state.sol_restart))

function step!(
        state::PDLPState{T, V},
        milp::MILP{T},
    ) where {T, V}
    (; sol, step_sizes, scratch) = state
    pdhg_update!(sol, scratch, step_sizes, milp, active_mask(state.stats))
    state.stats.kkt_passes += 1
    update_average!(state)
    add_inner!(state.iteration)
    return nothing
end

"""
    step!(state::PDLPState, milp, algo)

Apply one PDLP step, with the momentum of PDQP if `algo.generic.quadratic_momentum` is set and
`milp` is a quadratic program (see [`momentum_update!`](@ref)).

`isquadratic` is decided by the type of `milp.Q` for a [`ZeroMatrix`](@ref), and by its number of
stored entries for a sparse `Q`, a run-time check on the host. Under Reactant that number is
baked into the traced sparse wrapper, so the branch is still resolved at trace time.
"""
function step!(state::PDLPState, milp::MILP, algo::Algorithm{:PDLP})
    # the plain step is spelled out rather than delegated to `step!(state, milp)`: that call
    # boxed `milp` at every KKT pass
    if isquadratic(milp.Q) && algo.generic.quadratic_momentum
        momentum_update!(state, milp, algo.step_size)
        state.stats.kkt_passes += 1
        momentum_average!(state)
    else
        pdhg_update!(state.sol, state.scratch, state.step_sizes, milp, active_mask(state.stats))
        state.stats.kkt_passes += 1
        update_average!(state)
    end
    add_inner!(state.iteration)
    return nothing
end

"""
    momentum_update!(state::PDLPState, milp, params)

Apply one step of PDQP (Lu & Yang, 2023, `solver_gpu.jl` of PDQP.jl), with `k` the number of
steps since the last restart and `k' = k + 2`:

- the objective is linearized at the momentum point `x_md = (1 - β) x̄ + β x` with `β = 2 / k'`,
  where `x̄` is the aggregate iterate of PDQP, updated by [`momentum_average!`](@ref);
- the dual step extrapolates with `θ = k / (k + 1)` instead of `1`;
- the step size grows from that of [`quadratic_stepsize`](@ref) towards `1 / ‖A‖` as
  `η = k' / (‖Q‖/ω + √(k'² ‖A‖² + ‖Q‖²/ω²))`, scaled by `invnorm_scaling`, since the Lipschitz
  constant of the linearized objective is effectively `β ‖Q‖`.

The product by `Q` is taken at `x_md` instead of `x`, so a step still costs one product by each of
`A`, `Aᵀ` and `Q`.
"""
function momentum_update!(state::PDLPState{T}, milp::MILP, params::StepSizeParameters) where {T}
    (; sol, sol_avg, step_sizes, scratch, iteration) = state
    (; x, y) = sol
    (; c, lv, uv, A, At, Q, lc, uc) = milp
    k = iteration.inner
    kp = convert(T, k + 2)
    β = 2 / kp
    θ = convert(T, k) / (k + 1)
    (; norm_A, norm_Q) = step_sizes
    scaling = params.invnorm_scaling
    step_sizes.η = batched_apply!!(step_sizes.η, step_sizes.ω) do ω
        q = norm_Q / ω
        denominator = q + sqrt(kp^2 * norm_A^2 + q^2)
        ifelse(iszero(denominator), one(ω), scaling * kp / denominator)
    end
    (; η, ω) = step_sizes
    τ = transpose(primal_dual_steps!!(/, scratch.b1, η, ω))
    σ = transpose(primal_dual_steps!!(*, scratch.b2, η, ω))

    # xp = clamp.(x - τ * (c + Q * x_md - At * y), lv, uv)
    x_md = @. scratch.x = (1 - β) * sol_avg.x + β * x
    At_y = mul!(scratch.z, At, y)
    At_y = subtract_quadratic_gradient!(At_y, Q, x_md)
    xp = @. scratch.x = clamp(x - τ * (c - At_y), lv, uv)
    xdiff = @. scratch.z = xp + θ * (xp - x)
    active = active_mask(state.stats)
    masked_copyto!(x, active, xp)

    A_xdiff = mul!(scratch.y, A, xdiff)
    masked_dual_update!(y, active, y, σ, A_xdiff, lc, uc)
    return sol
end

"""
    momentum_average!(state::PDLPState)

Update the aggregate iterate of PDQP, `x̄ ← (1 - β) x̄ + β x⁺` (and the same for `y`) with
`β = 2 / (k + 2)`, in place of the step-size-weighted average of PDLP: with the momentum of
[`momentum_update!`](@ref), `state.sol_avg` holds this aggregate, which is both the momentum
point's anchor and the averaged candidate of the restarts and of the output.
"""
function momentum_average!(state::PDLPState{T}) where {T}
    (; sol, sol_avg, step_sizes, iteration) = state
    β = 2 / convert(T, iteration.inner + 2)
    active = active_mask(state.stats)
    # `β` is traced under Reactant: an argument of the broadcast rather than a captured value
    f(β, new, agg) = β * new + (1 - β) * agg
    masked_broadcast!(f, sol_avg.x, active, β, sol.x, sol_avg.x)
    masked_broadcast!(f, sol_avg.y, active, β, sol.y, sol_avg.y)
    add_stepsize!(step_sizes, active)
    return nothing
end

function update_average!(state::PDLPState)
    (; sol, sol_avg, step_sizes, scratch) = state
    (; η, η_sum) = step_sizes
    active = active_mask(state.stats)
    weight_new = broadcast!!((a, b) -> a / (a + b), scratch.b1, η, η_sum)
    weight_avg = broadcast!!((a, b) -> b / (a + b), scratch.b2, η, η_sum)
    masked_axpby!(weight_new, sol, weight_avg, sol_avg, active)
    add_stepsize!(step_sizes, active)
    return nothing
end

"""
    save_last!(state::PDLPState)

Save the current and average iterates of `state` into `sol_last` and `sol_avg_last`, which
[`restart_check!`](@ref) compares with.

Since `step!` updates the iterates in place, this must be called before the last step preceding
a restart check. Calling it before every step instead would copy both iterates each time.
"""
function save_last!(state::PDLPState)
    copy!(state.sol_last, state.sol)
    copy!(state.sol_avg_last, state.sol_avg)
    return nothing
end

"""
    restart_check!(state, milp, algo)

Refresh the restart statistics of `state` and decide whether the next iteration should restart.

Pick the better of the current and averaged iterates as the restart candidate, record which one
it was (column by column) in `state.restart_stats.restart_from_avg`, and return the decision of
[`should_restart`](@ref): a single boolean for the whole batch, which is what [`restart!`](@ref)
is guarded by.
"""
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
    # a frozen instance never moves to its average
    restart_stats.restart_from_avg = mask_and!!(
        broadcast!!(<=, restart_stats.restart_from_avg, abs2, abs1), active_mask(state.stats),
    )
    restart_stats.abs_candidate_last, _, _ = best_error!!(
        restart_stats.abs_candidate_last, err_last, err_avg_last,
        scratch, sol_last, sol_avg_last, milp, ω,
    )

    # `err_restart` stays valid between restarts, only the primal weight `ω` moves
    restart_stats.abs_restart = absolute!!(restart_stats.abs_restart, err_restart, ω)
    # the four KKT errors of `best_error!!`, two products each
    state.stats.extra_products += 8

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
    kkt_errors!(err1, scratch, sol1, milp, :L2)
    kkt_errors!(err2, scratch, sol2, milp, :L2)
    abs1 = absolute!!(scratch.b1, err1, ω)
    abs2 = absolute!!(scratch.b2, err2, ω)
    return broadcast!!(min, abs_err, abs1, abs2), abs1, abs2
end

"""
    restart!(state, milp, algo)

Restart the PDLP iterations from the candidate selected by [`restart_check!`](@ref).

For a quadratic program, the step size follows the new primal weight (see [`refresh_stepsize!`](@ref)).
"""
function restart!(state::PDLPState{T}, milp::MILP, algo::Algorithm{:PDLP}) where {T}
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
    refresh_stepsize!(step_sizes, milp.Q, algo.step_size)
    # update solutions
    zero!(sol_avg)
    copy!(sol_restart, sol)
    # update counters
    add_outer!(iteration)
    return nothing
end
