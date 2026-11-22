"""
    HPDLP(args...; kwargs...)

Shortcut for [`Algorithm{:HPDLP}`](@ref): reflected restarted Halpern PDHG (r²HPDHG), as in
[cuPDLPx](https://github.com/MIT-Lu-Lab/cuPDLPx).

Its defaults follow cuPDLPx: step size `0.998 / ‖A‖₂`, reflection coefficient `1`, restarts
on the fixed-point residual with decays `0.2` (sufficient), `0.5` (necessary) and `0.36`
(artificial), PID primal weight control and a check every `200` iterations. Unlike cuPDLPx,
the preconditioner starts with `curtis_reid_iter = 20` Curtis-Reid iterations (as in HPR-LP),
which saves KKT passes on Netlib, and `bound_objective_rescaling` stays off (cuPDLPx turns it on;
here it costs about 15 % more KKT passes on a Netlib sample at `1e-6`).

The reflected step is `γ (2T(z) - z) + (1 - γ) z`, so `reflection_coefficient = 1` is the fully
reflected step of cuPDLPx and `1/2` is plain Halpern on `T` (other values may not match the
parametrization of cuPDLPx's own reflection coefficient).

Unlike [`PDLP`](@ref), every instance of a batch restarts on its own, so
`restart_batch_aggregation` is ignored.
"""
function HPDLP(args...; kwargs...)
    return Algorithm{:HPDLP}(
        args...;
        invnorm_scaling = 0.998,
        necessary_decay = 0.5,
        check_every = 200,
        curtis_reid_iter = 20,
        kwargs...
    )
end

"""
    HalpernStats

Per-instance bookkeeping of [`HPDLP`](@ref): Halpern counters, fixed-point residuals and the
state of the PID primal weight controller.

Every field but `total` holds one value per instance of the batch.

# Fields

$(TYPEDFIELDS)
"""
mutable struct HalpernStats{T <: BatchedNumber, B <: BatchedNumber, I <: Number}
    "Halpern iterations since the last restart"
    inner::T
    "iterations since the start"
    total::I
    "number of restarts so far"
    restarts::T
    "fixed-point residual `‖z - T(z)‖_P` at the last check"
    fpe::T
    "fixed-point residual of the anchor, right after the last restart"
    fpe_initial::T
    "fixed-point residual at the previous check since the last restart (`Inf` right after it)"
    fpe_last::T
    "whether `fpe_initial` must be refreshed at the next step, because of a restart"
    fpe_pending::B
    "whether the instance restarted at the last check"
    restart::B
    "discounted sum of the PID errors"
    pid_error_sum::T
    "PID error at the last restart"
    pid_last_error::T
    "PID error being computed at a restart"
    pid_error::T
    "primal weight which balanced the primal and dual residuals best so far"
    best_ω::T
    "best balance `|log(rel_dual / rel_primal)|` so far (natural log: same ordering as the `log10` of cuPDLPx)"
    best_gap::T
end

function HalpernStats(sol::PrimalDualSolution{T}, ω::BatchedNumber) where {T}
    num(v) = batched_expand(sol.x, convert(T, v))
    return HalpernStats(
        num(0), 0, num(0),
        num(Inf), num(Inf), num(Inf),
        batched_expand(sol.x, true), batched_expand(sol.x, false),
        num(0), num(0), num(0),
        copy(ω), num(Inf),
    )
end

function instance(stats::HalpernStats, i::Int)
    return HalpernStats(
        instance_num(stats.inner, i),
        stats.total,
        instance_num(stats.restarts, i),
        instance_num(stats.fpe, i),
        instance_num(stats.fpe_initial, i),
        instance_num(stats.fpe_last, i),
        instance_num(stats.fpe_pending, i),
        instance_num(stats.restart, i),
        instance_num(stats.pid_error_sum, i),
        instance_num(stats.pid_last_error, i),
        instance_num(stats.pid_error, i),
        instance_num(stats.best_ω, i),
        instance_num(stats.best_gap, i),
    )
end

"""
    HPDLPState

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct HPDLPState{
        T <: Number,
        V <: AbstractVecOrMat{T},
        SS <: StepSizes,
        Sc <: Scratch,
        HS <: HalpernStats,
        CS <: ConvergenceStats,
        IW <: Union{Nothing, InfeasibilityWorkspace},
    } <: AbstractState{T, V}
    "PDHG image `T(z)` of the Halpern iterate before the last step, which is the reported solution"
    sol::PrimalDualSolution{T, V}
    "current Halpern iterate `z`"
    z::PrimalDualSolution{T, V}
    "Halpern anchor `z⁰`, set at the last restart"
    z_anchor::PrimalDualSolution{T, V}
    "step sizes"
    step_sizes::SS
    "scratch space"
    scratch::Sc
    "Halpern and restart bookkeeping"
    halpern::HS
    "convergence stats"
    stats::CS
    "infeasibility detection buffers"
    infeas::IW
end

nbinstances((; sol)::HPDLPState) = nbinstances(sol)
function instance(state::HPDLPState, i::Int)
    return HPDLPState(
        instance(state.sol, i),
        instance(state.z, i),
        instance(state.z_anchor, i),
        instance(state.step_sizes, i),
        instance(state.scratch, i),
        instance(state.halpern, i),
        instance(state.stats, i),
        instance(state.infeas, i),
    )
end


function initialize(
        milp::MILP{T},
        sol::PrimalDualSolution{T, V},
        algo::Algorithm{:HPDLP};
        starting_time::Float64
    ) where {T, V}
    z = copy(sol)
    z_anchor = copy(sol)
    # after bound and objective rescaling, both norms are below one and cuPDLPx starts from ω = 1
    ω_init = if algo.preconditioning.bound_objective_rescaling
        one(T)
    else
        primal_weight_init(milp, algo.step_size)
    end
    ω = batched_expand(sol.x, ω_init)
    η = batched_expand(sol.x, fixed_stepsize(milp, algo.step_size))
    step_sizes = StepSizes(; η, ω)
    scratch = Scratch(sol)
    halpern = HalpernStats(sol, ω)
    # the error history is seeded with the starting point, so its errors must be filled already
    err = KKTErrors(sol)
    kkt_errors!(err, scratch, sol, milp, algo.generic.termination_norm)
    tol = algo.termination.infeasibility_tol
    stats = ConvergenceStats(
        err; ray = ray_buffer(sol, tol), starting_time,
        extra_products = 2,
    )
    infeas = InfeasibilityWorkspace(sol, tol)
    return HPDLPState(; sol, z, z_anchor, step_sizes, scratch, halpern, stats, infeas)
end

"""
    solve!(state::HPDLPState, milp, algo)

Run reflected restarted Halpern PDHG until termination.

Each block of `check_every` steps measures the fixed-point residual on its first step (to
seed the reference residual of the instances which just restarted) and on its last one (for
the restart decision), then checks termination and restarts the instances which need it.
"""
function solve!(
        state::HPDLPState,
        milp::MILP,
        algo::Algorithm{:HPDLP}
    )
    check_every = algo.generic.check_every
    check_every >= 2 || throw(ArgumentError("HPDLP needs `check_every >= 2`, got $check_every"))
    prog = init_progress("HPDLP iterations:", algo.generic.show_progress)
    must_terminate = false
    @trace while !must_terminate
        must_terminate = check_pass!(state, milp, algo, prog)
    end
    export_certificate!(state.stats, state.infeas, milp)
    finish_progress!(prog)
    return state
end

"""
    check_pass!(state::HPDLPState, milp, algo, prog = NoProgress())

Run one block of `check_every` steps, the termination check and the restarts, and return whether
to stop (see [`solve!`](@ref)).
"""
function check_pass!(
        state::HPDLPState,
        milp::MILP,
        algo::Algorithm{:HPDLP},
        prog = NoProgress()
    )
    step!(state, milp, algo, Val(true))
    seed_fixed_point_error!(state.halpern)
    next_progress!(prog, state)
    @trace for _ in 1:(algo.generic.check_every - 2)
        # with an explicit `Val`: through the default argument, the call boxed `milp`
        # and `algo` at every KKT pass
        step!(state, milp, algo, Val(false))
        next_progress!(prog, state)
    end
    step!(state, milp, algo, Val(true))
    next_progress!(prog, state)
    must_terminate = termination_check!(state, milp, algo)
    # restarts are decided and applied column by column, with masks rather than branches
    restart_check!(state, algo, must_terminate)
    restart!(state, milp, algo)
    return must_terminate
end

"""
    ray_candidates(state::HPDLPState)

For HPDLP, the candidate directions follow cuPDLPx: the PDHG image `T(z)` minus the anchor of
the instance's last restart, and `T(z)` itself. (cuPDLPx also tries `T(z) - z`, which would
need the Halpern iterate from before the last step to be kept.)
"""
ray_candidates(state::HPDLPState) = ((state.sol, state.z_anchor), (state.sol, nothing))

"""
    step!(state::HPDLPState, milp, algo, Val(fixed_point_error))

Apply one reflected Halpern step

    z ← (k+1)/(k+2) * (γ (2T(z) - z) + (1 - γ) z) + 1/(k+2) * z⁰

where `T` is the PDHG operator, `k` the number of steps since the last restart and `z⁰` the
anchor. `T(z)` is kept in `state.sol`. With `Val(true)`, the fixed-point residual
`‖z - T(z)‖_P` is also stored in `state.halpern.fpe`.
"""
function step!(
        state::HPDLPState{T},
        milp::MILP{T},
        algo::Algorithm{:HPDLP},
        ::Val{fixed_point_error},
    ) where {T, fixed_point_error}
    (; sol, z, z_anchor, step_sizes, scratch, halpern) = state
    γ = algo.step_size.reflection_coefficient
    # a frozen instance keeps `T(z)` and `z` as they were when it terminated
    active = active_mask(state.stats)
    masked_copy!(sol, active, z)
    pdhg_update!(sol, scratch, step_sizes, milp, active)
    if fixed_point_error
        halpern.fpe = fixed_point_error!!(halpern.fpe, scratch, z, sol, milp, step_sizes)
        state.stats.extra_products += 1
    end
    w = broadcast!!(k -> (k + 1) / (k + 2), scratch.b1, halpern.inner)
    halpern_combine!(z.x, w, γ, sol.x, z_anchor.x, active)
    halpern_combine!(z.y, w, γ, sol.y, z_anchor.y, active)
    halpern.inner = masked_apply!!(k -> k + 1, halpern.inner, active, halpern.inner)
    halpern.total += 1
    state.stats.kkt_passes += 1
    return nothing
end

function halpern_combine!(
        z::AbstractVecOrMat, w::BatchedNumber, γ::Number, Tz::AbstractVecOrMat, z0::AbstractVecOrMat,
        active,
    )
    wr = transpose(w)
    masked_broadcast!(z, active, wr, z, Tz, z0) do w, z, Tz, z0
        w * (γ * (2Tz - z) + (1 - γ) * z) + (1 - w) * z0
    end
    return z
end

"""
    fixed_point_error!!(dest, scratch, z, Tz, milp, step_sizes)

Compute the fixed-point residual `‖z - T(z)‖_P` in the norm of the PDHG preconditioner, scaled
by the step size `η` as in cuPDLPx, column by column, into `dest`:

    ω ‖Δx‖² + ‖Δy‖² / ω + 2η ⟨Δx, Aᵀ Δy⟩

PDHG is a proximal point step in the metric `[I/τ  Aᵀ; A  I/σ]` (with the sign convention of `y`
in this package). The square is clamped below so that rounding cannot make it negative.
"""
function fixed_point_error!!(
        dest::BatchedNumber, scratch::Scratch,
        z::PrimalDualSolution, Tz::PrimalDualSolution,
        milp::MILP, step_sizes::StepSizes,
    )
    (; At) = milp
    (; η, ω) = step_sizes
    Δx = @. scratch.x = z.x - Tz.x
    Δy = @. scratch.y = z.y - Tz.y
    At_Δy = mul!(scratch.z, At, Δy)
    Δx_norm = colnorm!!(scratch.b1, Δx)
    Δy_norm = colnorm!!(scratch.b2, Δy)
    dest = broadcast!!(dest, Δx_norm, Δy_norm, ω) do a, b, w
        w * a^2 + b^2 / w
    end
    cross = colsum!!(scratch.b1, @. At_Δy = Δx * At_Δy)
    return broadcast!!(dest, dest, cross, η) do movement, cross, η
        sqrt(max(movement + 2η * cross, zero(movement)))
    end
end

"""
    seed_fixed_point_error!(halpern)

Take the fixed-point residual just measured as the reference residual of the instances which
restarted at the last check (or never did).
"""
function seed_fixed_point_error!(halpern::HalpernStats)
    halpern.fpe_initial = broadcast!!(
        ifelse, halpern.fpe_initial, halpern.fpe_pending, halpern.fpe, halpern.fpe_initial
    )
    halpern.fpe_pending = broadcast!!(zero, halpern.fpe_pending, halpern.fpe_pending)
    return nothing
end

"""
    restart_check!(state::HPDLPState, algo, must_terminate)

Decide, instance by instance, whether to restart, following cuPDLPx: sufficient decay of the
fixed-point residual, necessary decay without local progress, or a long inner loop. The
decision is stored in `state.halpern.restart`.
"""
function restart_check!(state::HPDLPState, algo::Algorithm{:HPDLP}, must_terminate)
    return halpern_restart_check!(state.halpern, algo.restart, must_terminate, active_mask(state.stats))
end

"""
    halpern_restart_check!(halpern, params, must_terminate, active)

Body of [`restart_check!`](@ref) for the Halpern algorithms, on their bookkeeping alone. The
frozen instances of a batch (see [`active_mask`](@ref)) never restart.
"""
function halpern_restart_check!(
        halpern::HalpernStats, params::RestartParameters, must_terminate, active,
    )
    (; sufficient_decay, necessary_decay, artificial_decay) = params
    (; total) = halpern
    halpern.restart = broadcast!!(
        halpern.restart, halpern.fpe, halpern.fpe_initial, halpern.fpe_last, halpern.inner,
    ) do fpe, fpe0, fpe_last, inner
        sufficient = fpe <= sufficient_decay * fpe0
        necessary = (fpe <= necessary_decay * fpe0) & (fpe > fpe_last)
        long_inner_loop = inner >= artificial_decay * total
        !must_terminate & (sufficient | necessary | long_inner_loop)
    end
    halpern.restart = mask_and!!(halpern.restart, active)
    halpern.fpe_last = broadcast!!(
        halpern.fpe_last, halpern.restart, halpern.fpe
    ) do restart, fpe
        ifelse(restart, oftype(fpe, Inf), fpe)
    end
    return halpern.restart
end

"""
    restart!(state::HPDLPState, milp, algo)

Restart the instances selected by [`restart_check!`](@ref): update their primal weight with
the PID controller, then move both their iterate and their anchor to `T(z)`.

The other instances are left untouched, since every update is masked column by column.
"""
function restart!(state::HPDLPState{T}, milp::MILP, algo::Algorithm{:HPDLP}) where {T}
    (; sol, z, z_anchor, step_sizes, scratch, halpern, stats) = state
    (; restart) = halpern
    pid_update!(halpern, step_sizes, scratch, sol, z_anchor, stats.err, algo.step_size)
    batched_select!(z_anchor, restart, sol)
    batched_select!(z, restart, sol)
    halpern.inner = broadcast!!(halpern.inner, restart, halpern.inner) do r, k
        ifelse(r, zero(k), k)
    end
    halpern.restarts = broadcast!!(halpern.restarts, restart, halpern.restarts) do r, n
        ifelse(r, n + 1, n)
    end
    halpern.fpe_pending = broadcast!!(identity, halpern.fpe_pending, restart)
    return nothing
end

"""
    pid_update!(halpern, step_sizes, scratch, sol, z_anchor, err, params)

PID control of the primal weight in log space, as in cuPDLPx, for the instances which restart.

The error is `log(‖Δy‖ / (ω ‖Δx‖))` between the new restart point `sol` and the previous
anchor. When the distances or the ratio of relative residuals are degenerate, the weight falls
back to the one which balanced the residuals best so far and the controller is reset.
"""
function pid_update!(
        halpern::HalpernStats, step_sizes::StepSizes, scratch::Scratch,
        sol::PrimalDualSolution, z_anchor::PrimalDualSolution, err::KKTErrors,
        params::StepSizeParameters,
    )
    (; restart) = halpern
    (; pid_kp, pid_ki, pid_kd, pid_integral_discount) = params
    (; primal, primal_scale, dual, dual_scale) = err
    Δx = colnorm!!(scratch.b1, @. scratch.x = sol.x - z_anchor.x)
    Δy = colnorm!!(scratch.b2, @. scratch.y = sol.y - z_anchor.y)

    function valid(Δx, Δy, p, ps, d, ds)
        ratio = (d / ds) / (p / ps)
        return (Δx > 1.0e-16) & (Δy > 1.0e-16) & (Δx < 1.0e12) & (Δy < 1.0e12) &
            (ratio > 1.0e-8) & (ratio < 1.0e8)
    end
    # the logarithms are guarded so that a degenerate instance gives a finite, unused value
    safelog(a) = log(ifelse(a > zero(a), a, one(a)))

    halpern.pid_error = broadcast!!(
        halpern.pid_error, Δx, Δy, step_sizes.ω,
    ) do Δx, Δy, w
        safelog(Δy) - safelog(Δx) - log(w)
    end
    step_sizes.ω = broadcast!!(
        step_sizes.ω, restart, Δx, Δy, primal, primal_scale, dual, dual_scale,
        step_sizes.ω, halpern.pid_error, halpern.pid_error_sum, halpern.pid_last_error, halpern.best_ω,
    ) do r, Δx, Δy, p, ps, d, ds, w, e, s, l, best
        s_new = pid_integral_discount * s + e
        w_new = w * exp(pid_kp * e + pid_ki * s_new + pid_kd * (e - l))
        ifelse(r, ifelse(valid(Δx, Δy, p, ps, d, ds), w_new, best), w)
    end
    halpern.pid_error_sum = broadcast!!(
        halpern.pid_error_sum, restart, Δx, Δy, primal, primal_scale, dual, dual_scale,
        halpern.pid_error, halpern.pid_error_sum,
    ) do r, Δx, Δy, p, ps, d, ds, e, s
        ifelse(r, ifelse(valid(Δx, Δy, p, ps, d, ds), pid_integral_discount * s + e, zero(s)), s)
    end
    halpern.pid_last_error = broadcast!!(
        halpern.pid_last_error, restart, Δx, Δy, primal, primal_scale, dual, dual_scale,
        halpern.pid_error, halpern.pid_last_error,
    ) do r, Δx, Δy, p, ps, d, ds, e, l
        ifelse(r, ifelse(valid(Δx, Δy, p, ps, d, ds), e, zero(l)), l)
    end

    # remember the weight which balances the relative residuals best, for the fallback above;
    # as in cuPDLPx (`perform_restart` in `solver.cu`), it is recorded from the weight *after*
    # the PID update of this restart, compared with the residuals measured before it
    balance(p, ps, d, ds) = abs(log((d / ds) / (p / ps)))
    halpern.best_ω = broadcast!!(
        halpern.best_ω, restart, primal, primal_scale, dual, dual_scale,
        halpern.best_gap, step_sizes.ω, halpern.best_ω,
    ) do r, p, ps, d, ds, gap, w, best
        ifelse(r & (balance(p, ps, d, ds) < gap), w, best)
    end
    halpern.best_gap = broadcast!!(
        halpern.best_gap, restart, primal, primal_scale, dual, dual_scale, halpern.best_gap,
    ) do r, p, ps, d, ds, gap
        ifelse(r & (balance(p, ps, d, ds) < gap), balance(p, ps, d, ds), gap)
    end
    return nothing
end
