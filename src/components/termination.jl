"""
    current_time()

Return the current time in seconds, like `Base.time()`.

This indirection exists for Reactant. `time()` is an ordinary Julia call, so tracing it folds
its trace-time value into the compiled program as a constant: the elapsed time would never
advance inside a compiled solve loop and the time limit could never fire. The `Reactant`
extension overlays this method with a host callback, which is re-evaluated at every iteration
of the compiled loop.
"""
current_time() = time()

"""
    update_time_elapsed!(stats, time_limit)

Set `stats.time_elapsed` to the time elapsed since `stats.starting_time`.

Inside a Reactant compilation context, do nothing when `time_limit` is infinite. Reading the
clock there costs a host callback at every termination check, and the Reactant extension
traces the time limit (in [`GenericParameters`](@ref)) as a constant, so without one the
callback is never emitted. The elapsed time of such a compiled solve is then left at its
initial value.
"""
function update_time_elapsed!(stats, time_limit)
    within_compile() && isinf(time_limit) && return nothing
    stats.time_elapsed = current_time() - stats.starting_time
    return nothing
end

"""
    DEFAULT_INFEASIBILITY_TOL

Default `infeasibility_tol` of [`TerminationParameters`](@ref) and [`Algorithm`](@ref).
"""
const DEFAULT_INFEASIBILITY_TOL = 1.0e-8

"""
    TerminationParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct TerminationParameters{T <: Number, I <: Number, E <: Union{Nothing, Number}}
    "relative tolerance on the KKT errors to decide termination, see [`termination_margin!!`](@ref)"
    termination_reltol::T
    "absolute tolerance on the KKT errors to decide termination, see [`termination_margin!!`](@ref)"
    termination_abstol::T = termination_reltol
    "maximum number of multiplications by both the KKT matrix `K` and its transpose `Kᵀ`"
    max_kkt_passes::I
    "relative tolerance for infeasibility certificates, see [`infeasibility_check!`](@ref); `nothing` disables the detection"
    infeasibility_tol::E = oftype(termination_reltol, DEFAULT_INFEASIBILITY_TOL)
end

function Base.show(io::IO, params::TerminationParameters)
    (; termination_reltol, termination_abstol, max_kkt_passes, infeasibility_tol) = params
    return print(io, "TerminationParameters: termination_reltol=$termination_reltol, termination_abstol=$termination_abstol, max_kkt_passes=$max_kkt_passes, infeasibility_tol=$infeasibility_tol")
end

"""
    excess(e, s, ϵₐ, ϵᵣ)

By how much the error `e` exceeds its tolerance `ϵₐ + ϵᵣ (s - 1)`, where `s` is the scale
stored in [`KKTErrors`](@ref), i.e. one plus the raw norm.
"""
@inline excess(e, s, ϵₐ, ϵᵣ) = e - (ϵₐ + ϵᵣ * (s - one(s)))

"""
    termination_margin!!(dest, err, params)

Compute, column by column into `dest`, by how much the KKT errors `err` exceed their tolerances:
the instance has converged when the result is nonpositive.

Following cuPDLP.jl (`optimality_criteria_met`) and cuOpt (`pdlp_termination_strategy`), with
`ϵₐ = termination_abstol` and `ϵᵣ = termination_reltol`, convergence means

    ‖primal residual‖ ≤ ϵₐ + ϵᵣ ‖b‖
    ‖dual residual‖   ≤ ϵₐ + ϵᵣ ‖c‖
    |pobj - dobj|     ≤ ϵₐ + ϵᵣ (|pobj| + |dobj|)

where `b` combines the finite constraint bounds and the norm is `termination_norm` (in [`GenericParameters`](@ref)). Since
the scales stored in [`KKTErrors`](@ref) are `1 + ‖b‖`, `1 + ‖c‖` and `1 + |pobj| + |dobj|`, the
default `ϵₐ = ϵᵣ` is the purely relative criterion `error / scale ≤ ϵᵣ`.
"""
function termination_margin!!(dest::BatchedNumber, err::KKTErrors, params::TerminationParameters)
    (; primal, primal_scale, dual, dual_scale, gap, gap_scale) = err
    (; termination_abstol, termination_reltol) = params
    return broadcast!!(
        dest, primal, primal_scale, dual, dual_scale, gap, gap_scale,
        termination_abstol, termination_reltol,
    ) do p, ps, d, ds, g, gs, ϵₐ, ϵᵣ
        max(excess(p, ps, ϵₐ, ϵᵣ), excess(d, ds, ϵₐ, ϵᵣ), excess(g, gs, ϵₐ, ϵᵣ))
    end
end


"""
    ConvergenceStats

# Fields

$(TYPEDFIELDS)
"""
mutable struct ConvergenceStats{
        T <: BatchedNumber, F <: Number, I <: Number, S <: Number, C <: BatchedNumber,
        R <: Union{Nothing, PrimalDualSolution},
        A <: Union{Nothing, AbstractVector},  # of `Bool`s, traced as `TracedRNumber{Bool}`s
        IP <: Union{Nothing, AbstractVector},
    }
    "current KKT error"
    err::KKTErrors{T}
    "time at which the algorithm started, in seconds"
    starting_time::F
    "time elapsed since the algorithm started, in seconds"
    time_elapsed::F
    "number of multiplications by both the KKT matrix and its transpose made by the algorithm itself, which is what `max_kkt_passes` limits"
    kkt_passes::I
    "number of KKT passes spent on infeasibility checks (see [`infeasibility_check!`](@ref)), kept out of `kkt_passes` so that turning the detection on or off does not change the iteration budget"
    ray_passes::I
    "integer code of the termination status (should be that of `MOI.OPTIMIZE_NOT_CALLED` until the algorithm actually terminates), read with [`termination_status`](@ref)"
    termination_status_code::S
    "integer code of the status reached by each instance (`MOI.OPTIMIZE_NOT_CALLED` while it is still running), see [`set_termination_status!!`](@ref)"
    instance_status_code::C
    "error of the best primal ray (certificate of dual infeasibility) found at the last check, `Inf` if none qualifies"
    primal_ray_error::T
    "error of the best dual ray (certificate of primal infeasibility) found at the last check, `Inf` if none qualifies"
    dual_ray_error::T
    "infeasibility certificates in the original (unscaled) space, filled at the end of `solve!`: `ray.x` is a primal ray (dual infeasibility) and `ray.y` a dual ray (primal infeasibility); `nothing` when the detection is disabled"
    ray::R
    "history of KKT errors, indexed by number of KKT passes"
    error_history::Vector{Tuple{I, KKTErrors{T}}}
    "whether each instance of a batch is still running: the others are frozen, see [`freeze_terminated!`](@ref) (`nothing` without batching)"
    active::A
    "number of KKT passes of each instance of a batch when it terminated, or so far if it is still running (`nothing` without batching)"
    instance_kkt_passes::IP
end

"""
    batched_tracker(like, val)

Return `nothing` if `like` is a number, or a vector shaped like `like` filled with `val`: the
per-instance bookkeeping of [`ConvergenceStats`](@ref) only exists for a batch.
"""
batched_tracker(::Number, ::Number) = nothing
batched_tracker(like::AbstractVector, val::Number) = fill!(similar(like, typeof(val)), val)

"""
    active_mask(stats)

Return the per-instance mask of the instances still running, or `true` without batching, to be
passed to the masked updates such as [`masked_broadcast!`](@ref).
"""
active_mask(stats::ConvergenceStats) = active_mask(stats.active)
active_mask(::Nothing) = true
active_mask(active::AbstractVector) = active

instance_or(::Nothing, ::Int, default) = default
instance_or(v::AbstractVector, i::Int, ::Any) = v[i]

"""
    batched_fill(like, val)

Return `val` itself if `like` is a number, or a vector shaped like `like` filled with `val`.
"""
batched_fill(::Number, val::Number) = val
batched_fill(like::AbstractVector, val::Number) = fill!(similar(like, typeof(val)), val)

function ConvergenceStats(
        err::KKTErrors{T};
        ray::Union{Nothing, PrimalDualSolution} = nothing,
        starting_time = current_time(),
        time_elapsed = 0.0,
        kkt_passes::I = 0,
        ray_passes::I = zero(kkt_passes),
        termination_status_code::S = status_code(MOI.OPTIMIZE_NOT_CALLED),
        instance_status_code::C = batched_fill(err.primal, termination_status_code),
        primal_ray_error = batched_fill(err.primal, convert(eltype(err.primal), Inf)),
        dual_ray_error = batched_fill(err.primal, convert(eltype(err.primal), Inf)),
        error_history = [(kkt_passes, copy(err))],
        active = batched_tracker(err.primal, true),
        instance_kkt_passes = batched_tracker(err.primal, kkt_passes),
    ) where {T, I, S, C}
    F = Base.promote_type(typeof(starting_time), typeof(time_elapsed))
    return ConvergenceStats{
        T, F, I, S, C, typeof(ray), typeof(active), typeof(instance_kkt_passes),
    }(
        err,
        starting_time,
        time_elapsed,
        kkt_passes,
        ray_passes,
        termination_status_code,
        instance_status_code,
        primal_ray_error,
        dual_ray_error,
        ray,
        error_history,
        active,
        instance_kkt_passes,
    )
end

"""
    status_code(status)

Return the integer code of a `MOI.TerminationStatusCode`, as stored in the
`termination_status_code` field of [`ConvergenceStats`](@ref).

Unlike the enum itself, this code is an ordinary number, so Reactant can trace it: a compiled
solve can write its own status instead of freezing the trace-time one.
"""
status_code(status::MOI.TerminationStatusCode) = Int32(status)

"""
    termination_status(stats)

Return the termination status of `stats` as a `MOI.TerminationStatusCode`.

Decodes the `termination_status_code` field, which is stored as a plain number so that it
survives a Reactant-compiled solve. Only call this outside a compilation context: mid-trace the
code is a traced number with no value yet.
"""
function termination_status(stats::ConvergenceStats)
    return MOI.TerminationStatusCode(Int32(stats.termination_status_code))
end

"""
    instance(stats::ConvergenceStats, i)

Return the convergence stats of the `i`-th instance of a batch.

Its termination status is the one this instance reached on its own (optimal or certified
infeasible) if any, and that of the whole batch otherwise (e.g. a time limit). Its number of KKT
passes is the one at which it terminated: the instance was frozen from then on.
"""
function instance(stats::ConvergenceStats, i::Int)
    code_i = instance_num(stats.instance_status_code, i)
    not_called = status_code(MOI.OPTIMIZE_NOT_CALLED)
    return ConvergenceStats(
        instance(stats.err, i);
        ray = instance(stats.ray, i),
        starting_time = stats.starting_time,
        time_elapsed = stats.time_elapsed,
        kkt_passes = instance_or(stats.instance_kkt_passes, i, stats.kkt_passes),
        ray_passes = stats.ray_passes,
        termination_status_code = ifelse(
            code_i == not_called, stats.termination_status_code, code_i
        ),
        instance_status_code = code_i,
        primal_ray_error = instance_num(stats.primal_ray_error, i),
        dual_ray_error = instance_num(stats.dual_ray_error, i),
        error_history = [(passes, instance(err, i)) for (passes, err) in stats.error_history],
    )
end

function Base.show(io::IO, stats::ConvergenceStats)
    (; err, time_elapsed, kkt_passes, ray_passes) = stats
    return print(
        io,
        """Convergence stats with termination status $(termination_status(stats)):
        - $err
        - time elapsed: $time_elapsed seconds
        - KKT passes: $kkt_passes (and $ray_passes on infeasibility checks)""",
    )
end

"""
    certified(ray_error, infeasibility_tol)

Whether a ray with error `ray_error` is an infeasibility certificate: never when the detection
is disabled (`infeasibility_tol === nothing`).
"""
certified(ray_error::Number, infeasibility_tol::Number) = ray_error <= infeasibility_tol
certified(::Number, ::Nothing) = false

"""
    set_termination_status!!(stats, dest, params, time_limit)

Decide how the algorithm terminates, using `dest` as scratch space for the margins computed by
[`termination_margin!!`](@ref). The time limit is passed separately from the other termination
parameters `params`, because it lives in [`GenericParameters`](@ref).

Set `stats.instance_status_code` and `stats.termination_status_code`, and return whether the
algorithm should stop.

Each instance is first given its own status, by decreasing priority:

1. `MOI.OPTIMAL` if its KKT errors meet the tolerances, see [`termination_margin!!`](@ref);
2. `MOI.INFEASIBLE` if a dual ray certifies primal infeasibility (`stats.dual_ray_error`
   below `params.infeasibility_tol`);
3. `MOI.DUAL_INFEASIBLE` if a primal ray certifies dual infeasibility (`stats.primal_ray_error`
   below `params.infeasibility_tol`);
4. `MOI.OPTIMIZE_NOT_CALLED` otherwise, meaning that the instance is still running.

In a batch, an instance keeps the status it terminated with, and is frozen from then on (see
[`freeze_terminated!`](@ref)).

The batch stops once every instance is done, i.e. optimal or certified infeasible, and its
status is then `MOI.OPTIMAL` if all instances are optimal, otherwise `MOI.INFEASIBLE` if any of
them is primal infeasible, otherwise `MOI.DUAL_INFEASIBLE`: read the status of each instance
with `termination_status(instance(stats, i))`. Failing that, the time limit comes before the
iteration limit, and both override the per-instance statuses only for the instances still
running.

The statuses are selected with nested `ifelse` calls rather than with branches, so that they
survive a Reactant-compiled solve: an assignment inside a `@trace if` only ever contributes its
trace-time value, whereas an `ifelse` over traced conditions is part of the compiled program.
The returned boolean is what the solve loops branch on, since a status code cannot drive traced
control flow.
"""
function set_termination_status!!(
        stats::ConvergenceStats,
        dest::BatchedNumber,
        params::TerminationParameters,
        time_limit::Number
    )
    (; err, time_elapsed, kkt_passes, primal_ray_error, dual_ray_error) = stats
    (; max_kkt_passes, infeasibility_tol) = params
    optimal = status_code(MOI.OPTIMAL)
    infeasible = status_code(MOI.INFEASIBLE)
    dual_infeasible = status_code(MOI.DUAL_INFEASIBLE)
    running = status_code(MOI.OPTIMIZE_NOT_CALLED)
    margin = termination_margin!!(dest, err, params)
    # nested `ifelse` instead of `if`/`elseif`: it is a plain (traceable) computation, and it
    # states the priority between simultaneous criteria in one place
    stats.instance_status_code = masked_apply!!(
        stats.instance_status_code, active_mask(stats), margin, dual_ray_error, primal_ray_error,
    ) do r, de, pe
        ifelse(
            r <= zero(r),
            optimal,
            ifelse(
                certified(de, infeasibility_tol),
                infeasible,
                ifelse(certified(pe, infeasibility_tol), dual_infeasible, running),
            ),
        )
    end
    codes = stats.instance_status_code
    all_optimal = batched_all(==(optimal), codes)
    all_done = batched_all(!=(running), codes)
    any_infeasible = !batched_all(!=(infeasible), codes)
    is_time_limit = time_elapsed >= time_limit
    is_iteration_limit = kkt_passes >= max_kkt_passes
    stats.termination_status_code = ifelse(
        all_optimal,
        optimal,
        ifelse(
            all_done,
            ifelse(any_infeasible, infeasible, dual_infeasible),
            ifelse(
                is_time_limit,
                status_code(MOI.TIME_LIMIT),
                ifelse(is_iteration_limit, status_code(MOI.ITERATION_LIMIT), running),
            ),
        ),
    )
    freeze_terminated!(stats)
    return all_done | is_time_limit | is_iteration_limit
end

"""
    freeze_terminated!(stats)

Record the KKT passes of the instances of a batch which are still running, then mark those
which just terminated as inactive. Does nothing without batching.

The iterates of an inactive instance are no longer updated: every step masks its columns (see
[`masked_broadcast!`](@ref)), so its solution, its KKT errors and its status stay those it
terminated with, while the rest of the batch keeps going. Its columns of the sparse products are
still computed, since the batched kernels run over the whole batch.
"""
freeze_terminated!(stats::ConvergenceStats) = freeze_terminated!(stats, stats.active)
freeze_terminated!(::ConvergenceStats, ::Nothing) = nothing
function freeze_terminated!(stats::ConvergenceStats, active::AbstractVector)
    (; kkt_passes) = stats
    running = status_code(MOI.OPTIMIZE_NOT_CALLED)
    stats.instance_kkt_passes = broadcast!!(
        ifelse, stats.instance_kkt_passes, active, kkt_passes, stats.instance_kkt_passes
    )
    stats.active = broadcast!!(==(running), active, stats.instance_status_code)
    return nothing
end
