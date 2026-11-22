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
    TerminationParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct TerminationParameters{T <: Number, I <: Number}
    "relative tolerance on the KKT errors to decide termination, see [`termination_margin!!`](@ref)"
    termination_reltol::T
    "absolute tolerance on the KKT errors to decide termination, see [`termination_margin!!`](@ref)"
    termination_abstol::T = termination_reltol
    "norm in which the primal and dual residuals are measured, `:L2` or `:Linf`"
    termination_norm::Symbol = :L2
    "maximum number of multiplications by both the KKT matrix `K` and its transpose `Kᵀ`"
    max_kkt_passes::I
end

function Base.show(io::IO, params::TerminationParameters)
    (; termination_reltol, termination_abstol, termination_norm, max_kkt_passes) = params
    return print(io, "TerminationParameters: termination_reltol=$termination_reltol, termination_abstol=$termination_abstol, termination_norm=$termination_norm, max_kkt_passes=$max_kkt_passes")
end

const TERMINATION_NORMS = (:L2, :Linf)

"""
    check_termination_parameters(params)

Throw an `ArgumentError` if the norm of `params` is not a supported one.
"""
function check_termination_parameters(params::TerminationParameters)
    if !(params.termination_norm in TERMINATION_NORMS)
        throw(ArgumentError("termination_norm must be one of $TERMINATION_NORMS, got $(repr(params.termination_norm))"))
    end
    return params
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

where `b` combines the finite constraint bounds and the norm is `params.termination_norm`. Since
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
mutable struct ConvergenceStats{T <: BatchedNumber, F <: Number, I <: Number, S <: Number}
    "current KKT error"
    err::KKTErrors{T}
    "time at which the algorithm started, in seconds"
    starting_time::F
    "time elapsed since the algorithm started, in seconds"
    time_elapsed::F
    "number of multiplications by both the KKT matrix and its transpose"
    kkt_passes::I
    "integer code of the termination status (should be that of `MOI.OPTIMIZE_NOT_CALLED` until the algorithm actually terminates), read with [`termination_status`](@ref)"
    termination_status_code::S
    "history of KKT errors, indexed by number of KKT passes"
    error_history::Vector{Tuple{I, KKTErrors{T}}}
end

function ConvergenceStats(
        err::KKTErrors{T};
        starting_time = current_time(),
        time_elapsed = 0.0,
        kkt_passes::I = 0,
        termination_status_code::S = status_code(MOI.OPTIMIZE_NOT_CALLED),
        error_history = [(kkt_passes, copy(err))]
    ) where {T, I, S}
    F = Base.promote_type(typeof(starting_time), typeof(time_elapsed))
    return ConvergenceStats{T, F, I, S}(
        err,
        starting_time,
        time_elapsed,
        kkt_passes,
        termination_status_code,
        error_history
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

function instance(stats::ConvergenceStats, i::Int)
    return ConvergenceStats(
        instance(stats.err, i);
        starting_time = stats.starting_time,
        time_elapsed = stats.time_elapsed,
        kkt_passes = stats.kkt_passes,
        termination_status_code = stats.termination_status_code,
        error_history = [(passes, instance(err, i)) for (passes, err) in stats.error_history],
    )
end

function Base.show(io::IO, stats::ConvergenceStats)
    (; err, time_elapsed, kkt_passes) = stats
    return print(
        io,
        """Convergence stats with termination status $(termination_status(stats)):
        - $err
        - time elapsed: $time_elapsed seconds
        - KKT passes: $kkt_passes""",
    )
end

"""
    set_termination_status!!(stats, dest, params, time_limit)

Decide how the algorithm terminates, using `dest` as scratch space for the margins computed by
[`termination_margin!!`](@ref).

The time limit is passed separately from the other termination parameters `params`, because it
lives in [`GenericParameters`](@ref).

Set `stats.termination_status_code` and return whether the algorithm should stop.

The status is selected with nested `ifelse` calls rather than with branches, so that it survives
a Reactant-compiled solve: an assignment inside a `@trace if` only ever contributes its
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
    (; err, time_elapsed, kkt_passes) = stats
    (; max_kkt_passes) = params
    is_optimal = batched_all(<=(0), termination_margin!!(dest, err, params))
    is_time_limit = time_elapsed >= time_limit
    is_iteration_limit = kkt_passes >= max_kkt_passes
    # nested `ifelse` instead of `if`/`elseif`: it is a plain (traceable) computation, and it
    # states the priority between simultaneous criteria in one place
    stats.termination_status_code = ifelse(
        is_optimal,
        status_code(MOI.OPTIMAL),
        ifelse(
            is_time_limit,
            status_code(MOI.TIME_LIMIT),
            ifelse(
                is_iteration_limit,
                status_code(MOI.ITERATION_LIMIT),
                status_code(MOI.OPTIMIZE_NOT_CALLED),
            ),
        ),
    )
    return is_optimal | is_time_limit | is_iteration_limit
end
