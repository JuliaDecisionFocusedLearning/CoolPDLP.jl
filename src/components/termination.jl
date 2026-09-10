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
    TerminationParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct TerminationParameters{T <: Number, I <: Number, F <: Number}
    "tolerance on KKT relative errors to decide termination"
    termination_reltol::T
    "maximum number of multiplications by both the KKT matrix `K` and its transpose `Kᵀ`"
    max_kkt_passes::I
    "time limit in seconds"
    time_limit::F
end

function Base.show(io::IO, params::TerminationParameters)
    (; termination_reltol, max_kkt_passes, time_limit) = params
    return print(io, "TerminationParameters: termination_reltol=$termination_reltol, max_kkt_passes=$max_kkt_passes, time_limit=$time_limit")
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
    set_termination_status!!(stats, dest, params)

Decide how the algorithm terminates, using `dest` as scratch space for the relative errors.

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
        params::TerminationParameters
    )
    (; err, time_elapsed, kkt_passes) = stats
    (; termination_reltol, time_limit, max_kkt_passes) = params
    is_optimal = batched_all(<=(termination_reltol), relative!!(dest, err))
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
