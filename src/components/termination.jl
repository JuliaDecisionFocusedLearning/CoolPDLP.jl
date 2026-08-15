"""
    TerminationStatus

Enum for the various ways that an algorithm can terminate.

Possible values:

- `OPTIMAL`
- `TIME_LIMIT`
- `ITERATION_LIMIT`
- `STILL_RUNNING`
"""
@enum TerminationStatus OPTIMAL TIME_LIMIT ITERATION_LIMIT STILL_RUNNING

"""
    TerminationParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct TerminationParameters{T <: Number}
    "tolerance on KKT relative errors to decide termination"
    termination_reltol::T
    "maximum number of multiplications by both the KKT matrix `K` and its transpose `Kᵀ`"
    max_kkt_passes::Int
    "time limit in seconds"
    time_limit::Float64
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
mutable struct ConvergenceStats{T <: BatchedNumber}
    "current KKT error"
    err::KKTErrors{T}
    "time at which the algorithm started, in seconds"
    starting_time::Float64
    "time elapsed since the algorithm started, in seconds"
    time_elapsed::Float64
    "number of multiplications by both the KKT matrix and its transpose"
    kkt_passes::Int
    "termination stats (should be `STILL_RUNNING` until the algorithm actually terminates)"
    termination_status::TerminationStatus
    "history of KKT errors, indexed by number of KKT passes"
    const error_history::Vector{Tuple{Int, KKTErrors{T}}}

    function ConvergenceStats(
            err::KKTErrors{T};
            starting_time = time(),
            time_elapsed = 0.0,
            kkt_passes = 0,
            termination_status = STILL_RUNNING,
            error_history = Tuple{Int, KKTErrors{T}}[]
        ) where {T}
        return new{T}(
            err,
            starting_time,
            time_elapsed,
            kkt_passes,
            termination_status,
            error_history
        )
    end
end

function instance(stats::ConvergenceStats, i::Int)
    return ConvergenceStats(
        instance(stats.err, i);
        starting_time = stats.starting_time,
        time_elapsed = stats.time_elapsed,
        kkt_passes = stats.kkt_passes,
        termination_status = stats.termination_status,
        error_history = [(passes, instance(err, i)) for (passes, err) in stats.error_history],
    )
end

function Base.show(io::IO, stats::ConvergenceStats)
    (; err, time_elapsed, kkt_passes, termination_status) = stats
    return print(
        io,
        """Convergence stats with termination status $termination_status:
        - $err
        - time elapsed: $(round(time_elapsed; digits = 3)) seconds
        - KKT passes: $kkt_passes""",
    )
end

"""
    termination_status!!(dest, stats, params)

Decide how the algorithm terminates, using `dest` as scratch space for the relative errors.
"""
function termination_status!!(
        dest::BatchedNumber, stats::ConvergenceStats, params::TerminationParameters
    )
    (; err, time_elapsed, kkt_passes) = stats
    (; termination_reltol, time_limit, max_kkt_passes) = params
    st = if batched_all(<=(termination_reltol), relative!!(dest, err))
        OPTIMAL
    elseif time_elapsed >= time_limit
        TIME_LIMIT
    elseif kkt_passes >= max_kkt_passes
        ITERATION_LIMIT
    else
        STILL_RUNNING
    end
    return st
end

function should_terminate!!(
        dest::BatchedNumber, stats::ConvergenceStats, params::TerminationParameters
    )
    (; err, time_elapsed, kkt_passes) = stats
    (; termination_reltol, time_limit, max_kkt_passes) = params
    is_optimal = batched_all(<=(termination_reltol), relative!!(dest, err))
    is_time_limit = time_elapsed >= time_limit
    is_iteration_limit = kkt_passes >= max_kkt_passes
    return is_optimal || is_time_limit || is_iteration_limit
end
