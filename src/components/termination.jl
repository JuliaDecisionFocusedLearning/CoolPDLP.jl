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
mutable struct ConvergenceStats{T <: BatchedNumber, F <: Number, I <: Number}
    "current KKT error"
    err::KKTErrors{T}
    "time at which the algorithm started, in seconds"
    starting_time::F
    "time elapsed since the algorithm started, in seconds"
    time_elapsed::F
    "number of multiplications by both the KKT matrix and its transpose"
    kkt_passes::I
    "termination status (should be `MOI.OPTIMIZE_NOT_CALLED` until the algorithm actually terminates)"
    termination_status::MOI.TerminationStatusCode
    "history of KKT errors, indexed by number of KKT passes"
    error_history::Vector{Tuple{I, KKTErrors{T}}}
end

function ConvergenceStats(
        err::KKTErrors{T};
        starting_time = time(),
        time_elapsed = 0.0,
        kkt_passes::I = 0,
        termination_status = MOI.OPTIMIZE_NOT_CALLED,
        error_history = [(kkt_passes, err)]
    ) where {T, I}
    F = Base.promote_type(typeof(starting_time), typeof(time_elapsed))
    return ConvergenceStats{T, F, I}(
        err,
        starting_time,
        time_elapsed,
        kkt_passes,
        termination_status,
        error_history
    )
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
        - time elapsed: $time_elapsed seconds
        - KKT passes: $kkt_passes""",
    )
end

"""
    set_termination_status!!(stats, dest, params)

Decide how the algorithm terminates, using `dest` as scratch space for the relative errors.
"""
function set_termination_status!!(
        stats::ConvergenceStats,
        dest::BatchedNumber,
        params::TerminationParameters
    )
    (; err, time_elapsed, kkt_passes) = stats
    (; termination_reltol, time_limit, max_kkt_passes) = params
    is_optimal = batched_all(<=(termination_reltol), relative!!(dest, err))
    # Reactant doesn't like `elseif`, see https://github.com/EnzymeAD/Reactant.jl/issues/2563#issuecomment-5584197336
    @trace if is_optimal
        stats.termination_status = MOI.OPTIMAL
    end
    @trace if time_elapsed >= time_limit
        stats.termination_status = MOI.TIME_LIMIT
    end
    @trace if kkt_passes >= max_kkt_passes
        stats.termination_status = MOI.ITERATION_LIMIT
    end
    return nothing
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
