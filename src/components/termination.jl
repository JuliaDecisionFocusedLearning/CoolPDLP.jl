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
mutable struct ConvergenceStats{T <: Number}
    "current KKT error"
    err::KKTErrors{T}
    "time at which the algorithm started, in seconds"
    starting_time::Float64
    "time elapsed since the algorithm started, in seconds"
    time_elapsed::Float64
    "number of multiplications by both the KKT matrix and its transpose"
    kkt_passes::Int
    "termination stats (should be `nothing` until the algorithm actuall terminates)"
    termination_status::Union{Nothing, TerminationStatusCode}
    "history of KKT errors, indexed by number of KKT passes"
    const error_history::Vector{Tuple{Int, KKTErrors{T}}}

    function ConvergenceStats(
            ::Type{T};
            err = KKTErrors(T),
            starting_time = time(),
            time_elapsed = 0.0,
            kkt_passes = 0,
            termination_status = nothing,
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

function Base.show(io::IO, stats::ConvergenceStats)
    (; err, time_elapsed, kkt_passes, termination_status) = stats
    return print(
        io,
        """Convergence stats with termination status $termination_status:
          - $err
          - time elapsed: $(round(time_elapsed; digits = 3)) seconds 
          - KKT passes: $kkt_passes
        """,
    )
end

function termination_status(stats::ConvergenceStats, params::TerminationParameters)
    (; err, time_elapsed, kkt_passes) = stats
    (; termination_reltol, time_limit, max_kkt_passes) = params
    if relative(err) <= termination_reltol
        return OPTIMAL
    elseif time_elapsed >= time_limit
        return TIME_LIMIT
    elseif kkt_passes >= max_kkt_passes
        return ITERATION_LIMIT
    else
        return nothing
    end
end
