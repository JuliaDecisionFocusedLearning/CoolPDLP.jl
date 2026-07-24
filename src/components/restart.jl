"""
    RestartParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct RestartParameters{T <: Number}
    "restart criterion: sufficient decay in normalized duality gap"
    sufficient_decay::T
    "restart criterion: necessary decay"
    necessary_decay::T
    "restart criterion: long inner loop"
    artificial_decay::T
end

function Base.show(io::IO, params::RestartParameters)
    (; sufficient_decay, necessary_decay, artificial_decay) = params
    return print(io, "RestartParameters: sufficient_decay=$sufficient_decay, necessary_decay=$necessary_decay, artificial_decay=$artificial_decay")
end

"""
    RestartStats

# Fields

$(TYPEDFIELDS)
"""
mutable struct RestartStats{T <: BatchedNumber, B <: Union{Bool, AbstractVector{Bool}}}
    "whether to restart from the average solution, column by column"
    restart_from_avg::B
    "KKT errors of the restart candidate"
    err_candidate::KKTErrors{T}
    "KKT errors of the previous restart candidate"
    err_candidate_last::KKTErrors{T}
    "KKT errors at the last restart"
    err_restart::KKTErrors{T}
    "buffer for the KKT errors of the discarded candidate"
    err_other::KKTErrors{T}
    "absolute error of the restart candidate"
    abs_candidate::T
    "absolute error of the previous restart candidate"
    abs_candidate_last::T
    "absolute error at the last restart"
    abs_restart::T

    function RestartStats(
            restart_from_avg::B,
            err_candidate::KKTErrors{T},
            err_candidate_last::KKTErrors{T},
            err_restart::KKTErrors{T},
            err_other::KKTErrors{T},
            abs_candidate::T,
            abs_candidate_last::T,
            abs_restart::T,
        ) where {T, B}
        return new{T, B}(
            restart_from_avg,
            err_candidate, err_candidate_last, err_restart, err_other,
            abs_candidate, abs_candidate_last, abs_restart,
        )
    end
end

function RestartStats(sol::PrimalDualSolution{T}) where {T}
    nan() = batch_expand(sol.x, convert(T, NaN))
    return RestartStats(
        batch_expand(sol.x, false),
        KKTErrors(sol), KKTErrors(sol), KKTErrors(sol), KKTErrors(sol),
        nan(), nan(), nan(),
    )
end

batch(stats::RestartStats, i::Int) = RestartStats(
    batch_num(stats.restart_from_avg, i),
    batch(stats.err_candidate, i),
    batch(stats.err_candidate_last, i),
    batch(stats.err_restart, i),
    batch(stats.err_other, i),
    batch_num(stats.abs_candidate, i),
    batch_num(stats.abs_candidate_last, i),
    batch_num(stats.abs_restart, i),
)

function should_restart(
        stats::RestartStats, iteration::IterationCounter, params::RestartParameters,
    )
    (; abs_candidate, abs_candidate_last, abs_restart) = stats
    (; sufficient_decay, necessary_decay, artificial_decay) = params
    (; inner, total) = iteration

    # the whole batch restarts at once, so every column has to agree
    restart_criterion = batch_all(
        abs_candidate, abs_candidate_last, abs_restart
    ) do candidate, candidate_last, restart
        sufficient = candidate <= sufficient_decay * restart
        necessary = candidate <= necessary_decay * restart
        no_local_progress = candidate > candidate_last
        return sufficient | (necessary & no_local_progress)
    end
    long_inner_loop = inner >= artificial_decay * total
    return restart_criterion || long_inner_loop
end
