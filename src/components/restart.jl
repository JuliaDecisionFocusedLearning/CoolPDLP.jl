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
    "scratch for the KKT errors of a candidate"
    err::KKTErrors{T}
    "scratch for the KKT errors of the candidate it is compared against"
    err_other::KKTErrors{T}
    "absolute error of the restart candidate"
    abs_candidate::T
    "absolute error of the previous restart candidate"
    abs_candidate_last::T
    "absolute error at the last restart"
    abs_restart::T
end

function RestartStats(sol::PrimalDualSolution{T}) where {T}
    nan() = batched_expand(sol.x, convert(T, NaN))
    return RestartStats(
        batched_expand(sol.x, false),
        KKTErrors(sol), KKTErrors(sol),
        nan(), nan(), nan(),
    )
end

instance(stats::RestartStats, i::Int) = RestartStats(
    instance_num(stats.restart_from_avg, i),
    instance(stats.err, i),
    instance(stats.err_other, i),
    instance_num(stats.abs_candidate, i),
    instance_num(stats.abs_candidate_last, i),
    instance_num(stats.abs_restart, i),
)

"""
    should_restart(stats, iteration, params)

Decide whether the whole batch restarts, based on the fixed-point residual averaged over the columns.

Since every instance of the batch restarts at the same time, the three usual criteria (sufficient decay, necessary decay without local progress, long inner loop) are applied to the mean of the per-column absolute KKT errors instead of requiring each column to agree.
"""
function should_restart(
        stats::RestartStats, iteration::IterationCounter, params::RestartParameters,
    )
    (; abs_candidate, abs_candidate_last, abs_restart) = stats
    (; sufficient_decay, necessary_decay, artificial_decay) = params
    (; inner, total) = iteration

    candidate = batched_mean(abs_candidate)
    candidate_last = batched_mean(abs_candidate_last)
    restart = batched_mean(abs_restart)

    sufficient = candidate <= sufficient_decay * restart
    necessary = candidate <= necessary_decay * restart
    no_local_progress = candidate > candidate_last
    long_inner_loop = inner >= artificial_decay * total
    return sufficient || (necessary && no_local_progress) || long_inner_loop
end
