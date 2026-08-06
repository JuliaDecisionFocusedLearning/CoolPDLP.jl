"""
    RestartParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct RestartParameters{T <: Number, F}
    "restart criterion: sufficient decay in normalized duality gap"
    sufficient_decay::T
    "restart criterion: necessary decay"
    necessary_decay::T
    "restart criterion: long inner loop"
    artificial_decay::T
    "how the per-instance absolute errors are reduced to the single restart decision shared by the batch"
    batch_aggregation::F = batched_mean
end

function Base.show(io::IO, params::RestartParameters)
    (; sufficient_decay, necessary_decay, artificial_decay, batch_aggregation) = params
    return print(io, "RestartParameters: sufficient_decay=$sufficient_decay, necessary_decay=$necessary_decay, artificial_decay=$artificial_decay, batch_aggregation=$batch_aggregation")
end

"""
    RestartStats

# Fields

$(TYPEDFIELDS)
"""
mutable struct RestartStats{T <: BatchedNumber, B <: Union{Bool, AbstractVector{Bool}}}
    "whether to restart from the average solution, column by column"
    restart_from_avg::B
    "KKT errors of the current solution"
    err_current::KKTErrors{T}
    "KKT errors of the current average solution"
    err_avg::KKTErrors{T}
    "KKT errors of the last solution"
    err_last::KKTErrors{T}
    "KKT errors of the last average solution"
    err_avg_last::KKTErrors{T}
    "KKT errors of the solution at the last restart"
    err_restart::KKTErrors{T}
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
        KKTErrors(sol), KKTErrors(sol), KKTErrors(sol), KKTErrors(sol), KKTErrors(sol),
        nan(), nan(), nan(),
    )
end

instance(stats::RestartStats, i::Int) = RestartStats(
    instance_num(stats.restart_from_avg, i),
    instance(stats.err_current, i),
    instance(stats.err_avg, i),
    instance(stats.err_last, i),
    instance(stats.err_avg_last, i),
    instance(stats.err_restart, i),
    instance_num(stats.abs_candidate, i),
    instance_num(stats.abs_candidate_last, i),
    instance_num(stats.abs_restart, i),
)

"""
    should_restart(stats, iteration, params)

Decide whether the whole batch restarts, based on an aggregate of the per-column fixed-point residuals.

Since every instance of the batch restarts at the same time, the three usual criteria (sufficient decay, necessary decay without local progress, long inner loop) are applied to `params.batch_aggregation` (the mean by default) of the per-column absolute KKT errors instead of requiring each column to agree.
"""
function should_restart(
        stats::RestartStats, iteration::IterationCounter, params::RestartParameters,
    )
    (; abs_candidate, abs_candidate_last, abs_restart) = stats
    (; sufficient_decay, necessary_decay, artificial_decay, batch_aggregation) = params
    (; inner, total) = iteration

    candidate = batch_aggregation(abs_candidate)
    candidate_last = batch_aggregation(abs_candidate_last)
    restart = batch_aggregation(abs_restart)

    sufficient = candidate <= sufficient_decay * restart
    necessary = candidate <= necessary_decay * restart
    no_local_progress = candidate > candidate_last
    long_inner_loop = inner >= artificial_decay * total
    return sufficient || (necessary && no_local_progress) || long_inner_loop
end
