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
    err_candidate::KKTErrors{T}
    err_candidate_last::KKTErrors{T}
    err_restart::KKTErrors{T}

    function RestartStats(
            restart_from_avg::B,
            err_candidate::KKTErrors{T},
            err_candidate_last::KKTErrors{T},
            err_restart::KKTErrors{T},
        ) where {T, B}
        return new{T, B}(restart_from_avg, err_candidate, err_candidate_last, err_restart)
    end
end

function RestartStats(sol::PrimalDualSolution)
    return RestartStats(
        batch_expand(sol.x, false), KKTErrors(sol), KKTErrors(sol), KKTErrors(sol)
    )
end

batch(stats::RestartStats, i::Int) = RestartStats(
    batch_num(stats.restart_from_avg, i),
    batch(stats.err_candidate, i),
    batch(stats.err_candidate_last, i),
    batch(stats.err_restart, i),
)

function should_restart(
        stats::RestartStats, step_sizes::StepSizes, iteration::IterationCounter, params::RestartParameters,
    )
    (; ω) = step_sizes
    (; err_candidate, err_candidate_last, err_restart) = stats
    (; sufficient_decay, necessary_decay, artificial_decay) = params
    (; inner, total) = iteration

    candidate = absolute(err_candidate, ω)
    candidate_last = absolute(err_candidate_last, ω)
    restart = absolute(err_restart, ω)

    sufficient = @. candidate <= sufficient_decay * restart
    necessary = @. candidate <= necessary_decay * restart
    no_local_progress = @. candidate > candidate_last
    long_inner_loop = inner >= artificial_decay * total

    # the whole batch restarts at once, so every column has to agree
    restart_criterion = batch_all(@. sufficient | (necessary & no_local_progress))
    return restart_criterion || long_inner_loop
end
