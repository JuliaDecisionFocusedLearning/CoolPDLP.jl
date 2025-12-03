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
mutable struct RestartStats{T <: Number}
    restart_from_avg::Bool
    err_candidate::KKTErrors{T}
    err_candidate_last::KKTErrors{T}
    err_restart::KKTErrors{T}

    function RestartStats(::Type{T}) where {T}
        return new{T}(false, KKTErrors(T), KKTErrors(T), KKTErrors(T))
    end
end

function should_restart(
        stats::RestartStats, step_sizes::StepSizes, iteration::IterationCounter, params::RestartParameters,
    )
    (; ω) = step_sizes
    (; err_candidate, err_candidate_last, err_restart) = stats
    (; sufficient_decay, necessary_decay, artificial_decay) = params
    (; inner, total) = iteration

    sufficient = absolute(err_candidate, ω) <= sufficient_decay * absolute(err_restart, ω)
    necessary = absolute(err_candidate, ω) <= necessary_decay * absolute(err_restart, ω)
    no_local_progress = absolute(err_candidate, ω) > absolute(err_candidate_last, ω)
    long_inner_loop = inner >= artificial_decay * total

    restart_criterion = sufficient ||
        (necessary && no_local_progress) ||
        long_inner_loop
    return restart_criterion
end
