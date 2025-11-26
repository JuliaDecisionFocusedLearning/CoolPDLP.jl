"""
    RestartParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct RestartParameters{T <: Number}
    "restart criterion: sufficient decay in normalized duality gap"
    β_sufficient::T
    "restart criterion: necessary decay"
    β_necessary::T
    "restart criterion: long inner loop"
    β_artificial::T
end

function Base.show(io::IO, params::RestartParameters)
    (; β_sufficient, β_necessary, β_artificial) = params
    return print(io, "RestartParameters: β_sufficient=$β_sufficient, β_necessary=$β_necessary, β_artificial=$β_artificial")
end

"""
    RestartStats

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct RestartStats{T <: Number}
    err_restart_candidate::KKTErrors{T}
    err_restart_candidate_last::KKTErrors{T}
    err_previous_restart::KKTErrors{T}
    inner_iterations::Int
    total_iterations::Int
end

function restart_check(stats::RestartStats, params::RestartParameters)
    (;
        err_restart_candidate, err_restart_candidate_last, err_previous_restart,
        inner_iterations, total_iterations,
    ) = stats
    (; β_sufficient, β_necessary, β_artificial) = params

    sufficient_decay = absolute(err_restart_candidate) <= β_sufficient * absolute(err_previous_restart)
    necessary_decay = absolute(err_restart_candidate) <= β_necessary * absolute(err_previous_restart)
    no_local_progress = absolute(err_restart_candidate) > absolute(err_restart_candidate_last)
    long_inner_loop = inner_iterations >= β_artificial * total_iterations

    restart_criterion = sufficient_decay ||
        (necessary_decay && no_local_progress) ||
        long_inner_loop
    return restart_criterion
end
