module CoolPDLPReactantExt

using CoolPDLP:
    Algorithm,
    CoolPDLP,
    ConvergenceStats,
    ConversionParameters,
    GenericParameters,
    KKTErrors,
    MILP,
    PDHGState,
    PreconditioningParameters,
    PrimalDualSolution,
    RestartParameters,
    Scratch,
    StepSizeParameters,
    StepSizes,
    TerminationParameters,
    custom_to_rarray
using Reactant: ConcreteRArray, ConcreteRNumber, to_rarray

function CoolPDLP.custom_to_rarray(milp::MILP; kwargs...)
    return to_rarray(milp; kwargs...)
end

function CoolPDLP.custom_to_rarray(sol::PrimalDualSolution; kwargs...)
    return to_rarray(sol; kwargs...)
end

function CoolPDLP.custom_to_rarray(scratch::Scratch; kwargs...)
    (; x, y, z, b1, b2) = scratch
    xr = to_rarray(x; kwargs...)
    yr = to_rarray(y; kwargs...)
    zr = to_rarray(z; kwargs...)
    b1r = to_rarray(b1; kwargs...)
    b2r = to_rarray(b2; kwargs...)
    return Scratch(;
        x = xr, y = yr, z = zr, b1 = b1r, b2 = b2r
    )
end

function CoolPDLP.custom_to_rarray(stats::ConvergenceStats; kwargs...)
    (; err, starting_time, time_elapsed, kkt_passes, termination_status, error_history) = stats
    return ConvergenceStats(
        to_rarray(err; kwargs...);
        starting_time,
        time_elapsed,
        kkt_passes,
        termination_status,
        error_history = to_rarray(error_history; kwargs...)
    )
end

function CoolPDLP.custom_to_rarray(state::PDHGState; kwargs...)
    (; sol, sol_last, step_sizes, scratch, stats) = state
    sol_r = to_rarray(sol; kwargs...)
    sol_last_r = to_rarray(sol_last; kwargs...)
    step_sizes_r = to_rarray(step_sizes; kwargs...)
    scratch_r = custom_to_rarray(scratch; kwargs...)
    stats_r = custom_to_rarray(stats; kwargs...)
    return PDHGState(;
        sol = sol_r,
        sol_last = sol_last_r,
        step_sizes = step_sizes_r,
        scratch = scratch_r,
        stats = stats_r,
    )
end

function CoolPDLP.custom_to_rarray(conversion::ConversionParameters; kwargs...)
    return to_rarray(conversion; kwargs...)
end

function CoolPDLP.custom_to_rarray(preconditioning::PreconditioningParameters; kwargs...)
    return to_rarray(preconditioning; kwargs...)
end

function CoolPDLP.custom_to_rarray(step_size::StepSizeParameters; kwargs...)
    return to_rarray(step_size; kwargs...)
end

function CoolPDLP.custom_to_rarray(restart::RestartParameters; kwargs...)
    (; sufficient_decay, necessary_decay, artificial_decay, batch_aggregation) = restart
    return RestartParameters(;
        sufficient_decay = ConcreteRNumber(sufficient_decay),
        necessary_decay = ConcreteRNumber(necessary_decay),
        artificial_decay = ConcreteRNumber(artificial_decay),
        batch_aggregation = batch_aggregation
    )
end

function CoolPDLP.custom_to_rarray(generic::GenericParameters; kwargs...)
    (; show_progress, check_every, record_error_history) = generic
    return GenericParameters(;
        show_progress,
        check_every = ConcreteRNumber(check_every),
        record_error_history
    )
end

function CoolPDLP.custom_to_rarray(termination::TerminationParameters; kwargs...)
    (; termination_reltol, max_kkt_passes, time_limit) = termination
    return TerminationParameters(;
        termination_reltol = ConcreteRNumber(termination_reltol),
        max_kkt_passes = ConcreteRNumber(max_kkt_passes),
        time_limit = time_limit
    )
end

function CoolPDLP.custom_to_rarray(algo::Algorithm{A}; kwargs...) where {A}
    conversion_r = custom_to_rarray(algo.conversion; kwargs...)
    preconditioning_r = custom_to_rarray(algo.preconditioning; kwargs...)
    step_size_r = custom_to_rarray(algo.step_size; kwargs...)
    restart_r = custom_to_rarray(algo.restart; kwargs...)
    generic_r = custom_to_rarray(algo.generic; kwargs...)
    termination_r = custom_to_rarray(algo.termination; kwargs...)
    return Algorithm{A}(
        conversion_r,
        preconditioning_r,
        step_size_r,
        restart_r,
        generic_r,
        termination_r,
    )
    return to_rarray(algo; kwargs...)
end

end
