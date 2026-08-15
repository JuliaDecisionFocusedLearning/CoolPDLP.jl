module CoolPDLPReactantExt

using CoolPDLP:
    CoolPDLP,
    ConvergenceStats,
    KKTErrors,
    MILP,
    PDHGState,
    PrimalDualSolution,
    Scratch,
    StepSizes,
    custom_to_rarray
using Reactant: ConcreteRArray, to_rarray

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

end
