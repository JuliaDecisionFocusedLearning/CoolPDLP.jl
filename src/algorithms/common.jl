abstract type AbstractState{T, V} end

"""
    preprocess(milp_init, sol_init, params)

Apply preconditioning, type conversion and device transfer to `milp_init` and `sol_init` for the algorithm defined by `params`.

Return a tuple `(milp, sol)`.
"""
function preprocess(
        milp_init_cpu::MILP,
        sol_init_cpu::PrimalDualSolution,
        params::Parameters,
    )
    # on CPU
    prec = pdlp_preconditioner(milp_init_cpu, params.preconditioning)
    milp_cpu = precondition(milp_init_cpu, prec)
    sol_cpu = precondition(sol_init_cpu, prec)

    # moving to GPU
    milp = to_device(milp_cpu, params.generic)
    sol = to_device(sol_cpu, params.generic)

    return milp, sol
end

"""
    initialize(milp, sol, params)

Initialize the appropriate state for solving `milp` starting from `sol` with the algorithm defined by `params`.
"""
function initialize end

"""
    solve(milp, sol, params; show_progress=true)
    solve(milp, params; show_progress=true)
    
Solve the continuous relaxation of `milp` starting from solution `sol` using the algorithm defined by `params`.

Return a couple `(sol, stats)` where `sol` is the last solution and `stats` contains convergence information.
"""
function solve(
        milp_init_cpu::MILP,
        sol_init_cpu::PrimalDualSolution,
        params::Parameters;
        show_progress::Bool = true,
    )
    starting_time = time()
    milp, sol = preprocess(milp_init_cpu, sol_init_cpu, params)
    state = initialize(milp, sol, params; starting_time)
    solve!(state, milp, params; show_progress)
    return get_solution(state, milp), state.stats
end

function solve(
        milp_init_cpu::MILP,
        params::Parameters;
        show_progress::Bool = true,
    )
    sol_init_cpu = PrimalDualSolution(zero(milp_init_cpu.lv), zero(milp_init_cpu.lc))
    return solve(milp_init_cpu, sol_init_cpu, params; show_progress)
end

"""
    solve!(state, milp, sol, params)

Modify `state` in-place to solve the continuous relaxation of `milp` starting from solution `sol` using the algorithm defined by `params`.
"""
function solve! end

function termination_check!(
        state::AbstractState,
        milp::MILP,
        params::Parameters
    )
    (; sol, scratch, stats) = state
    stats.time_elapsed = time() - stats.starting_time
    stats.err = kkt_errors!(scratch, sol, milp)
    if params.generic.record_error_history
        push!(stats.error_history, (stats.kkt_passes, stats.err))
    end
    stats.termination_status = termination_status(stats, params.termination)
    return nothing
end

function get_solution(state::AbstractState, milp::MILP)
    return unprecondition(state.sol, Preconditioner(milp))
end
