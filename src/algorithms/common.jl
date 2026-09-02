"""
    Algorithm

# Fields

$(TYPEDFIELDS)
"""
struct Algorithm{
        A,
        T <: Number,
        Ti <: Integer,
        M <: AbstractMatrix,
        B <: Backend,
        R <: RestartParameters{T},
        P <: Union{Nothing, AbstractPresolver},
    }
    conversion::ConversionParameters{T, Ti, M, B}
    preconditioning::PreconditioningParameters{T}
    step_size::StepSizeParameters{T}
    restart::R
    generic::GenericParameters
    termination::TerminationParameters{T}
    presolver::P
end

"""
    Algorithm{:ALGNAME}(
        # conversion
        _T::Type{T} = Float64,
        ::Type{Ti} = Int,
        ::Type{M} = SparseMatrixCSC;
        backend::B = CPU(),
        # preconditioning
        chambolle_pock_alpha = 1.0,
        ruiz_iter = 10,
        # step sizes
        invnorm_scaling = 0.9,
        primal_weight_damping = 0.5,
        zero_tol = 1.0e-8,
        spectral_norm_tol = 1.0e-3,
        spectral_norm_maxiter = 1000,
        # restart
        sufficient_decay = 0.2,
        necessary_decay = 0.8,
        artificial_decay = 0.36,
        restart_batch_aggregation = batched_mean,
        # generic
        show_progress = false,
        check_every = 100,
        record_error_history = true,
        # termination
        termination_reltol = 1.0e-4,
        max_kkt_passes = 10^5,
        time_limit = 100.0,
        # presolve
        presolver = nothing,
    )

Constructor for algorithm configs. `presolver` is `nothing` (presolve disabled) or an
[`AbstractPresolver`](@ref) instance, e.g. `presolver = PaPILOPresolver()` (`using PaPILO`
first).
"""
function Algorithm{A}(
        # conversion
        _T::Type{T} = Float64,
        ::Type{Ti} = Int,
        ::Type{M} = SparseMatrixCSC;
        backend::B = CPU(),
        # preconditioning
        chambolle_pock_alpha = 1.0,
        ruiz_iter = 10,
        # step sizes
        invnorm_scaling = 0.9,
        primal_weight_damping = 0.5,
        zero_tol = 1.0e-8,
        spectral_norm_tol = 1.0e-3,
        spectral_norm_maxiter = 1000,
        # restart
        sufficient_decay = 0.2,
        necessary_decay = 0.8,
        artificial_decay = 0.36,
        restart_batch_aggregation = batched_mean,
        # generic
        show_progress = false,
        check_every = 100,
        record_error_history = true,
        # termination
        termination_reltol = 1.0e-4,
        max_kkt_passes = 10^5,
        time_limit = 100.0,
        # presolve
        presolver::Union{Nothing, AbstractPresolver} = nothing,
    ) where {A, T, Ti, M, B}

    conversion = ConversionParameters(
        T, Ti, M; backend,
    )
    preconditioning = PreconditioningParameters(;
        chambolle_pock_alpha = _T(chambolle_pock_alpha),
        ruiz_iter
    )
    step_size = StepSizeParameters(;
        invnorm_scaling = _T(invnorm_scaling),
        primal_weight_damping = _T(primal_weight_damping),
        zero_tol = _T(zero_tol),
        spectral_norm_tol = _T(spectral_norm_tol),
        spectral_norm_maxiter,
    )
    restart = RestartParameters(;
        sufficient_decay = _T(sufficient_decay),
        necessary_decay = _T(necessary_decay),
        artificial_decay = _T(artificial_decay),
        batch_aggregation = restart_batch_aggregation,
    )
    generic = GenericParameters(;
        show_progress,
        check_every,
        record_error_history
    )
    termination = TerminationParameters(;
        termination_reltol = _T(termination_reltol),
        max_kkt_passes,
        time_limit
    )
    return Algorithm{A, T, Ti, M, B, typeof(restart), typeof(presolver)}(
        conversion,
        preconditioning,
        step_size,
        restart,
        generic,
        termination,
        presolver
    )
end

function Base.show(io::IO, algo::Algorithm{A}) where {A}
    (; conversion, preconditioning, step_size, restart, generic, termination, presolver) = algo
    return print(
        io, """
        $A algorithm:
        - $conversion
        - $preconditioning
        - $step_size
        - $restart
        - $generic
        - $termination
        - presolver=$presolver"""
    )
end

abstract type AbstractState{T, V} end

function prog_showvalues(state::AbstractState)
    err = state.stats.err
    (; primal, primal_scale, dual, dual_scale, gap, gap_scale) = err
    rel_primal = primal ./ primal_scale
    rel_dual = dual ./ dual_scale
    rel_gap = gap ./ gap_scale
    return (
        ("primal", progress_value(rel_primal)),
        ("dual", progress_value(rel_dual)),
        ("gap", progress_value(rel_gap)),
    )
end

"""
    progress_value(rel)

Format a relative error for the progress display: the value itself for a single instance, the maximum and mean over the instances for a batch.

The two summaries are printed in fixed width so that they line up across the progress rows.
"""
progress_value(rel::Number) = rel
function progress_value(rel::AbstractVector)
    return "max $(format_error(maximum(rel))), mean $(format_error(batched_mean(rel)))"
end

"""
    preprocess(milp_init, sol_init, algo)

Apply preconditioning, type conversion and device transfer to `milp_init` and `sol_init` for the algorithm defined by `algo`.

Return a tuple `(milp, sol)`.
"""
function preprocess(
        milp_init_cpu::MILP,
        sol_init_cpu::PrimalDualSolution,
        algo::Algorithm,
    )
    # on CPU
    prec = pdlp_preconditioner(milp_init_cpu, algo.preconditioning)
    milp_cpu = precondition(milp_init_cpu, prec)
    sol_cpu = precondition(sol_init_cpu, prec)

    # moving to GPU
    milp = perform_conversion(milp_cpu, algo.conversion)
    sol = perform_conversion(sol_cpu, algo.conversion)

    return milp, sol
end

"""
    initialize(milp, sol, algo)

Initialize the appropriate state for solving `milp` starting from `sol` with the algorithm defined by `algo`.
"""
function initialize end

"""
    solve(milp, sol, algo)
    solve(milp, algo)

Solve the continuous relaxation of `milp` starting from solution `sol` using the algorithm defined by `algo`.

Return a couple `(sol, stats)` where `sol` is the last solution and `stats` contains convergence information.
"""
function solve(
        milp_init_cpu::MILP,
        sol_init_cpu::PrimalDualSolution,
        algo::Algorithm
    )
    starting_time = time()
    milp, sol = preprocess(milp_init_cpu, sol_init_cpu, algo)
    state = initialize(milp, sol, algo; starting_time)
    (; c, lv, uv) = milp
    if nbcons(milp) == 0
        # with no constraint rows, the box-constrained optimum can be read off `c` and the
        # bounds directly, as long as the box is feasible and bounded in the direction `c`
        # pushes towards (otherwise fall through to the general loop below, same as any other
        # infeasible/unbounded problem: this package has no dedicated status for either, so it
        # relies on the iteration/time limit rather than early-exiting with a wrong `OPTIMAL`)
        box_feasible = all(lv .<= uv)
        bounded_below = !any(@. (c > 0) & isinf(lv))
        bounded_above = !any(@. (c < 0) & isinf(uv))
        if box_feasible && bounded_below && bounded_above
            @. sol.x = ifelse(c > 0, lv, ifelse(c < 0, uv, clamp(zero(eltype(lv)), lv, uv)))
            kkt_errors!(state.stats.err, state.scratch, sol, milp)
            state.stats.time_elapsed = time() - starting_time
            state.stats.termination_status = MOI.OPTIMAL
            return get_solution(state, milp), state.stats
        end
    end
    solve!(state, milp, algo)
    return get_solution(state, milp), state.stats
end

function solve(
        milp_init_cpu::MILP,
        algo::Algorithm{A, T, Ti, M, B, R, Nothing}
    ) where {A, T, Ti, M, B, R}
    sol_init_cpu = PrimalDualSolution(milp_init_cpu)
    return solve(milp_init_cpu, sol_init_cpu, algo)
end

# Method split to contain the impact of presolve-related type instabilities
@unstable function solve(
        milp_init_cpu::MILP,
        algo::Algorithm{A, T, Ti, M, B, R, P}
    ) where {A, T, Ti, M, B, R, P <: AbstractPresolver}
    isbatched(milp_init_cpu) && throw(ArgumentError("Presolve does not support batched MILPs"))
    milp_reduced, presolve_state = presolve(algo.presolver, milp_init_cpu)
    sol_init_reduced = PrimalDualSolution(milp_reduced)
    sol_reduced, stats = solve(milp_reduced, sol_init_reduced, algo)
    sol = postsolve(algo.presolver, presolve_state, sol_reduced)
    return sol, stats
end

"""
    solve!(state, milp, algo)

Modify `state` in-place to solve the continuous relaxation of `milp` using the algorithm defined by `algo`.
"""
function solve! end

function termination_check!(
        state::AbstractState,
        milp::MILP,
        algo::Algorithm
    )
    (; sol, scratch, stats) = state
    stats.time_elapsed = time() - stats.starting_time
    kkt_errors!(stats.err, scratch, sol, milp)
    if algo.generic.record_error_history
        push!(stats.error_history, (stats.kkt_passes, copy(stats.err)))
    end
    stats.termination_status = termination_status!!(scratch.b1, stats, algo.termination)
    return stats.termination_status !== MOI.OPTIMIZE_NOT_CALLED
end

function get_solution(state::AbstractState, milp::MILP)
    return unprecondition(state.sol, Preconditioner(milp))
end
