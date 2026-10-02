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

If `algo` has a presolver, `solve(milp, algo)` runs the algorithm on the reduced problem returned by [`presolve`](@ref), then maps its solution back to `milp` with [`postsolve`](@ref). In that case:

- `stats.err` holds the KKT errors of the returned solution on `milp`, and a solution which meets the tolerance on the reduced problem but not on `milp` is reported as `ALMOST_OPTIMAL` instead of `OPTIMAL`.
- `stats.kkt_passes` and `stats.error_history` describe the iterations on the reduced problem.
- `stats.time_elapsed` includes presolve and postsolve, but the time limit only applies to the iterations.

`solve(milp, sol, algo)` never presolves, whatever the presolver of `algo`: `sol` is a starting point for `milp`, not for a reduced problem.
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
        algo::Algorithm
    )
    if !isnothing(algo.presolver) && isbatched(milp_init_cpu)
        # `presolve` maps one problem to one problem, so a batch has no contract to rely on.
        # `algo.presolver`'s type is a type parameter of `algo`, so this test costs nothing.
        throw(ArgumentError("Presolve does not support batched MILPs"))
    end
    starting_time = time()
    # the reduced problem lives wherever the presolver put it (for a file-based backend like
    # `PaPILOPresolver`, a host `Float64` problem), and the inner `solve` preconditions and
    # converts it just like it would the original one. Without a presolver this is `milp_init_cpu`
    # itself, and every step below is likewise an identity
    milp_reduced, presolve_info = presolve(algo.presolver, milp_init_cpu)
    # `sol_reduced` solves the *reduced* problem, in the element and array types of
    # `algo.conversion` (the inner `solve` unpreconditions it on the way out, so its values are
    # in the reduced problem's own scale, not the preconditioned one)
    sol_reduced, stats = solve(milp_reduced, PrimalDualSolution(milp_reduced), algo)
    # `sol` solves the *original* problem, in those same types: that is the contract
    # `postsolve` implementations must respect
    sol = postsolve(algo.presolver, presolve_info, sol_reduced)
    regrade!(stats, algo.presolver, sol, milp_init_cpu, algo)
    stats.starting_time = starting_time
    stats.time_elapsed = time() - starting_time
    return sol, stats
end

"""
    regrade!(stats, presolver, sol, milp_init_cpu, algo)

Make `stats` grade `sol`, a postsolved solution, on the problem `milp_init_cpu` that the caller asked about rather than on the reduced problem that the algorithm iterated on.

Solving the reduced problem to `termination_reltol` does not guarantee that much on the original problem: postsolve reintroduces the eliminated rows and columns, and a presolver's dual reconstruction can be far off when it is handed an inexact solution to begin with. So the KKT errors are recomputed on `milp_init_cpu`, and an `OPTIMAL` status that they do not back up is demoted to `ALMOST_OPTIMAL`.

Without a presolver (`presolver = nothing`) there was no reduction, and `stats` is left untouched.
"""
function regrade!(
        stats::ConvergenceStats,
        ::AbstractPresolver,
        sol::PrimalDualSolution,
        milp_init_cpu::MILP,
        algo::Algorithm,
    )
    # `sol` has the shape of the original problem and the element and array types of
    # `algo.conversion`, which is what `postsolve` promises, so the original problem is converted
    # to match (but not preconditioned, since `sol` is not in a preconditioned scale)
    milp = perform_conversion(milp_init_cpu, algo.conversion)
    kkt_errors!(stats.err, Scratch(sol), sol, milp)
    solved = batched_all(<=(algo.termination.termination_reltol), relative(stats.err))
    if !solved && stats.termination_status === MOI.OPTIMAL
        stats.termination_status = MOI.ALMOST_OPTIMAL
    end
    return stats
end

function regrade!(
        stats::ConvergenceStats, ::Nothing, ::PrimalDualSolution, ::MILP, ::Algorithm
    )
    return stats
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
