"""
    Algorithm

# Fields

$(TYPEDFIELDS)
"""
struct Algorithm{
        A,
        C <: ConversionParameters,
        P <: PreconditioningParameters,
        S <: StepSizeParameters,
        R <: RestartParameters,
        G <: GenericParameters,
        T <: TerminationParameters,
    }
    conversion::C
    preconditioning::P
    step_size::S
    restart::R
    generic::G
    termination::T

    function Algorithm{A}(
            conversion::C,
            preconditioning::P,
            step_size::S,
            restart::R,
            generic::G,
            termination::T,
        ) where {A, C, P, S, R, G, T}
        return new{A, C, P, S, R, G, T}(
            conversion,
            preconditioning,
            step_size,
            restart,
            generic,
            termination,
        )
    end
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
        geometric_mean_iter = 0,
        curtis_reid_iter = 0,
        bound_objective_rescaling = false,
        # step sizes
        invnorm_scaling = 0.9,
        primal_weight_damping = 0.5,
        zero_tol = 1.0e-8,
        spectral_norm_tol = 1.0e-3,
        spectral_norm_maxiter = 1000,
        reflection_coefficient = 1.0,
        pid_kp = 0.99,
        pid_ki = 0.01,
        pid_kd = 0.0,
        pid_integral_discount = 0.3,
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
        termination_abstol = termination_reltol,
        termination_norm = L2Norm(),
        max_kkt_passes = 10^5,
        time_limit = Inf,
        infeasibility_tol = DEFAULT_INFEASIBILITY_TOL,
    )

Constructor for algorithm configs.

The algorithm stops when, for every instance, the KKT errors meet
`error ≤ termination_abstol + termination_reltol * scale` (see
[`termination_margin!!`](@ref)), with the residuals measured in the norm `termination_norm`
(a [`TerminationNorm`](@ref)). The default `termination_abstol = termination_reltol`
is the purely relative criterion `error / (1 + scale) ≤ termination_reltol`.

With `infeasibility_tol` (default `1e-8`, as in PDLP), PDHG and PDLP look for infeasibility
certificates at every termination check, and stop with `MOI.INFEASIBLE` or
`MOI.DUAL_INFEASIBLE` once a ray's error is below it (see [`infeasibility_check!`](@ref)). These
checks cost two KKT passes each, which do not count towards `max_kkt_passes`.
`infeasibility_tol = nothing` removes the detection altogether.

The preconditioning stages and the bound and objective rescaling are described on the math page
of the documentation.
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
        geometric_mean_iter = 0,
        curtis_reid_iter = 0,
        bound_objective_rescaling = false,
        # step sizes
        invnorm_scaling = 0.9,
        primal_weight_damping = 0.5,
        zero_tol = 1.0e-8,
        spectral_norm_tol = 1.0e-3,
        spectral_norm_maxiter = 1000,
        reflection_coefficient = 1.0,
        pid_kp = 0.99,
        pid_ki = 0.01,
        pid_kd = 0.0,
        pid_integral_discount = 0.3,
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
        termination_abstol = termination_reltol,
        termination_norm = L2Norm(),
        max_kkt_passes = 10^5,
        time_limit = Inf,
        infeasibility_tol = DEFAULT_INFEASIBILITY_TOL,
    ) where {A, T, Ti, M, B}

    conversion = ConversionParameters(
        T, Ti, M; backend,
    )
    preconditioning = PreconditioningParameters(;
        chambolle_pock_alpha = _T(chambolle_pock_alpha),
        ruiz_iter,
        geometric_mean_iter,
        curtis_reid_iter,
        bound_objective_rescaling,
    )
    step_size = StepSizeParameters(;
        invnorm_scaling = _T(invnorm_scaling),
        primal_weight_damping = _T(primal_weight_damping),
        zero_tol = _T(zero_tol),
        spectral_norm_tol = _T(spectral_norm_tol),
        spectral_norm_maxiter,
        reflection_coefficient = _T(reflection_coefficient),
        pid_kp = _T(pid_kp),
        pid_ki = _T(pid_ki),
        pid_kd = _T(pid_kd),
        pid_integral_discount = _T(pid_integral_discount),
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
        record_error_history,
        time_limit,
        termination_norm,
    )
    termination = TerminationParameters(;
        termination_reltol = _T(termination_reltol),
        termination_abstol = _T(termination_abstol),
        max_kkt_passes,
        infeasibility_tol = convert_tol(_T, infeasibility_tol),
    )

    return Algorithm{A}(
        conversion,
        preconditioning,
        step_size,
        restart,
        generic,
        termination
    )
end

function Base.show(io::IO, algo::Algorithm{A}) where {A}
    (; conversion, preconditioning, step_size, restart, generic, termination) = algo
    return print(
        io, """
        $A algorithm:
        - $conversion
        - $preconditioning
        - $step_size
        - $restart
        - $generic
        - $termination"""
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
@stable default_union_limit = 2 function preprocess(
        milp_init_cpu::MILP,
        sol_init_cpu::PrimalDualSolution,
        algo::Algorithm,
    )
    # on CPU
    prec = pdlp_preconditioner(milp_init_cpu, algo.preconditioning)
    milp_cpu = precondition(milp_init_cpu, prec)
    sol_cpu = precondition(sol_init_cpu, prec)
    # per-instance factors turn shared variable bounds into batched ones, hence a small union
    if algo.preconditioning.bound_objective_rescaling
        β, γ = bound_objective_factors(milp_cpu)
        milp_cpu = rescale_bounds_objective(milp_cpu, β, γ)
        sol_cpu = rescale_bounds_objective(sol_cpu, β, γ)
    end

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
    starting_time = current_time()
    milp, sol = preprocess(milp_init_cpu, sol_init_cpu, algo)
    state = initialize(milp, sol, algo; starting_time)
    trivial_solve = (
        try_solve_crossed_bounds!(state, milp_init_cpu) ||
            (nbcons(milp) == 0 && try_solve_noconstraints!(state, milp, algo))
    )
    if !trivial_solve
        solve!(state, milp, algo)
    end
    return get_solution(state, milp), state.stats
end

"""
    convert_tol(T, tol)

Convert an optional tolerance to `T`, keeping `nothing` as is.
"""
convert_tol(::Type{T}, tol::Number) where {T} = convert(T, tol)
convert_tol(::Type, ::Nothing) = nothing

function try_solve_noconstraints!(state::AbstractState, milp::MILP, algo::Algorithm)
    (; c, lv, uv) = milp
    (; sol) = state
    # with no constraint rows, the box-constrained optimum can be read off `c` and the
    # bounds directly, as long as the box is feasible and bounded in the direction `c`
    # pushes towards (otherwise fall through to the general loop: crossed bounds have already
    # been reported by `try_solve_crossed_bounds!`, and an unbounded direction is certified by
    # the infeasibility detection)
    box_feasible = all(lv .<= uv)
    bounded_below = !any(@. (c > 0) & isinf(lv))
    bounded_above = !any(@. (c < 0) & isinf(uv))
    if box_feasible && bounded_below && bounded_above
        @. sol.x = ifelse(c > 0, lv, ifelse(c < 0, uv, clamp(zero(eltype(lv)), lv, uv)))
        kkt_errors!(state.stats.err, state.scratch, sol, milp, algo.generic.termination_norm)
        state.stats.time_elapsed = current_time() - state.stats.starting_time
        state.stats.termination_status_code = status_code(MOI.OPTIMAL)
        state.stats.instance_status_code = batched_fill(
            state.stats.instance_status_code, status_code(MOI.OPTIMAL)
        )
        return true
    else
        return false
    end
end

"""
    crossed_bounds(milp)

Return whether some lower bound of `milp` exceeds the matching upper bound, for each instance.

Such an instance is trivially infeasible, but its ray certificate would need separate
multipliers for the lower and upper bounds, which [`infeasibility_check!`](@ref) does not have.
"""
function crossed_bounds(milp::MILP)
    (; lv, uv, lc, uc) = milp
    crossed_v = vec(any(lv .> uv; dims = 1))
    crossed_c = vec(any(lc .> uc; dims = 1))
    crossed = crossed_v .| crossed_c
    return isbatched(milp) ? Array(crossed) : only(crossed)
end

"""
    try_solve_crossed_bounds!(state, milp_cpu)

Terminate with `MOI.INFEASIBLE` right away if every instance of `milp_cpu` has crossed bounds,
see [`crossed_bounds`](@ref). No ray certificate is available in that case.

When only some instances of a batch have crossed bounds, those are marked `MOI.INFEASIBLE` and
frozen (see [`freeze_terminated!`](@ref)) before the batch is solved as usual, so that they
neither keep the batch running until its limit nor end up with a limit status.
"""
function try_solve_crossed_bounds!(state::AbstractState, milp_cpu::MILP)
    crossed = crossed_bounds(milp_cpu)
    if !batched_all(identity, crossed)
        mark_crossed_instances!(state.stats, crossed)
        return false
    end
    state.stats.time_elapsed = current_time() - state.stats.starting_time
    state.stats.termination_status_code = status_code(MOI.INFEASIBLE)
    state.stats.instance_status_code = batched_fill(
        state.stats.instance_status_code, status_code(MOI.INFEASIBLE)
    )
    return true
end

"""
    mark_crossed_instances!(stats, crossed)

Give the instances of a batch flagged in `crossed` the status `MOI.INFEASIBLE` and freeze them.
Does nothing without batching, or when no instance is flagged.
"""
mark_crossed_instances!(::ConvergenceStats, ::Bool) = nothing
function mark_crossed_instances!(stats::ConvergenceStats, crossed::AbstractVector{Bool})
    any(crossed) || return nothing
    codes = Array(stats.instance_status_code)
    codes[crossed] .= status_code(MOI.INFEASIBLE)
    copyto!(stats.instance_status_code, codes)
    freeze_terminated!(stats)
    return nothing
end

function solve(
        milp_init_cpu::MILP,
        algo::Algorithm
    )
    sol_init_cpu = PrimalDualSolution(milp_init_cpu)
    return solve(milp_init_cpu, sol_init_cpu, algo)
end

"""
    solve!(state, milp, algo)

Modify `state` in-place to solve the continuous relaxation of `milp` using the algorithm defined by `algo`.

Unlike [`solve`](@ref), `solve!` does not look for crossed bounds (some `l > u`, see
[`crossed_bounds`](@ref)): that check reads the bounds on the host, so it cannot run inside a
loop compiled by Reactant. An instance with crossed bounds has no ray certificate, so `solve!`
alone runs it to an iteration or time limit. When calling `solve!` directly (e.g. to compile it),
call `CoolPDLP.try_solve_crossed_bounds!(state, milp)` first, outside the compiled function, and
skip `solve!` when it returns `true`: it then marks the state `MOI.INFEASIBLE`, and otherwise
freezes the instances of a batch which have crossed bounds.
"""
function solve! end

function termination_check!(
        state::AbstractState,
        milp::MILP,
        algo::Algorithm
    )
    (; sol, scratch, stats) = state
    update_time_elapsed!(stats, algo.generic.time_limit)
    kkt_errors!(stats.err, scratch, sol, milp, algo.generic.termination_norm)
    stats.extra_products += 2
    refine_kkt_errors!(state, milp, algo.generic.termination_norm)
    # dispatches to a no-op when the detection is disabled (`state.infeas === nothing`): that
    # is a type-level choice, so it costs nothing and is resolved at trace time under Reactant
    infeasibility_check!(
        stats, state.infeas, scratch, ray_candidates(state)..., milp,
        algo.termination.infeasibility_tol, active_mask(stats),
    )
    record_error_history!(stats, algo.generic.record_error_history)
    return set_termination_status!!(stats, scratch.b1, algo.termination, algo.generic.time_limit)
end

"""
    refine_kkt_errors!(state, milp, norm)

Improve the KKT errors just computed by [`termination_check!`](@ref) with an algorithm-specific
candidate. Does nothing by default.
"""
refine_kkt_errors!(::AbstractState, ::MILP, ::TerminationNorm) = nothing

"""
    record_error_history!(stats, record)

Append a snapshot of the current errors to `stats.error_history` when `record` is `true`.

Skipped inside a Reactant compilation context, where the history cannot be grown.
"""
function record_error_history!(stats::ConvergenceStats, record)
    within_compile() && return nothing
    if record
        push!(stats.error_history, (stats.kkt_passes, copy(stats.err)))
    end
    return nothing
end

"""
    NoProgress

Placeholder for the progress bar inside a Reactant compilation context.

It is a dedicated singleton rather than `nothing` because the traced loops capture it, and
ReactantCore 0.1.27 treats a captured `nothing` as an undefined variable
(https://github.com/EnzymeAD/Reactant.jl/pull/3507).
"""
struct NoProgress end

"""
    init_progress(desc, show_progress)

Build a progress bar, or a [`NoProgress`](@ref) placeholder inside a Reactant compilation context, where it cannot be updated.
"""
function init_progress(desc::String, show_progress)
    within_compile() && return NoProgress()
    return ProgressUnknown(; desc, enabled = show_progress)
end

"""
    next_progress!(prog, state)

Advance the progress bar built by [`init_progress`](@ref), unless it was skipped.
"""
next_progress!(prog, state::AbstractState) = next!(prog; showvalues = () -> prog_showvalues(state))
next_progress!(::NoProgress, ::AbstractState) = nothing

"""
    finish_progress!(prog)

Close the progress bar built by [`init_progress`](@ref), unless it was skipped.
"""
finish_progress!(prog) = finish!(prog)
finish_progress!(::NoProgress) = nothing

function get_solution(state::AbstractState, milp::MILP)
    (; bound_scale, obj_scale) = milp
    sol = rescale_bounds_objective(state.sol, inv.(bound_scale), inv.(obj_scale))
    return unprecondition(sol, Preconditioner(milp))
end
