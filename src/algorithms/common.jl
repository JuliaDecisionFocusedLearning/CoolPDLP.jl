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
    }
    conversion::ConversionParameters{T, Ti, M, B}
    preconditioning::PreconditioningParameters{T}
    step_size::StepSizeParameters{T}
    restart::RestartParameters{T}
    generic::GenericParameters
    termination::TerminationParameters{T}
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
        # restart
        sufficient_decay = 0.2,
        necessary_decay = 0.8,
        artificial_decay = 0.36,
        # generic
        show_progress = true,
        check_every = 100,
        record_error_history = true,
        # termination
        termination_reltol = 1.0e-4,
        max_kkt_passes = 10^5,
        time_limit = 100.0,
    )

Constructor for algorithm configs.
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
        # restart
        sufficient_decay = 0.2,
        necessary_decay = 0.8,
        artificial_decay = 0.36,
        # generic
        show_progress = true,
        check_every = 100,
        record_error_history = true,
        # termination
        termination_reltol = 1.0e-4,
        max_kkt_passes = 10^5,
        time_limit = 100.0,
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
    )
    restart = RestartParameters(;
        sufficient_decay = _T(sufficient_decay),
        necessary_decay = _T(necessary_decay),
        artificial_decay = _T(artificial_decay),
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

    return Algorithm{A, T, Ti, M, B}(
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
    rel_primal = @sprintf("%.3e", primal / primal_scale)
    rel_dual = @sprintf("%.3e", dual / dual_scale)
    rel_gap = @sprintf("%.3e", gap / gap_scale)
    return (
        ("primal", rel_primal),
        ("dual", rel_dual),
        ("gap", rel_gap),
    )
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
        algo::Algorithm;
        show_progress::Bool = true,
    )
    starting_time = time()
    milp, sol = preprocess(milp_init_cpu, sol_init_cpu, algo)
    state = initialize(milp, sol, algo; starting_time)
    solve!(state, milp, algo)
    return get_solution(state, milp), state.stats
end

function solve(
        milp_init_cpu::MILP,
        algo::Algorithm
    )
    sol_init_cpu = PrimalDualSolution(zero(milp_init_cpu.lv), zero(milp_init_cpu.lc))
    return solve(milp_init_cpu, sol_init_cpu, algo)
end

"""
    solve!(state, milp, sol, algo)

Modify `state` in-place to solve the continuous relaxation of `milp` starting from solution `sol` using the algorithm defined by `algo`.
"""
function solve! end

function termination_check!(
        state::AbstractState,
        milp::MILP,
        algo::Algorithm
    )
    (; sol, scratch, stats) = state
    stats.time_elapsed = time() - stats.starting_time
    stats.err = kkt_errors!(scratch, sol, milp)
    if algo.generic.record_error_history
        push!(stats.error_history, (stats.kkt_passes, stats.err))
    end
    stats.termination_status = termination_status(stats, algo.termination)
    return stats.termination_status !== STILL_RUNNING
end

function get_solution(state::AbstractState, milp::MILP)
    return unprecondition(state.sol, Preconditioner(milp))
end
