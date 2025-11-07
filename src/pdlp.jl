"""
    PDLPParameters

Parameters for configuration of the baseline primal-dual hybrid gradient.
    
# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PDLPParameters{Tv <: Number} <: AbstractParameters{Tv}
    "tolerance when checking KKT relative errors to decide termination"
    termination_reltol::Tv = 1.0e-4
    "scaling of the inverse spectral norm of `K` when defining the step size"
    stepsize_scaling::Tv = typeof(termination_reltol)(0.9)
    "norm parameter in the Chambolle-pock preconditioner"
    preconditioner_chambollepock_alpha::Tv = typeof(termination_reltol)(1)
    "iteration parameter in the Ruiz preconditioner"
    preconditioner_ruiz_iterations::Int = 10
    "restart criterion: sufficient decay in normalized duality gap"
    β_sufficient::Tv = typeof(termination_reltol)(0.2)
    "restart criterion: necessary decay"
    β_necessary::Tv = typeof(termination_reltol)(0.8)
    "restart criterion: long inner loop"
    β_artificial::Tv = typeof(termination_reltol)(0.36)
    "maximum number of multiplications by both the KKT matrix `K` and its transpose `Kᵀ`"
    max_kkt_passes::Int = 100_000
    "time limit in seconds"
    time_limit::Float64 = 100.0
    "frequency of termination checks"
    check_every::Int = 40
    "whether or not to record error evolution"
    record_error_history::Bool = false
end

"""
    PDLPState

Current solution, step sizes and various buffers / metrics for the baseline primal-dual hybrid gradient.

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct PDLPState{
        Tv <: Number, V <: AbstractVector{Tv},
    } <: AbstractState{Tv, V}
    # solution
    "current primal solution"
    const x::V
    "current dual solution"
    const y::V
    # step sizes
    "step size"
    η::Tv
    "primal weight"
    ω::Tv = one(η)
    "sum of step sizes since the last restart"
    η_sum::Tv = zero(η)
    # restart stuff
    "current primal solution average"
    const x_avg::V = copy(x)
    "current dual solution average"
    const y_avg::V = copy(y)
    "current primal restart candiate"
    const x_restart_cand::V = copy(x)
    "current dual restart candidate"
    const y_restart_cand::V = copy(y)
    "previous primal restart candiate"
    const x_prev_restart_cand::V = copy(x)
    "previous dual restart candidate"
    const y_prev_restart_cand::V = copy(y)
    "primal solution at last restart"
    const x_last_restart::V = copy(x)
    "dual solution at last restart"
    const y_last_restart::V = copy(y)
    # scratch spaces
    "buffer in the shape of `x`"
    const x_scratch1::V = copy(x)
    "buffer in the shape of `x`"
    const x_scratch2::V = copy(x)
    "buffer in the shape of `x`"
    const x_scratch3::V = copy(x)
    "buffer in the shape of `y`"
    const y_scratch::V = copy(y)
    # counters
    "number of outer iterations (`n`)"
    outer_iterations::Int = 0
    "number of inner iterations (`t`)"
    inner_iterations::Int = 0
    "total number of iterations (`k`)"
    total_iterations::Int = 0
    "time at which the algorithm started, in seconds"
    starting_time::Float64 = time()
    "time elapsed since the algorithm started, in seconds"
    time_elapsed::Float64 = 0.0
    "number of multiplications by both the KKT matrix and its transpose"
    kkt_passes::Int = 0
    "current relative KKT error"
    relative_error::Tv = typemax(eltype(x))
    # termination and history
    "termination reason (should be `STILL_RUNNING` until the algorithm actuall terminates)"
    termination_reason::TerminationReason = STILL_RUNNING
    "history of relative KKT errors, indexed by number of KKT passes"
    const relative_error_history::Vector{Tuple{Int, Tv}} = Tuple{Int, eltype(x)}[]
end

"""
    pdlp(
        milp::MILP,
        params::PDLPParameters,
        x_init::AbstractVector=zero(milp.c);
        show_progress::Bool=true
    )
    
Apply the primal-dual hybrid gradient algorithm to solve the continuous relaxation of `milp` using configuration `params`, starting from primal variable `x_init`.
"""
function pdlp(
        milp::MILP{Tv, V},
        params::PDLPParameters{Tv},
        x_init::V = zero(milp.c);
        show_progress::Bool = true
    ) where {Tv, V}
    starting_time = time()
    sad = SaddlePointProblem(milp)
    y_init = zero(sad.q)
    return pdlp(sad, params, x_init, y_init; show_progress, starting_time)
end

"""
    pdlp(
        sad::SaddlePointProblems,
        params::PDLPParameters,
        x_init::AbstractVector=zero(sad.c);
        y_init::AbstractVector=zero(sad.q);
        show_progress::Bool=true
    )
    
Apply the primal-dual hybrid gradient algorithm to solve the saddle-point problem `sad` using configuration `params`, starting from `(x_init, y_init)`.

Return a triplet `(x, y, state)` where `x` is the primal solution, `y` is the dual solution and `state` is the algorithm's final state, including convergence information.
"""
function pdlp(
        sad::SaddlePointProblem{Tv, V},
        params::PDLPParameters{Tv},
        x_init::V = zero(sad.c),
        y_init::V = zero(sad.q);
        show_progress::Bool = true,
        starting_time::Float64 = time()
    ) where {Tv, V}
    x, y = copy(x_init), copy(y_init)
    sad = precondition_pdlp(
        sad;
        ruiz_iterations = params.preconditioner_ruiz_iterations,
        chambollepock_alpha = params.preconditioner_chambollepock_alpha,
    )
    η = fixed_stepsize(sad, params)
    state = PDLPState(; x, y, η, starting_time)
    prog = ProgressUnknown(desc = "PDLP iterations:", enabled = show_progress)
    must_restart = false
    must_terminate = false
    try
        while true
            while true
                yield()
                for _ in 1:params.check_every
                    next!(prog; showvalues = (("relative_error", state.relative_error),))
                    pdlp_step!(state, sad)
                    finish_inner_iteration!(state, sad)
                end
                must_restart = restart_check!(state, sad, params)
                must_terminate = termination_check!(state, sad, params)
                if must_restart || must_terminate
                    break
                end
            end
            state.outer_iterations += 1
            if must_restart
                restart!(state)
            elseif must_terminate
                break
            end
        end
    catch e
        if e isa InterruptException
            finish!(prog)
        else
            rethrow(e)
        end
    end
    return primal_solution(state, sad), dual_solution(state, sad), state
end

function fixed_stepsize(
        sad::SaddlePointProblem{Tv},
        params::PDLPParameters{Tv}
    ) where {Tv}
    (; K, Kᵀ) = sad
    (; stepsize_scaling) = params
    η = Tv(stepsize_scaling) * inv(spectral_norm(K, Kᵀ))
    return η
end

function pdlp_step!(
        state::PDLPState{Tv, V},
        sad::SaddlePointProblem{Tv, V},
    ) where {Tv, V}
    (; x, y, η, ω, x_scratch1, y_scratch) = state
    (; c, q, K, Kᵀ, l, u, ineq_cons) = sad

    xp, yp = x_scratch1, y_scratch
    τ, σ = η / ω, η * ω

    # xp = proj_X(x - τ * (c - Kᵀ * y))
    xp .= x .- τ .* c
    mul!(xp, Kᵀ, y, τ, true)
    proj_X!(xp, l, u)

    # yp = proj_Y(y + σ * (q - K * (2 * xp - x)))
    yp .= y .+ σ .* q
    x .= 2 .* xp .- x
    mul!(yp, K, x, -σ, true)
    proj_Y!(yp, ineq_cons)

    copyto!(x, xp)
    copyto!(y, yp)

    state.kkt_passes += 1
    return nothing
end

function restart!(state::PDLPState)
    (; x, y, x_restart_cand, y_restart_cand, x_last_restart, y_last_restart) = state
    copyto!(x, x_restart_cand)
    copyto!(y, y_restart_cand)
    copyto!(x_last_restart, x)
    copyto!(y_last_restart, y)
    state.inner_iterations = 0
    return nothing
end

function finish_inner_iteration!(
        state::PDLPState,
        sad::SaddlePointProblem,
    )
    (;
        η, ω, η_sum,
        x, y,
        x_avg, y_avg,
        x_restart_cand, y_restart_cand,
        x_prev_restart_cand, y_prev_restart_cand,
    ) = state

    η_sum += η
    @. x_avg = (η * x + (η_sum - η) * x_avg) / η_sum
    @. y_avg = (η * y + (η_sum - η) * y_avg) / η_sum
    state.η_sum = η_sum

    err = aggregated_absolute_kkt_error!(state, sad, x, y, ω)
    err_avg = aggregated_absolute_kkt_error!(state, sad, x_avg, y_avg, ω)

    copyto!(x_prev_restart_cand, x_restart_cand)
    copyto!(y_prev_restart_cand, y_restart_cand)
    if err < err_avg
        copyto!(x_restart_cand, x)
        copyto!(y_restart_cand, y)
    else
        copyto!(x_restart_cand, x_avg)
        copyto!(y_restart_cand, y_avg)
    end

    state.inner_iterations += 1
    state.total_iterations += 1

    return nothing
end

function restart_check!(
        state::PDLPState,
        sad::SaddlePointProblem,
        params::PDLPParameters
    )
    (;
        ω,
        x_restart_cand, y_restart_cand,
        x_prev_restart_cand, y_prev_restart_cand,
        x_last_restart, y_last_restart,
        inner_iterations, total_iterations,
    ) = state
    (; β_sufficient, β_necessary, β_artificial) = params

    err_restart_cand = aggregated_absolute_kkt_error!(
        state, sad, x_restart_cand, y_restart_cand, ω
    )
    err_prev_restart_cand = aggregated_absolute_kkt_error!(
        state, sad, x_prev_restart_cand, y_prev_restart_cand, ω
    )
    err_last_restart = aggregated_absolute_kkt_error!(
        state, sad, x_last_restart, y_last_restart, ω
    )

    sufficient_decay = err_restart_cand <= β_sufficient * err_last_restart
    necessary_decay = err_restart_cand <= β_necessary * err_last_restart
    no_local_progress = err_restart_cand > err_prev_restart_cand
    long_inner_loop = inner_iterations >= β_artificial * total_iterations
    return false
    return sufficient_decay || (necessary_decay && no_local_progress) || long_inner_loop
end
