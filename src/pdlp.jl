"""
    PDLPParameters

Parameters for configuration of the baseline primal-dual hybrid gradient.
    
# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PDLPParameters{T <: Number} <: AbstractParameters{T}
    "whether to enable restarts"
    enable_restarts::Bool = true
    "whether to enable scaling"
    enable_scaling::Bool = true
    # step size parameters
    "scaling of the inverse spectral norm of `K` when defining the step size"
    stepsize_scaling::T = 0.9
    # preconditioning parameters
    "norm parameter in the Chambolle-pock preconditioner"
    preconditioner_chambollepock_alpha::T = 1.0
    "iteration parameter in the Ruiz preconditioner"
    preconditioner_ruiz_iterations::Int = enable_scaling ? 10 : 0
    # restart parameters
    "restart criterion: sufficient decay in normalized duality gap"
    β_sufficient::T = enable_restarts ? 0.2 : NaN
    "restart criterion: necessary decay"
    β_necessary::T = enable_restarts ? 0.8 : NaN
    "restart criterion: long inner loop"
    β_artificial::T = enable_restarts ? 0.36 : NaN
    # termination parameters
    "tolerance on KKT relative errors to decide termination"
    termination_reltol::T = 1.0e-4
    "maximum number of multiplications by both the KKT matrix `K` and its transpose `Kᵀ`"
    max_kkt_passes::Int = 100_000
    "time limit in seconds"
    time_limit::Float64 = 100.0
    # meta parameters
    "frequency of restart or termination checks"
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
        T <: Number, V <: AbstractVector{T},
    } <: AbstractState{T, V}
    # solutions
    const z::PrimalDualSolution{T, V}
    const z_avg::PrimalDualSolution{T, V} = copy(z)
    const z_restart_candidate::PrimalDualSolution{T, V} = copy(z)
    const z_prev_restart_candidate::PrimalDualSolution{T, V} = copy(z)
    const z_last_restart::PrimalDualSolution{T, V} = copy(z)
    # step sizes
    "step size"
    η::T
    "primal weight"
    ω::T = one(η)
    "sum of step sizes since the last restart"
    η_sum::T = zero(η)
    # scratch spaces
    err_primal_scratch::V = zero(z.y)
    err_dual_scratch::V = zero(z.x)
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
    # termination and history
    "termination reason (should be `STILL_RUNNING` until the algorithm actually terminates)"
    termination_reason::TerminationReason = STILL_RUNNING
    "history of KKT errors, indexed by number of KKT passes"
    const error_history::Vector{Tuple{Int, KKTErrors{T}}} = Tuple{Int, KKTErrors{eltype(z)}}[]
end

function Base.show(io::IO, state::PDLPState)
    (; z, time_elapsed, kkt_passes, termination_reason) = state
    return print(
        io,
        @sprintf(
            "%s with termination reason %s: %.6e relative KKT error after %g seconds elapsed and %s KKT passes",
            nameof(typeof(state)),
            termination_reason,
            z.err.max_rel_err,
            time_elapsed,
            kkt_passes,
        )
    )
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
        milp::MILP{T, V},
        params::PDLPParameters{T},
        x_init::V = zero(milp.c);
        show_progress::Bool = true
    ) where {T, V}
    starting_time = time()
    sad = SaddlePointProblem(milp)
    # TODO: handle preconditioning
    y_init = zero(sad.q)
    z_init = PrimalDualSolution(x_init, y_init)
    return pdlp(sad, params, z_init; show_progress, starting_time)
end

"""
    pdlp(
        sad::SaddlePointProblems,
        params::PDLPParameters,
        z_init::PrimalDualSolution;
        show_progress::Bool=true
    )
    
Apply the primal-dual hybrid gradient algorithm to solve the saddle-point problem `sad` using configuration `params`, starting from `z_init = (x_init, y_init)`.

Return a triplet `(x, y, state)` where `x` is the primal solution, `y` is the dual solution and `state` is the algorithm's final state, including convergence information.
"""
function pdlp(
        sad::SaddlePointProblem{T, V},
        params::PDLPParameters{T},
        x_init::V = zero(sad.c),
        y_init::V = zero(sad.q);
        show_progress::Bool = true,
        starting_time::Float64 = time()
    ) where {T, V}
    sad = precondition_pdlp(
        sad;
        ruiz_iterations = params.preconditioner_ruiz_iterations,
        chambollepock_alpha = params.preconditioner_chambollepock_alpha,
    )
    x, y = preconditioned_solution(sad, x_init, y_init)
    η = fixed_stepsize(sad, params)
    z = PrimalDualSolution(sad, x, y)
    state = PDLPState(; z, η, starting_time)
    push!(state.error_history, (0, kkt_errors!(state, sad, z)))
    prog = ProgressUnknown(desc = "PDLP iterations:", enabled = show_progress)
    must_restart = false
    must_terminate = false
    try
        while true
            while true
                yield()
                for _ in 1:params.check_every
                    next!(prog; showvalues = (("relative_error", state.z.err.max_rel_err),))
                    pdlp_step!(state, sad)
                    update_average!(state, sad)
                    update_restart_candidate!(state)
                    state.inner_iterations += 1
                    state.total_iterations += 1
                end
                must_restart = restart_check(state, params)
                must_terminate = termination_check(state, params)
                if must_restart || must_terminate
                    break
                end
            end
            state.outer_iterations += 1
            if must_restart
                restart!(state)
                state.inner_iterations = 0
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
    return unpreconditioned_solution(sad, state.z.x, state.z.y), state
end

function fixed_stepsize(
        sad::SaddlePointProblem{T},
        params::PDLPParameters{T}
    ) where {T}
    (; K, Kᵀ) = sad
    (; stepsize_scaling) = params
    η = T(stepsize_scaling) * inv(spectral_norm(K, Kᵀ))
    return η
end

function proj_λ⁺(λ::T, l::T) where {T <: Number}
    return ifelse(l == typemin(T), zero(T), positive_part(λ))
end

function proj_λ⁻(λ::T, u::T) where {T <: Number}
    return ifelse(u == typemax(T), zero(T), negative_part(λ))
end

function kkt_errors!(
        state::PDLPState{T, V},
        sad::SaddlePointProblem{T, V},
        z::PrimalDualSolution{T, V},
    ) where {T, V}
    (; x, y, Kx, Kᵀy, λ, λ⁺, λ⁻) = z
    (; ω, err_primal_scratch, err_dual_scratch) = state
    (; c, q, l_noinf, u_noinf, ineq_cons) = sad

    qᵀy = dot(q, y)
    cᵀx = dot(c, x)
    lᵀλ⁺ = dot(l_noinf, λ⁺)
    uᵀλ⁻ = dot(u_noinf, λ⁻)

    @. err_primal_scratch = ifelse(ineq_cons, positive_part(q - Kx), q - Kx)
    err_primal = norm(err_primal_scratch)
    err_primal_scale = one(T) + norm(q)

    @. err_dual_scratch = c - Kᵀy - λ
    err_dual = norm(err_dual_scratch)
    err_dual_scale = one(T) + norm(c)

    err_gap = abs(qᵀy + lᵀλ⁺ - uᵀλ⁻ - cᵀx)
    err_gap_scale = one(T) + abs(qᵀy + lᵀλ⁺ - uᵀλ⁻) + abs(cᵀx)

    weighted_aggregate_err = sqrt(ω^2 * err_primal^2 + inv(ω^2) * err_dual^2 + err_gap^2)

    rel_err_primal = err_primal / err_primal_scale
    rel_err_dual = err_dual / err_dual_scale
    rel_err_gap = err_gap / err_gap_scale
    max_rel_err = max(rel_err_primal, rel_err_dual, rel_err_gap)

    return KKTErrors(;
        err_primal,
        err_dual,
        err_gap,
        err_primal_scale,
        err_dual_scale,
        err_gap_scale,
        weighted_aggregate_err,
        max_rel_err,
    )
end

function pdlp_step!(state::PDLPState{T, V}, sad::SaddlePointProblem{T, V}) where {T, V}
    (; z, η, ω) = state
    (; c, q, K, Kᵀ, l, u, ineq_cons) = sad
    (; x, y, Kx, Kᵀy, λ, λ⁺, λ⁻) = z

    τ, σ = η / ω, η * ω

    # xp = proj_X(x - τ * (c - Kᵀ * y))
    @. x -= τ * (c - Kᵀy)
    @. x = proj_box(x, l, u)

    # yp = proj_Y(y + σ * (q - K * (2 * xp - x)))
    @. y += σ * (q + Kx)
    mul!(Kx, K, x)
    @. y -= σ * 2 * Kx
    @. y = ifelse(ineq_cons, positive_part(y), y)
    mul!(Kᵀy, Kᵀ, y)

    # λ = proj_Λ(c - Kᵀy)
    @. λ = proj_λ(c - Kᵀy, l, u)
    @. λ⁺ = positive_part(λ)
    @. λ⁻ = negative_part(λ)

    # update errors
    z.err = kkt_errors!(state, sad, z)

    state.kkt_passes += 1
    return nothing
end


function restart!(state::PDLPState)
    (; z, z_restart_candidate, z_last_restart) = state
    copyto!(z, z_restart_candidate)
    copyto!(z_last_restart, z)
    return nothing
end

function update_average!(state::PDLPState, sad::SaddlePointProblem)
    (; z_avg, z, η, η_sum) = state
    weighted_sum!(z_avg, z, η / (η + η_sum), η_sum / (η + η_sum))
    z_avg.err = kkt_errors!(state, sad, z_avg)
    state.η_sum += η
    return nothing
end

function update_restart_candidate!(state::PDLPState)
    (; z, z_avg, z_restart_candidate, z_prev_restart_candidate) = state
    copyto!(z_prev_restart_candidate, z_restart_candidate)
    if z.err.max_rel_err < z_avg.err.max_rel_err
        copyto!(z_restart_candidate, z)
    else
        copyto!(z_restart_candidate, z_avg)
    end
    return nothing
end

function restart_check(state::PDLPState, params::PDLPParameters)
    (; β_sufficient, β_necessary, β_artificial) = params
    (;
        z_restart_candidate, z_prev_restart_candidate, z_last_restart,
        inner_iterations, total_iterations,
    ) = state

    err_restart_candidate = z_restart_candidate.err.weighted_aggregate_err
    err_prev_restart_candidate = z_prev_restart_candidate.err.weighted_aggregate_err
    err_last_restart = z_last_restart.err.weighted_aggregate_err

    sufficient_decay = err_restart_candidate <= β_sufficient * err_last_restart
    necessary_decay = err_restart_candidate <= β_necessary * err_last_restart
    no_local_progress = err_restart_candidate > err_prev_restart_candidate
    long_inner_loop = inner_iterations >= β_artificial * total_iterations

    restart_criterion = sufficient_decay || (necessary_decay && no_local_progress) || long_inner_loop
    return restart_criterion
end

function termination_check(state::PDLPState, params::PDLPParameters)
    (; termination_reltol, time_limit, max_kkt_passes, record_error_history) = params
    state.time_elapsed = time() - state.starting_time

    if record_error_history
        push!(state.error_history, (state.kkt_passes, state.z.err))
    end

    if state.z.err.max_rel_err <= termination_reltol
        state.termination_reason = CONVERGENCE
        return true
    elseif state.time_elapsed >= time_limit
        state.termination_reason = TIME
        return true
    elseif state.kkt_passes >= max_kkt_passes
        state.termination_reason = ITERATIONS
        return true
    else
        state.termination_reason = STILL_RUNNING
        return false
    end
end
