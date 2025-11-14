"""
    PDLPParameters

Parameters for configuration of the primal-dual hybrid gradient for linear programming (PDLP).
    
# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PDLPParameters{T <: Number} <: AbstractParameters{T}
    "whether to enable restarts"
    enable_restarts::Bool = true
    "whether to enable scaling"
    enable_scaling::Bool = true
    "whether to enable primal weight heuristics"
    enable_primal_weight::Bool = true
    "whether to enable adaptive step sizes"
    enable_step_size::Bool = true
    # primal weight parameters
    primal_weight_θ::T = 0.5
    # step size parameters
    "scaling of the inverse spectral norm of `K` when defining the step size"
    stepsize_scaling::T = 0.9
    # preconditioning parameters
    "norm parameter in the Chambolle-pock preconditioner"
    precond_cp_α::T = 1.0
    "iteration parameter in the Ruiz preconditioner"
    precond_ruiz_iter::Int = 10
    # restart parameters
    "restart criterion: sufficient decay in normalized duality gap"
    β_sufficient::T = 0.2
    "restart criterion: necessary decay"
    β_necessary::T = 0.8
    "restart criterion: long inner loop"
    β_artificial::T = 0.36
    # termination parameters
    "tolerance on KKT relative errors to decide termination"
    termination_reltol::T = 1.0e-4
    "maximum number of multiplications by both the KKT matrix `K` and its transpose `Kᵀ`"
    max_kkt_passes::Int = 100_000
    "time limit in seconds"
    time_limit::Float64 = 100.0
    # meta parameters
    "tolerance in absolute comparisons to zero"
    zero_tol::T = 1.0e-8
    "frequency of restart or termination checks"
    check_every::Int = 40
    "whether or not to record error evolution"
    record_error_history::Bool = false
    "name of the parametrization for comparative benchmarks"
    name::String = ""
end

"""
    PDLPState

Current solution, step sizes and various buffers / metrics in the baseline primal-dual hybrid gradient for linear programming (PDLP).

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct PDLPState{
        T <: Number, V <: AbstractVector{T},
    } <: AbstractState{T, V}
    # solutions
    const z::PrimalDualSolution{T, V}
    const z_avg::PrimalDualSolution{T, V} = zero(z)
    const z_last::PrimalDualSolution{T, V} = copy(z)
    const z_avg_last::PrimalDualSolution{T, V} = zero(z)
    z_restart_candidate::PrimalDualSolution{T, V} = z
    z_restart_candidate_last::PrimalDualSolution{T, V} = z_last
    const z_previous_restart::PrimalDualSolution{T, V} = copy(z)
    #errors
    err::KKTErrors{T} = KKTErrors(eltype(z))
    err_avg::KKTErrors{T} = KKTErrors(eltype(z))
    err_last::KKTErrors{T} = KKTErrors(eltype(z))
    err_avg_last::KKTErrors{T} = KKTErrors(eltype(z))
    err_restart_candidate::KKTErrors{T} = KKTErrors(eltype(z))
    err_restart_candidate_last::KKTErrors{T} = KKTErrors(eltype(z))
    err_previous_restart::KKTErrors{T} = KKTErrors(eltype(z))
    # step sizes
    "primal weight"
    ω::T
    "initialization for step size line search"
    η_init::T
    "current step size"
    η::T = η_init
    "sum of step sizes since the last restart"
    η_sum::T = zero(η_init)
    # error scales
    err_primal_scale = T(NaN)
    err_dual_scale = T(NaN)
    # scratch spaces
    primal_scratch::V = zero(z.x)
    primal_scratch2::V = zero(z.x)
    dual_scratch::V = zero(z.y)
    dual_scratch2::V = zero(z.y)
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

"""
    pdlp(
        milp::MILP,
        params::PDLPParameters,
        x_init::AbstractVector=zero(milp.c);
        show_progress::Bool=true
    )
    
Apply the primal-dual hybrid gradient algorithm for linear programming (PDLP) to solve the continuous relaxation of `milp` using configuration `params`, starting from primal variable `x_init`.
"""
function pdlp(
        milp::MILP,
        params::PDLPParameters{T},
        x_init::V = zero(milp.c);
        show_progress::Bool = true
    ) where {T, V}
    starting_time = time()
    sad = SaddlePointProblem(milp)
    y_init = zero(sad.q)
    return pdlp(sad, params, x_init, y_init; show_progress, starting_time)
end

"""
    pdlp(
        sad::SaddlePointProblem,
        params::PDLPParameters,
        x_init, y_init;
        show_progress::Bool=true
    )
    
Apply the primal-dual hybrid gradient algorithm to solve the saddle-point problem `sad` using configuration `params`, starting from `z_init = (x_init, y_init)`.
"""
function pdlp(
        sad_init::SaddlePointProblem{T, V},
        params::PDLPParameters{T},
        x_init::V = zero(sad_init.c),
        y_init::V = zero(sad_init.q);
        starting_time::Float64 = time(),
        show_progress::Bool = true,
    ) where {T, V}
    sad, state = initialization(sad_init, params, x_init, y_init; starting_time)
    prog = ProgressUnknown(desc = "PDLP iterations:", enabled = show_progress)
    try
        while true
            must_restart = false
            while true
                yield()
                for _ in 1:params.check_every
                    step!(state, sad, params)
                    update_average!(state, sad)
                    state.inner_iterations += 1
                    state.total_iterations += 1
                    next!(
                        prog;
                        showvalues = (("relative_error", state.err.rel_max),)
                    )
                end
                prepare_check!(state, sad)
                must_restart = restart_check(state, params)
                must_terminate = termination_check(state, params)
                if must_restart || must_terminate
                    break
                end
            end
            if must_restart
                primal_weight_update!(state, params)
                restart!(state)
                state.outer_iterations += 1
                state.inner_iterations = 0
            else # must_terminate
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
    return return_value(sad, state)
end

function initialization(
        sad_init::SaddlePointProblem{T, V},
        params::PDLPParameters{T},
        x_init::V,
        y_init::V;
        starting_time::Float64
    ) where {T, V}
    preconditioner = compute_preconditioner(sad_init, params)
    sad = apply(preconditioner, sad_init)
    x, y = preconditioned_solution(preconditioner, x_init, y_init)
    z = PrimalDualSolution(sad, x, y)
    η_init = initial_stepsize(sad, params)
    ω = initialize_primal_weight(sad, params)
    err_primal_scale, err_dual_scale = error_scales(sad)
    state = PDLPState(; z, η_init, ω, err_primal_scale, err_dual_scale, starting_time)
    return sad, state
end

function compute_preconditioner(
        sad::SaddlePointProblem, params::PDLPParameters,
    )
    (; K, Kᵀ) = sad
    (; enable_scaling, precond_ruiz_iter, precond_cp_α) = params
    p1 = ruiz_preconditioner(K, Kᵀ; iterations = enable_scaling * precond_ruiz_iter)
    K, Kᵀ = apply(p1, K, Kᵀ)
    p2 = chambolle_pock_preconditioner(K, Kᵀ; α = precond_cp_α)
    return p1 * p2
end

function initial_stepsize(
        sad::SaddlePointProblem{T},
        params::PDLPParameters{T}
    ) where {T}
    (; K, Kᵀ) = sad
    (; enable_step_size, stepsize_scaling) = params
    if enable_step_size
        η_init = inv(opnorm(K, Inf))
    else
        η_init = T(stepsize_scaling) * inv(spectral_norm(K, Kᵀ))
    end
    return η_init
end

function initialize_primal_weight(
        sad::SaddlePointProblem{T}, params::PDLPParameters
    ) where {T}
    (; c, q) = sad
    (; enable_primal_weight, zero_tol) = params
    c_norm, q_norm = norm(c), norm(q)
    if enable_primal_weight && c_norm > zero_tol && q_norm > zero_tol
        return c_norm / q_norm
    else
        return one(T)
    end
end

function error_scales(sad::SaddlePointProblem{T}) where {T}
    (; c, q) = sad
    err_primal_scale = one(T) + norm(q)
    err_dual_scale = one(T) + norm(c)
    return (; err_primal_scale, err_dual_scale)
end

function step!(
        state::PDLPState{T, V},
        sad::SaddlePointProblem{T, V}, params::PDLPParameters{T}
    ) where {T, V}
    (; z, z_last) = state
    copy!(z_last, z)
    if params.enable_step_size
        return adaptive_pdlp_step!(state, sad, params)
    else
        return fixed_pdlp_step!(state, sad)
    end
end

function fixed_pdlp_step!(
        state::PDLPState{T, V},
        sad::SaddlePointProblem{T, V}
    ) where {T, V}
    (; z, η, ω) = state
    (; c, q, K, Kᵀ, l, u, ineq_cons) = sad
    (; x, y, Kx, Kᵀy, λ) = z

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

    state.kkt_passes += 1
    return nothing
end

function adaptive_pdlp_step!(
        state::PDLPState{T, V},
        sad::SaddlePointProblem{T, V},
        params::PDLPParameters{T}
    ) where {T, V}
    (;
        z, ω, η_init, total_iterations,
        primal_scratch, dual_scratch,
        primal_scratch2, dual_scratch2,
    ) = state
    (; c, q, K, Kᵀ, l, u, ineq_cons) = sad
    (; x, y, Kx, Kᵀy, λ) = z
    (; zero_tol) = params

    η, ηp = η_init, η_init
    xp, yp = primal_scratch, dual_scratch
    xp_minus_x, yp_minus_y = primal_scratch2, dual_scratch2
    k = total_iterations

    for _ in 1:100
        yield()
        # xp = proj_X(x - τ * (c - Kᵀ * y))
        @. xp = x - (η / ω) * (c - Kᵀy)
        @. xp = proj_box(xp, l, u)

        # yp = proj_Y(y + σ * (q - K * (2 * xp - x)))
        @. yp = y + (η * ω) * (q + Kx)
        mul!(yp, K, xp, -2(η * ω), 1)
        @. yp = ifelse(ineq_cons, positive_part(yp), yp)

        @. xp_minus_x = xp - x
        @. yp_minus_y = yp - y

        η_bar_num = custom_sqnorm(xp_minus_x, yp_minus_y, ω)
        η_bar_den = 2 * abs(dot(yp_minus_y, K, xp_minus_x))  # TODO: why should this be > 0?
        if η_bar_den < zero_tol
            η_bar = typemax(T)
        else
            η_bar = η_bar_num / η_bar_den
        end

        @assert !isnan(η_bar)

        state.kkt_passes += 1

        ηp = min(
            (1 - (k + 2)^T(-0.3)) * η_bar,
            (1 + (k + 2)^T(-0.6)) * η
        )

        if η <= η_bar
            break
        else
            η = ηp
        end
    end

    state.η = η
    state.η_init = ηp

    copy!(x, xp)
    copy!(y, yp)

    mul!(Kx, K, x)
    mul!(Kᵀy, Kᵀ, y)
    state.kkt_passes += 1

    # λ = proj_Λ(c - Kᵀy)
    @. λ = proj_λ(c - Kᵀy, l, u)

    return nothing
end

function update_average!(state::PDLPState, sad::SaddlePointProblem)
    (; z, z_avg, z_avg_last, η, η_sum) = state
    copy!(z_avg_last, z_avg)
    axpby!(η / (η + η_sum), z, η_sum / (η + η_sum), z_avg)
    state.η_sum += η
    return nothing
end

function kkt_errors!(
        state::PDLPState{T, V},
        sad::SaddlePointProblem{T, V},
        z::PrimalDualSolution{T, V},
    ) where {T, V}
    (; ω, err_primal_scale, err_dual_scale, primal_scratch, dual_scratch) = state
    (; x, y, Kx, Kᵀy, λ) = z
    (; c, q, l, u, ineq_cons) = sad

    qᵀy = dot(q, y)
    cᵀx = dot(c, x)
    lᵀλ⁺ = mapreduce(prod_rightpos, +, l, λ)
    uᵀλ⁻ = mapreduce(prod_rightneg, +, u, λ)

    @. dual_scratch = ifelse(ineq_cons, positive_part(q - Kx), q - Kx)
    primal = norm(dual_scratch)

    @. primal_scratch = c - Kᵀy - λ
    dual = norm(primal_scratch)

    gap = abs(qᵀy + lᵀλ⁺ - uᵀλ⁻ - cᵀx)
    gap_scale = one(T) + abs(qᵀy + lᵀλ⁺ - uᵀλ⁻) + abs(cᵀx)

    weighted_aggregate = sqrt(ω^2 * primal^2 + inv(ω^2) * dual^2 + gap^2)

    rel_primal = primal / err_primal_scale
    rel_dual = dual / err_dual_scale
    rel_gap = gap / gap_scale
    rel_max = max(rel_primal, rel_dual, rel_gap)

    return KKTErrors(;
        primal,
        dual,
        gap,
        primal_scale = err_primal_scale,
        dual_scale = err_dual_scale,
        gap_scale,
        weighted_aggregate,
        rel_max,
    )
end

function prepare_check!(state::PDLPState, sad::SaddlePointProblem)
    (; z, z_last, z_avg, z_avg_last) = state
    # errors
    state.err = kkt_errors!(state, sad, z)
    state.err_avg = kkt_errors!(state, sad, z_avg)
    state.err_last = kkt_errors!(state, sad, z_last)
    state.err_avg_last = kkt_errors!(state, sad, z_avg_last)
    # pointers
    if absolute(state.err) < absolute(state.err_avg)
        state.z_restart_candidate = z
    else
        state.z_restart_candidate = z_avg
    end
    if absolute(state.err_last) < absolute(state.err_avg_last)
        state.z_restart_candidate_last = z_last
    else
        state.z_restart_candidate_last = z_avg_last
    end
    # counter updates
    state.time_elapsed = time() - state.starting_time
    if record_error_history
        push!(state.error_history, (state.kkt_passes, state.err))
    end
    return nothing
end

function restart_check(state::PDLPState, params::PDLPParameters)
    (; enable_restarts, β_sufficient, β_necessary, β_artificial) = params
    (;
        err_restart_candidate, err_restart_candidate_last, err_previous_restart,
        inner_iterations, total_iterations,
    ) = state

    sufficient_decay = absolute(err_restart_candidate) <= β_sufficient * absolute(err_previous_restart)
    necessary_decay = absolute(err_restart_candidate) <= β_necessary * absolute(err_previous_restart)
    no_local_progress = absolute(err_restart_candidate) > absolute(err_restart_candidate_last)
    long_inner_loop = inner_iterations >= β_artificial * total_iterations

    restart_criterion = sufficient_decay ||
        (necessary_decay && no_local_progress) ||
        long_inner_loop
    return enable_restarts && restart_criterion
end

function restart!(state::PDLPState{T}) where {T}
    (; z, z_avg, z_restart_candidate, z_previous_restart) = state
    copy!(z, z_restart_candidate)
    copy!(z_previous_restart, z)
    zero!(z_avg)
    state.err_previous_restart = state.err_restart_candidate
    state.η_sum = zero(T)
    return nothing
end


function primal_weight_update!(
        state::PDLPState{T}, params::PDLPParameters,
    ) where {T}
    (; ω, z_restart_candidate, z_previous_restart, primal_scratch, dual_scratch) = state
    (; enable_primal_weight, primal_weight_θ, zero_tol) = params
    @. primal_scratch = z_restart_candidate.x - z_previous_restart.x
    @. dual_scratch = z_restart_candidate.y - z_previous_restart.y
    Δx = norm(primal_scratch)
    Δy = norm(dual_scratch)
    if enable_primal_weight && Δx > zero_tol && Δy > zero_tol
        new_ω = exp(primal_weight_θ * log(Δy / Δx) + (one(T) - primal_weight_θ) * log(ω))
        state.ω = new_ω
    end
    return nothing
end

function return_value(
        sad::SaddlePointProblem,
        state::PDLPState
    )
    (; preconditioner) = sad
    (; x, y) = state.z
    return unpreconditioned_solution(preconditioner, Array(x), Array(y)), state
end
