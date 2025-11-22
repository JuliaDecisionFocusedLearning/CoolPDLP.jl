"""
    PDLPRestartParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PDLPRestartParameters{T <: Number}
    restart_enabled::Bool = true
    "restart criterion: sufficient decay in normalized duality gap"
    β_sufficient::T = 0.2
    "restart criterion: necessary decay"
    β_necessary::T = 0.8
    "restart criterion: long inner loop"
    β_artificial::T = 0.36
end

function Base.show(io::IO, params::PDLPRestartParameters)
    (; restart_enabled, β_sufficient, β_necessary, β_artificial) = params
    return print(io, "Restart ($restart_enabled): β_sufficient=$β_sufficient, β_necessary=$β_necessary, β_artificial=$β_artificial")
end

"""
    PDLPPreconditioningParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PDLPPreconditioningParameters{T <: Number}
    scaling_enabled::Bool = true
    "norm parameter in the Chambolle-pock preconditioner"
    cp_α::T = 1.0
    "iteration parameter in the Ruiz preconditioner"
    ruiz_iter::Int = 10
end

function Base.show(io::IO, params::PDLPPreconditioningParameters)
    (; scaling_enabled, cp_α, ruiz_iter) = params
    return print(io, "Preconditioning ($scaling_enabled): cp_α=$cp_α, ruiz_iter=$ruiz_iter")
end

"""
    PDLPPrimalWeightParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PDLPPrimalWeightParameters{T <: Number}
    primal_weight_enabled::Bool = true
    "scaling of the inverse spectral norm of `K` when defining the step size"
    θ::T = 0.5
end

function Base.show(io::IO, params::PDLPPrimalWeightParameters)
    (; primal_weight_enabled, θ) = params
    return print(io, "Primal weight ($primal_weight_enabled): θ=$θ")
end

"""
    PDLPStepSizeParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PDLPStepSizeParameters{T <: Number}
    adaptive_steps_enabled::Bool = true
    "scaling of the inverse spectral norm of `K` when defining the non-adaptive step size"
    invnorm_scaling::T = 0.9
end

function Base.show(io::IO, params::PDLPStepSizeParameters)
    (; adaptive_steps_enabled, invnorm_scaling) = params
    return print(io, "Step size ($adaptive_steps_enabled): invnorm_scaling=$invnorm_scaling")
end

"""
    PDLPTerminationParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PDLPTerminationParameters{T <: Number}
    "tolerance on KKT relative errors to decide termination"
    termination_reltol::T = 1.0e-4
    "maximum number of multiplications by both the KKT matrix `K` and its transpose `Kᵀ`"
    max_kkt_passes::Int = 100_000
    "time limit in seconds"
    time_limit::Float64 = 100.0
end

function Base.show(io::IO, params::PDLPTerminationParameters)
    (; termination_reltol, max_kkt_passes, time_limit) = params
    return print(io, "Termination: termination_reltol=$termination_reltol, max_kkt_passes=$max_kkt_passes, time_limit=$time_limit")
end

"""
    PDLPGenericParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PDLPGenericParameters{T <: Number}
    "tolerance in absolute comparisons to zero"
    zero_tol::T = 1.0e-8
    "frequency of restart or termination checks"
    check_every::Int = 40
    "whether or not to record error evolution"
    record_error_history::Bool = false
end

function Base.show(io::IO, params::PDLPGenericParameters)
    (; zero_tol, check_every, record_error_history) = params
    return print(io, "Generic: zero_tol=$zero_tol, check_every=$check_every, record_error_history=$record_error_history")
end

"""
    PDLPParameters

Parameters for configuration of the PDLP algorithm.
    
# Fields

$(TYPEDFIELDS)
"""
struct PDLPParameters{
        T <: AbstractFloat, Ti <: Integer, M <: AbstractMatrix, B <: Backend,
    } <: AbstractParameters{T}
    backend::B
    restart::PDLPRestartParameters{T}
    preconditioning::PDLPPreconditioningParameters{T}
    primal_weight::PDLPPrimalWeightParameters{T}
    step_size::PDLPStepSizeParameters{T}
    termination::PDLPTerminationParameters{T}
    generic::PDLPGenericParameters{T}
end

function Base.show(io::IO, params::PDLPParameters{T, Ti, M}) where {T, Ti, M}
    (; backend, restart, preconditioning, primal_weight, step_size, termination, generic) = params
    return print(
        io,
        """PDLP with types ($T, $Ti, $M) on $backend:
        - $restart
        - $preconditioning
        - $primal_weight
        - $step_size
        - $termination
        - $generic
        """
    )
end

function PDLPParameters(
        _T::Type{T} = Float64,
        ::Type{Ti} = Int,
        ::Type{M} = SparseMatrixCSC,
        backend::B = CPU();
        restart_enabled::Bool = true,
        scaling_enabled::Bool = true,
        primal_weight_enabled = true,
        adaptive_steps_enabled = true,
        β_sufficient = _T(0.2),
        β_necessary = _T(0.8),
        β_artificial = _T(0.36),
        cp_α = _T(1.0),
        ruiz_iter = 10,
        θ = _T(0.5),
        invnorm_scaling = _T(0.9),
        termination_reltol = _T(1.0e-4),
        max_kkt_passes = 100_000,
        time_limit = 100.0,
        zero_tol = _T(1.0e-8),
        check_every = 40,
        record_error_history = false
    ) where {T, Ti, M, B}
    restart = PDLPRestartParameters(
        restart_enabled, _T(β_sufficient), _T(β_necessary), _T(β_artificial)
    )
    preconditioning = PDLPPreconditioningParameters(
        scaling_enabled, _T(cp_α), ruiz_iter
    )
    primal_weight = PDLPPrimalWeightParameters(
        primal_weight_enabled, _T(θ)
    )
    step_size = PDLPStepSizeParameters(
        adaptive_steps_enabled, _T(invnorm_scaling)
    )
    termination = PDLPTerminationParameters(
        _T(termination_reltol), max_kkt_passes, time_limit
    )
    generic = PDLPGenericParameters(
        _T(zero_tol), check_every, record_error_history
    )
    return PDLPParameters{T, Ti, M, B}(
        backend, restart, preconditioning, primal_weight, step_size, termination, generic
    )
end

"""
    PDLPState

Current solution, step sizes and various buffers / metrics in the PDLP algorithm.

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
    const z_avg_last::PrimalDualSolution{T, V} = copy(z_avg)
    const z_previous_restart::PrimalDualSolution{T, V} = copy(z)
    z_restart_candidate::PrimalDualSolution{T, V} = z
    z_restart_candidate_last::PrimalDualSolution{T, V} = z_last
    # errors
    err::KKTErrors{T} = KKTErrors(eltype(z))
    err_avg::KKTErrors{T} = KKTErrors(eltype(z))
    err_last::KKTErrors{T} = KKTErrors(eltype(z))
    err_avg_last::KKTErrors{T} = KKTErrors(eltype(z))
    err_previous_restart::KKTErrors{T} = KKTErrors(eltype(z))
    err_restart_candidate::KKTErrors{T} = KKTErrors(eltype(z))
    err_restart_candidate_last::KKTErrors{T} = KKTErrors(eltype(z))
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
    err_primal_scale::T
    err_dual_scale::T
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
    
Apply the PDLP algorithm to solve the continuous relaxation of `milp` using configuration `params`, starting from primal variable `x_init`.
"""
function pdlp(
        milp::MILP,
        params::PDLPParameters,
        x_init::Vector = zero(milp.c);
        show_progress::Bool = true
    )
    starting_time = time()
    sad = SaddlePointProblem(milp)
    y_init = zero(sad.q)
    return pdlp(sad, params, x_init, y_init; show_progress, starting_time)
end

"""
    pdlp(
        sad::SaddlePointProblem,
        params::PDLPParameters,
        x_init,
        y_init;
        show_progress::Bool=true
    )
    
Apply the PDLP algorithm to solve the saddle-point problem `sad` using configuration `params`, starting from `z_init = (x_init, y_init)`.
"""
function pdlp(
        sad_init::SaddlePointProblem,
        params::PDLPParameters,
        x_init::Vector = zero(sad_init.c),
        y_init::Vector = zero(sad_init.q);
        starting_time::Float64 = time(),
        show_progress::Bool = true,
    )
    sad, state = initialize(sad_init, params, x_init, y_init; starting_time)
    prog = ProgressUnknown(desc = "PDLP iterations:", enabled = show_progress)
    try
        while true
            must_terminate = false
            while true
                yield()
                for _ in 1:params.generic.check_every
                    step!(state, sad, params)
                    update_average!(state, sad)
                    state.inner_iterations += 1
                    state.total_iterations += 1
                    next!(
                        prog;
                        showvalues = (("relative_error", relative(state.err)),)
                    )
                end
                prepare_check!(state, sad, params)
                must_restart = restart_check(state, params)
                must_terminate = termination_check!(state, params.termination)
                if must_restart || must_terminate
                    break
                end
            end
            primal_weight_update!(state, params)
            restart!(state)
            state.outer_iterations += 1
            state.inner_iterations = 0
            if must_terminate
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
    return get_results(sad, state)
end

function initialize(
        sad_init::SaddlePointProblem,
        params::PDLPParameters{T, Ti, M},
        x_init::Vector,
        y_init::Vector;
        starting_time::Float64
    ) where {T, Ti, M}
    (; backend) = params
    preconditioner = compute_preconditioner(sad_init, params)
    sad = apply(preconditioner, sad_init)
    x, y = preconditioned_solution(preconditioner, x_init, y_init)
    sad_righttypes = set_matrix_type(M, set_indtype(Ti, set_eltype(T, sad)))
    η_init = initial_stepsize(sad_righttypes, params)
    ω = initialize_primal_weight(sad_righttypes, params)
    err_primal_scale, err_dual_scale = error_scales(sad_righttypes)
    sad_gpu = adapt(backend, sad_righttypes)
    x_gpu = adapt(backend, set_eltype(T, x))
    y_gpu = adapt(backend, set_eltype(T, y))
    z = PrimalDualSolution(sad_gpu, x_gpu, y_gpu)
    state = PDLPState(; z, η_init, ω, err_primal_scale, err_dual_scale, starting_time)
    return sad_gpu, state
end

function compute_preconditioner(
        sad::SaddlePointProblem, params::PDLPParameters,
    )
    (; K, Kᵀ) = sad
    (; scaling_enabled, ruiz_iter, cp_α) = params.preconditioning
    if scaling_enabled
        p1 = ruiz_preconditioner(K, Kᵀ; iterations = ruiz_iter)
        K, Kᵀ = apply(p1, K, Kᵀ)
    else
        p1 = identity_preconditioner(K)
    end
    p2 = chambolle_pock_preconditioner(K, Kᵀ; α = cp_α)
    return p2 * p1
end

function initial_stepsize(
        sad::SaddlePointProblem{T},
        params::PDLPParameters{T}
    ) where {T}
    (; K, Kᵀ) = sad
    (; adaptive_steps_enabled, invnorm_scaling) = params.step_size
    if adaptive_steps_enabled
        η_init = inv(opnorm(K, Inf))
    else
        η_init = T(invnorm_scaling) * inv(spectral_norm(K, Kᵀ))
    end
    return η_init
end

function initialize_primal_weight(
        sad::SaddlePointProblem{T}, params::PDLPParameters
    ) where {T}
    (; c, q) = sad
    (; primal_weight_enabled) = params.primal_weight
    (; zero_tol) = params.generic
    c_norm, q_norm = norm(c), norm(q)
    if primal_weight_enabled && c_norm > zero_tol && q_norm > zero_tol
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
    (; adaptive_steps_enabled) = params.step_size
    copy!(z_last, z)
    if adaptive_steps_enabled
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
        dual_scratch2,
    ) = state
    (; c, q, K, Kᵀ, l, u, ineq_cons) = sad
    (; x, y, Kx, Kᵀy, λ) = z
    (; zero_tol) = params.generic

    η, ηp = η_init, η_init
    xp, yp = primal_scratch, dual_scratch
    Kxp = dual_scratch2
    k = total_iterations

    while true
        τ, σ = η / ω, η * ω

        # xp = proj_X(x - τ * (c - Kᵀ * y))
        @. xp = x - τ * (c - Kᵀy)
        @. xp = proj_box(xp, l, u)

        # yp = proj_Y(y + σ * (q - K * (2 * xp - x)))
        @. yp = y + σ * (q + Kx)
        mul!(Kxp, K, xp)
        @. yp -= 2 * σ * Kxp
        @. yp = ifelse(ineq_cons, positive_part(yp), yp)

        @. xp = xp - x  # now it stores the difference
        @. yp = yp - y  # now it stores the difference

        η_bar_num = custom_sqnorm(xp, yp, ω)
        η_bar_den = 2 * abs(dot(yp, Kxp))  # TODO: why should this be > 0?
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

    @. x += xp  # we get back the actual new solution by adding the difference
    @. y += yp  # we get back the actual new solution by adding the difference

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
    lᵀλ⁺ = mapreduce(safeprod_rightpos, +, l, λ)
    uᵀλ⁻ = mapreduce(safeprod_rightneg, +, u, λ)

    @. dual_scratch = ifelse(ineq_cons, positive_part(q - Kx), q - Kx)
    primal = norm(dual_scratch)

    @. primal_scratch = c - Kᵀy - λ
    dual = norm(primal_scratch)

    gap = abs(qᵀy + lᵀλ⁺ - uᵀλ⁻ - cᵀx)
    gap_scale = one(T) + abs(qᵀy + lᵀλ⁺ - uᵀλ⁻) + abs(cᵀx)

    weighted_agg = sqrt(ω^2 * primal^2 + inv(ω^2) * dual^2 + gap^2)

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
        weighted_agg,
        rel_max,
    )
end

function prepare_check!(state::PDLPState, sad::SaddlePointProblem, params::PDLPParameters)
    (; z, z_last, z_avg, z_avg_last) = state
    (; record_error_history) = params.generic
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
    (; restart_enabled, β_sufficient, β_necessary, β_artificial) = params.restart
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
    return restart_enabled && restart_criterion
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
    (; primal_weight_enabled, θ) = params.primal_weight
    (; zero_tol) = params.generic

    @. primal_scratch = z_restart_candidate.x - z_previous_restart.x
    @. dual_scratch = z_restart_candidate.y - z_previous_restart.y
    Δx = norm(primal_scratch)
    Δy = norm(dual_scratch)
    if primal_weight_enabled && Δx > zero_tol && Δy > zero_tol
        new_ω = exp(θ * log(Δy / Δx) + (one(T) - θ) * log(ω))
        state.ω = new_ω
    end
    return nothing
end

function get_results(
        sad::SaddlePointProblem,
        state::PDLPState
    )
    (; preconditioner) = sad
    (; x, y) = state.z
    x_cpu, y_cpu = Array(x), Array(y)
    x_unprec, y_unprec = unpreconditioned_solution(preconditioner, x_cpu, y_cpu)
    return (; x = x_unprec, y = y_unprec), state
end
