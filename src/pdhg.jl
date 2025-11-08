"""
    PDHGParameters

Parameters for configuration of the baseline primal-dual hybrid gradient.
    
# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PDHGParameters{T <: Number} <: AbstractParameters{T}
    "scaling of the inverse spectral norm of `K` when defining the step size"
    stepsize_scaling::T = 0.9
    "norm parameter in the Chambolle-pock preconditioner"
    preconditioner_chambollepock_alpha::T = 1.0
    # termination parameters
    "tolerance when checking KKT relative errors to decide termination"
    termination_reltol::T = 1.0e-4
    "maximum number of multiplications by both the KKT matrix `K` and its transpose `Kᵀ`"
    max_kkt_passes::Int = 100_000
    "time limit in seconds"
    time_limit::Float64 = 100.0
    # meta parameters
    "frequency of termination checks"
    check_every::Int = 40
    "whether or not to record error evolution"
    record_error_history::Bool = false
end

"""
    PDHGState

Current solution, step sizes and various buffers / metrics for the baseline primal-dual hybrid gradient.

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct PDHGState{
        T <: Number, V <: AbstractVector{T},
    } <: AbstractState{T, V}
    "current primal solution"
    const x::V
    "current dual solution"
    const y::V
    "step size"
    η::T
    "primal weight"
    ω::T = one(η)
    "time at which the algorithm started, in seconds"
    starting_time::Float64 = time()
    "time elapsed since the algorithm started, in seconds"
    time_elapsed::Float64 = 0.0
    "number of multiplications by both the KKT matrix and its transpose"
    kkt_passes::Int = 0
    "current KKT error"
    current_err::KKTErrors{T} = KKTErrors(eltype(x))
    "termination reason (should be `STILL_RUNNING` until the algorithm actuall terminates)"
    termination_reason::TerminationReason = STILL_RUNNING
    "history of KKT errors, indexed by number of KKT passes"
    const error_history::Vector{Tuple{Int, KKTErrors{T}}} = Tuple{Int, KKTErrors{eltype(x)}}[]
end

function Base.show(io::IO, state::PDHGState)
    (; current_err, time_elapsed, kkt_passes, termination_reason) = state
    return print(
        io,
        @sprintf(
            "%s with termination reason %s: %.6e relative KKT error after %g seconds elapsed and %s KKT passes",
            nameof(typeof(state)),
            termination_reason,
            current_err.max_rel_err,
            time_elapsed,
            kkt_passes,
        )
    )
end

"""
    pdhg(
        milp::MILP,
        params::PDHGParameters,
        x_init::AbstractVector=zero(milp.c);
        show_progress::Bool=true
    )
    
Apply the primal-dual hybrid gradient algorithm to solve the continuous relaxation of `milp` using configuration `params`, starting from primal variable `x_init`.
"""
function pdhg(
        milp::MILP{T, V},
        params::PDHGParameters{T},
        x_init::V = zero(milp.c);
        show_progress::Bool = true
    ) where {T, V}
    starting_time = time()
    sad = SaddlePointProblem(milp)
    y_init = zero(sad.q)
    return pdhg(sad, params, x_init, y_init; show_progress, starting_time)
end

"""
    pdhg(
        sad::SaddlePointProblems,
        params::PDHGParameters,
        x_init::AbstractVector=zero(sad.c);
        y_init::AbstractVector=zero(sad.q);
        show_progress::Bool=true
    )
    
Apply the primal-dual hybrid gradient algorithm to solve the saddle-point problem `sad` using configuration `params`, starting from `(x_init, y_init)`.

Return a triplet `(x, y, state)` where `x` is the primal solution, `y` is the dual solution and `state` is the algorithm's final state, including convergence information.
"""
function pdhg(
        sad::SaddlePointProblem{T, V},
        params::PDHGParameters{T},
        x_init::V = zero(sad.c),
        y_init::V = zero(sad.q);
        show_progress::Bool = true,
        starting_time::Float64 = time()
    ) where {T, V}
    sad = precondition_chambolle_pock(sad; α = params.preconditioner_chambollepock_alpha)
    x, y = preconditioned_solution(sad, x_init, y_init)
    η = fixed_stepsize(sad, params)
    state = PDHGState(; x, y, η, starting_time)
    push!(state.error_history, (0, kkt_errors(state, sad)))
    prog = ProgressUnknown(desc = "PDHG iterations:", enabled = show_progress)
    while true
        yield()
        for _ in 1:params.check_every
            next!(prog; showvalues = (("relative_error", state.current_err.max_rel_err),))
            pdhg_step!(state, sad)
        end
        if termination_check(state, sad, params)
            break
        end
    end
    finish!(prog)
    return unpreconditioned_solution(sad, state.x, state.y), state
end

function fixed_stepsize(
        sad::SaddlePointProblem{T},
        params::PDHGParameters{T}
    ) where {T}
    (; K, Kᵀ) = sad
    (; stepsize_scaling) = params
    η = T(stepsize_scaling) * inv(spectral_norm(K, Kᵀ))
    return η
end

function proj_λ(λ::T, l::T, u::T) where {T <: Number}
    if l == typemin(T) && u == typemax(T)
        return zero(T)  # project on {0}
    elseif l == typemin(T)
        return -negative_part(λ)  # project on ℝ⁻
    elseif u == typemax(T)
        return positive_part(λ)  # project on ℝ⁺
    else
        return λ  # project on ℝ
    end
end

function pdhg_step!(
        state::PDHGState{T, V},
        sad::SaddlePointProblem{T, V},
    ) where {T, V}
    (; x, y, η, ω) = state
    (; c, q, K, Kᵀ, l, u, ineq_cons) = sad

    τ, σ = η / ω, η * ω

    # xp = proj_X(x - τ * (c - Kᵀ * y))
    x_step = x - τ * (c - Kᵀ * y)
    xp = proj_box.(x_step, l, u)

    # yp = proj_Y(y + σ * (q - K * (2 * xp - x)))
    y_step = y + σ * (q - K * (2 * xp - x))
    yp = ifelse.(ineq_cons, positive_part.(y_step), y_step)

    copyto!(x, xp)
    copyto!(y, yp)

    state.kkt_passes += 1
    return nothing
end

function kkt_errors(
        state::PDHGState,
        sad::SaddlePointProblem{T, V},
    ) where {T, V}
    (; x, y, ω) = state
    (; c, q, K, Kᵀ, l, u, ineq_cons) = sad

    λ = proj_λ.(c - Kᵀ * y, l, u)
    λ⁺ = positive_part.(λ)
    λ⁻ = negative_part.(λ)
    l_noinf = max.(nextfloat(typemin(T)), l)
    u_noinf = min.(prevfloat(typemax(T)), u)

    lᵀλ⁺ = dot(l_noinf, λ⁺)
    uᵀλ⁻ = dot(u_noinf, λ⁻)
    qᵀy = dot(q, y)
    cᵀx = dot(c, x)

    err_primal = norm(
        ifelse.(
            ineq_cons,
            positive_part.(q - K * x),
            q - K * x
        )
    )
    err_primal_scale = one(T) + norm(q)

    err_dual = norm(c - Kᵀ * y - λ)
    err_dual_scale = one(T) + norm(c)

    err_gap = abs(qᵀy + lᵀλ⁺ - uᵀλ⁻ - cᵀx)
    err_gap_scale = one(T) + abs(qᵀy + lᵀλ⁺ - uᵀλ⁻) + abs(cᵀx)

    weighted_aggregate_err = sqrt(ω^2 * err_primal^2 + inv(ω^2) * err_dual^2 + err_gap^2)

    relative_error_primal = err_primal / err_primal_scale
    relative_error_dual = err_dual / err_dual_scale
    relative_error_gap = err_gap / err_gap_scale

    max_rel_err = max(relative_error_primal, relative_error_dual, relative_error_gap)

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

function termination_check(
        state::PDHGState,
        sad::SaddlePointProblem,
        params::PDHGParameters,
    )
    (; starting_time) = state
    (; termination_reltol, time_limit, max_kkt_passes, record_error_history) = params
    state.time_elapsed = time() - starting_time
    state.current_err = kkt_errors(state, sad)

    if record_error_history
        push!(state.error_history, (state.kkt_passes, state.current_err))
    end

    if state.current_err.max_rel_err <= termination_reltol
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
