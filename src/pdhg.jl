"""
    PDHGParameters

Parameters for configuration of the baseline primal-dual hybrid gradient.
    
# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PDHGParameters{Tv <: Number} <: AbstractParameters{Tv}
    "tolerance when checking KKT relative errors to decide termination"
    termination_reltol::Tv = 1.0e-4
    "scaling of the inverse spectral norm of `K` when defining the step size"
    stepsize_scaling::Tv = typeof(termination_reltol)(0.9)
    "norm parameter in the Chambolle-pock preconditioner"
    preconditioner_chambollepock_alpha::Tv = typeof(termination_reltol)(1)
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
    PDHGState

Current solution, step sizes and various buffers / metrics for the baseline primal-dual hybrid gradient.

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct PDHGState{
        Tv <: Number, V <: AbstractVector{Tv},
    } <: AbstractState{Tv, V}
    "current primal solution"
    const x::V
    "current dual solution"
    const y::V
    "step size"
    η::Tv
    "primal weight"
    ω::Tv = one(η)
    "buffer"
    const x_scratch1::V = copy(x)
    "buffer"
    const x_scratch2::V = copy(x)
    "buffer"
    const x_scratch3::V = copy(x)
    "buffer"
    const y_scratch::V = copy(y)
    "time at which the algorithm started, in seconds"
    starting_time::Float64 = time()
    "time elapsed since the algorithm started, in seconds"
    time_elapsed::Float64 = 0.0
    "number of multiplications by both the KKT matrix and its transpose"
    kkt_passes::Int = 0
    "current relative KKT error"
    relative_error::Tv = typemax(eltype(x))
    "termination reason (should be `STILL_RUNNING` until the algorithm actuall terminates)"
    termination_reason::TerminationReason = STILL_RUNNING
    "history of relative KKT errors, indexed by number of KKT passes"
    const relative_error_history::Vector{Tuple{Int, Tv}} = Tuple{Int, eltype(x)}[]
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
        milp::MILP{Tv, V},
        params::PDHGParameters{Tv},
        x_init::V = zero(milp.c);
        show_progress::Bool = true
    ) where {Tv, V}
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
        sad::SaddlePointProblem{Tv, V},
        params::PDHGParameters{Tv},
        x_init::V = zero(sad.c),
        y_init::V = zero(sad.q);
        show_progress::Bool = true,
        starting_time::Float64 = time()
    ) where {Tv, V}
    x, y = copy(x_init), copy(y_init)
    sad = precondition_chambolle_pock(sad; α = params.preconditioner_chambollepock_alpha)
    η = fixed_stepsize(sad, params)
    state = PDHGState(; x, y, η, starting_time)
    prog = ProgressUnknown(desc = "PDHG iterations:", enabled = show_progress)
    while true
        yield()
        for _ in 1:params.check_every
            next!(prog; showvalues = (("relative_error", state.relative_error),))
            pdhg_step!(state, sad)
        end
        if termination_check!(state, sad, params)
            break
        end
    end
    finish!(prog)
    return primal_solution(state, sad), dual_solution(state, sad), state
end

function fixed_stepsize(
        sad::SaddlePointProblem{Tv},
        params::PDHGParameters{Tv}
    ) where {Tv}
    (; K, Kᵀ) = sad
    (; stepsize_scaling) = params
    η = Tv(stepsize_scaling) * inv(spectral_norm(K, Kᵀ))
    return η
end

function pdhg_step!(
        state::PDHGState{Tv, V},
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
