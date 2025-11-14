"""
    PDHGParameters

Parameters for configuration of the PDHG algorithm.
    
# Fields

$(TYPEDFIELDS)
"""
struct PDHGParameters{
        T <: AbstractFloat, Ti <: Integer, M <: AbstractMatrix, B <: Backend,
    } <: AbstractParameters{T}
    "CPU or GPU backend used for computations"
    backend::B
    "scaling of the inverse spectral norm of `K` when defining the step size"
    stepsize_scaling::T
    "norm parameter in the Chambolle-pock preconditioner"
    precond_cb_α::T
    # termination parameters
    "tolerance when checking KKT relative errors to decide termination"
    termination_reltol::T
    "maximum number of multiplications by both the KKT matrix `K` and its transpose `Kᵀ`"
    max_kkt_passes::Int
    "time limit in seconds"
    time_limit::Float64
    # meta parameters
    "frequency of termination checks"
    check_every::Int
    "whether or not to record error evolution"
    record_error_history::Bool

    function PDHGParameters(
            ::Type{T} = Float64,
            ::Type{Ti} = Int,
            ::Type{M} = SparseMatrixCSC,
            backend::B = CPU();
            stepsize_scaling = 0.9,
            precond_cb_α = 1.0,
            termination_reltol = 1.0e-4,
            max_kkt_passes = 100_000,
            time_limit = 100.0,
            check_every = 40,
            record_error_history = false,
        ) where {T, Ti, M, B}

        return new{T, Ti, M, B}(
            backend,
            stepsize_scaling,
            precond_cb_α,
            termination_reltol,
            max_kkt_passes,
            time_limit,
            check_every,
            record_error_history,
        )
    end
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
    err::KKTErrors{T} = KKTErrors(eltype(x))
    "termination reason (should be `STILL_RUNNING` until the algorithm actuall terminates)"
    termination_reason::TerminationReason = STILL_RUNNING
    "history of KKT errors, indexed by number of KKT passes"
    const error_history::Vector{Tuple{Int, KKTErrors{T}}} = Tuple{Int, KKTErrors{eltype(x)}}[]
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
        milp::MILP,
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
        x_init::Vector=zero(sad.c);
        y_init::Vector=zero(sad.q);
        show_progress::Bool=true
    )
    
Apply the primal-dual hybrid gradient algorithm to solve the saddle-point problem `sad` using configuration `params`, starting from `(x_init, y_init)`.

Return a triplet `(x, y, state)` where `x` is the primal solution, `y` is the dual solution and `state` is the algorithm's final state, including convergence information.
"""
function pdhg(
        sad_init::SaddlePointProblem,
        params::PDHGParameters,
        x_init::Vector = zero(sad_init.c),
        y_init::Vector = zero(sad_init.q);
        show_progress::Bool = true,
        starting_time::Float64 = time()
    )
    sad, state = initialize(sad_init, params, x_init, y_init; starting_time)
    prog = ProgressUnknown(desc = "PDHG iterations:", enabled = show_progress)
    while true
        yield()
        for _ in 1:params.check_every
            step!(state, sad)
            next!(prog; showvalues = (("relative_error", relative(state.err)),))
        end
        prepare_check!(state, sad, params)
        if termination_check!(state, params)
            break
        end
    end
    finish!(prog)
    return get_results(state, sad)
end

function initialize(
        sad_init::SaddlePointProblem,
        params::PDHGParameters{T, Ti, M},
        x_init::Vector,
        y_init::Vector;
        starting_time::Float64
    ) where {T, Ti, M}
    (; backend) = params
    preconditioner = compute_preconditioner(sad_init, params)
    sad = apply(preconditioner, sad_init)
    x, y = preconditioned_solution(preconditioner, x_init, y_init)
    sad_gpu = adapt(backend, set_matrix_type(M, set_indtype(Ti, set_eltype(T, sad))))
    x_gpu = adapt(backend, set_eltype(T, x))
    y_gpu = adapt(backend, set_eltype(T, y))
    η = fixed_stepsize(sad, params)
    state = PDHGState(; x = x_gpu, y = y_gpu, η, starting_time)
    return sad_gpu, state
end

function compute_preconditioner(
        sad::SaddlePointProblem,
        params::PDHGParameters
    )
    (; K, Kᵀ) = sad
    (; precond_cb_α) = params
    preconditioner = chambolle_pock_preconditioner(K, Kᵀ; α = precond_cb_α)
    return preconditioner
end

function fixed_stepsize(
        sad::SaddlePointProblem{T},
        params::PDHGParameters
    ) where {T}
    (; K, Kᵀ) = sad
    (; stepsize_scaling) = params
    η = T(stepsize_scaling) * inv(spectral_norm(K, Kᵀ))
    return η
end

function step!(
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

    primal = norm(
        ifelse.(
            ineq_cons,
            positive_part.(q - K * x),
            q - K * x
        )
    )
    primal_scale = one(T) + norm(q)

    dual = norm(c - Kᵀ * y - λ)
    dual_scale = one(T) + norm(c)

    gap = abs(qᵀy + lᵀλ⁺ - uᵀλ⁻ - cᵀx)
    gap_scale = one(T) + abs(qᵀy + lᵀλ⁺ - uᵀλ⁻) + abs(cᵀx)

    weighted_agg = sqrt(ω^2 * primal^2 + inv(ω^2) * dual^2 + gap^2)

    rel_primal = primal / primal_scale
    rel_dual = dual / dual_scale
    rel_gap = gap / gap_scale

    rel_max = max(rel_primal, rel_dual, rel_gap)

    return KKTErrors(;
        primal,
        dual,
        gap,
        primal_scale,
        dual_scale,
        gap_scale,
        rel_max,
        weighted_agg,
    )
end

function prepare_check!(
        state::PDHGState,
        sad::SaddlePointProblem,
        params::PDHGParameters
    )
    (; starting_time) = state
    (; record_error_history) = params
    state.time_elapsed = time() - starting_time
    state.err = kkt_errors(state, sad)
    if record_error_history
        push!(state.error_history, (state.kkt_passes, state.err))
    end
    return nothing
end

function get_results(
        state::PDHGState,
        sad::SaddlePointProblem,
    )
    (; x, y) = state
    (; preconditioner) = sad
    x_cpu, y_cpu = Array(x), Array(y)
    return unpreconditioned_solution(preconditioner, x_cpu, y_cpu), state
end
