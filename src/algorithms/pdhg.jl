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
    precond_cb_alpha::T
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
            _T::Type{T} = Float64,
            ::Type{Ti} = Int,
            ::Type{M} = SparseMatrixCSC,
            backend::B = CPU();
            stepsize_scaling = _T(0.9),
            precond_cb_alpha = _T(1.0),
            termination_reltol = _T(1.0e-4),
            max_kkt_passes = 100_000,
            time_limit = 100.0,
            check_every = 100,
            record_error_history = false,
        ) where {T, Ti, M, B}
        return new{T, Ti, M, B}(
            backend,
            stepsize_scaling,
            precond_cb_alpha,
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

Current solution, step sizes and various buffers / metrics for the PDHG algorithm.

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
        x_init=zero(milp.lv),
        y_init=zero(milp.lc);
        show_progress::Bool=true
    )
    
Apply the PDHG algorithm to solve the continuous relaxation of `milp` using configuration `params`, starting from primal variable `x_init` and dual variable `y_init`.

Return a triplet `(x, y, state)` where `x` is the primal solution, `y` is the dual solution and `state` is the algorithm's final state, including convergence information.
"""
function pdhg(
        milp_init::MILP,
        params::PDHGParameters,
        x_init::AbstractVector = zero(milp_init.lv),
        y_init::AbstractVector = zero(milp_init.lc);
        show_progress::Bool = true,
        starting_time::Float64 = time()
    )
    milp, x, y = preprocess(milp_init, params, x_init, y_init)
    state = pdhg_preprocessed(milp, params, x, y; show_progress, starting_time)
    return get_solution(state, milp), state
end

function preprocess(
        milp_init::MILP,
        params::PDHGParameters{T, Ti, M},
        x_init::AbstractVector,
        y_init::AbstractVector,
    ) where {T, Ti, M}
    (; backend) = params
    D1, D2 = compute_preconditioner(milp_init, params)

    milp_precond = precondition(milp_init, D1, D2)
    milp_righttypes = set_matrix_type(M, set_indtype(Ti, set_eltype(T, milp_precond)))
    milp_adapted = adapt(backend, milp_righttypes)

    x_precond = D2 \ x_init
    x_righttype = set_eltype(T, x_precond)
    x_adapted = adapt(backend, x_righttype)

    y_precond = D1 * y_init
    y_righttype = set_eltype(T, y_precond)
    y_adapted = adapt(backend, y_righttype)

    return milp_adapted, x_adapted, y_adapted
end

function compute_preconditioner(
        milp::MILP,
        params::PDHGParameters
    )
    (; A, At) = milp
    (; precond_cb_alpha) = params
    D1, D2 = chambolle_pock_preconditioner(A, At; alpha = precond_cb_alpha)
    return D1, D2
end

function pdhg_preprocessed(
        milp::MILP{T, V},
        params::PDHGParameters{T},
        x::V,
        y::V;
        show_progress::Bool,
        starting_time::Float64
    ) where {T, V}
    state = initialize(milp, params, x, y; starting_time)
    prog = ProgressUnknown(desc = "PDHG iterations:", enabled = show_progress)
    while true
        yield()
        for _ in 1:params.check_every
            step!(state, milp)
            next!(prog; showvalues = (("relative_error", relative(state.err)),))
        end
        prepare_check!(state, milp, params)
        if termination_check!(state, params)
            break
        end
    end
    finish!(prog)
    return state
end

function initialize(
        milp::MILP{T, V},
        params::PDHGParameters{T, Ti, M},
        x::V,
        y::V;
        starting_time::Float64
    ) where {T, Ti, V, M}
    η = fixed_stepsize(milp, params)
    state = PDHGState(; x, y, η, starting_time)
    return state
end

function fixed_stepsize(
        milp::MILP{T},
        params::PDHGParameters{T}
    ) where {T}
    (; A, At) = milp
    (; stepsize_scaling) = params
    η = T(stepsize_scaling) * inv(spectral_norm(A, At))
    return η
end

function step!(
        state::PDHGState{T, V},
        milp::MILP{T, V},
    ) where {T, V}
    (; x, y, η, ω) = state
    (; c, lv, uv, A, At, lc, uc) = milp

    τ, σ = η / ω, η * ω

    # xp = proj_X(x - τ * (c - At * y))
    x_next = proj_box.(x - τ * (c - At * y), lv, uv)

    # yp = proj_Y(y + σ * (q - K * (2 * xp - x)))
    Axdiff = A * (2 * x_next - x)
    y_next = y - σ * Axdiff - σ * proj_box.(inv(σ) * y - Axdiff, -uc, -lc)

    copy!(x, x_next)
    copy!(y, y_next)

    state.kkt_passes += 1
    return nothing
end

function kkt_errors(
        state::PDHGState,
        milp::MILP{T, V},
    ) where {T, V}
    # TODO: go back to initial problem
    (; x, y, ω) = state
    (; c, lv, uv, A, At, lc, uc) = milp

    Ax = A * x
    r = proj_multiplier.(c - At * y, lv, uv)

    pc = p(-y, lc, uc)
    pv = p(-r, lv, uv)

    primal = norm(Ax - proj_box.(Ax, lc, uc))
    primal_scale = one(T) + norm(bound_scale.(lc, uc))

    dual = norm(c - At * y - r)
    dual_scale = one(T) + norm(c)

    gap = abs(dot(c, x) + pc + pv)
    gap_scale = one(T) + abs(pc + pv) + abs(dot(c, x))

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
        milp::MILP,
        params::PDHGParameters
    )
    (; starting_time) = state
    (; record_error_history) = params
    state.time_elapsed = time() - starting_time
    state.err = kkt_errors(state, milp)
    if record_error_history
        push!(state.error_history, (state.kkt_passes, state.err))
    end
    return nothing
end

function get_solution(
        state::PDHGState,
        milp::MILP,
    )
    (; x, y) = state
    (; D1, D2) = milp
    return D2 * x, D1 \ y
end
