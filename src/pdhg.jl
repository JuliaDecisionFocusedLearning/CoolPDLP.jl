"""
    PDHGParameters

Parameters for configuration of the baseline primal-dual hybrid gradient.
    
# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PDHGParameters{T <: Number}
    "scaling of the inverse spectral norm of `K` when defining the step size"
    stepsize_scaling::T = 0.9
    "tolerance when checking KKT relative errors to decide termination"
    tol_termination::T = 1.0e-4
    "maximum number of multiplications by both the KKT matrix `K` and its transpose `Kᵀ`"
    max_kkt_passes::Int = 100_000
    "time limit in seconds"
    time_limit::Float64 = 100.0
    "frequency of termination checks"
    check_every::Int = 40
    "whether or not to record error evolution"
    record_error_history::Bool = false
end

function Base.show(io::IO, params::PDHGParameters)
    (; stepsize_scaling, tol_termination, max_kkt_passes, time_limit, check_every) = params
    return print(
        io,
        "PDHGParameters(; " *
            "stepsize_scaling=$stepsize_scaling, " *
            "tol_termination=$tol_termination, " *
            "max_kkt_passes=$max_kkt_passes, " *
            "time_limit=$time_limit, " *
            "check_every=$check_every" *
            ")"
    )
end

"""
    PDHGState

Current solution, step sizes and various buffers / metrics for the baseline primal-dual hybrid gradient.

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct PDHGState{T <: Number, V <: AbstractVector{T}}
    "current primal solution"
    const x::V
    "current dual solution"
    const y::V
    "step size"
    η::T
    "primal weight"
    ω::T = one(η)
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
    elapsed::Float64 = 0.0
    "number of multiplications by both the KKT matrix and its transpose"
    kkt_passes::Int = 0
    "current relative KKT error"
    relative_error::T = typemax(eltype(x))
    "termination reason (should be `STILL_RUNNING` until the algorithm actuall terminates)"
    termination_reason::TerminationReason = STILL_RUNNING
    "history of relative KKT errors, indexed by number of KKT passes"
    const relative_error_history::Vector{Tuple{Int, T}} = Tuple{Int, eltype(x)}[]
end

function Base.show(io::IO, state::PDHGState)
    (; elapsed, kkt_passes, relative_error, termination_reason) = state
    return print(
        io,
        @sprintf(
            "PDHG state with termination reason %s: %.2e relative KKT error after %g seconds elapsed and %s KKT passes",
            termination_reason,
            relative_error,
            elapsed,
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
        milp::MILP{T},
        params::PDHGParameters,
        x_init::AbstractVector{T} = zero(milp.c);
        show_progress::Bool = true
    ) where {T}
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
"""
function pdhg(
        sad::SaddlePointProblem,
        params::PDHGParameters,
        x_init::AbstractVector{T} = zero(sad.c),
        y_init::AbstractVector{T} = zero(sad.q);
        show_progress::Bool = true,
        starting_time::Float64 = time()
    ) where {T}
    x, y = copy(x_init), copy(y_init)
    η = fixed_stepsize(sad, params)
    state = PDHGState(; x, y, η, starting_time)
    prog = ProgressUnknown(desc = "PDHG iterations:", enabled = show_progress)
    while true
        yield()
        next!(prog; showvalues = (("relative_error", state.relative_error),))
        pdhg_step!(state, sad)
        if (state.kkt_passes % params.check_every == 0) &&
                termination_check!(state, sad, params)
            break
        end
    end
    finish!(prog)
    return state
end

function fixed_stepsize(sad::SaddlePointProblem{T}, params::PDHGParameters) where {T}
    (; K, Kᵀ) = sad
    (; stepsize_scaling) = params
    η = T(stepsize_scaling) * inv(spectral_norm(K, Kᵀ))
    return η
end

function proj_X!(
        x::AbstractVector,
        l::AbstractVector,
        u::AbstractVector
    )
    return x .= proj_box.(x, l, u)
end

function proj_Y!(
        y::AbstractVector,
        m₁::Ti
    ) where {Ti <: Integer}
    return y[Ti(1):m₁] .= positive_part.(@view(y[Ti(1):m₁]))
end

function proj_Λ!(
        λ::AbstractVector,
        l::AbstractVector,
        u::AbstractVector
    )
    return λ .= proj_λ.(λ, l, u)
end


function pdhg_step!(
        state::PDHGState{T},
        sad::SaddlePointProblem{T, Ti},
    ) where {T, Ti}
    (; x, y, η, ω, x_scratch1, y_scratch) = state
    (; c, q, K, Kᵀ, l, u, m₁) = sad

    xp, yp = x_scratch1, y_scratch
    τ, σ = η / ω, η * ω

    # xp = proj_X(x - τ * (c - Kᵀ * y))
    xp .= x .- τ .* c
    mul!(xp, Kᵀ, y, τ, Ti(1))
    proj_X!(xp, l, u)

    # yp = proj_Y(y + σ * (q - K * (2 * xp - x)))
    yp .= y .+ σ .* q
    x .= 2 .* xp .- x
    mul!(yp, K, x, -σ, Ti(1))
    proj_Y!(yp, m₁)

    copyto!(x, xp)
    copyto!(y, yp)

    state.kkt_passes += 1
    return nothing
end


function individual_kkt_errors!(
        state::PDHGState{T},
        sad::SaddlePointProblem{T, Ti},
    ) where {T, Ti}
    (; x, y, x_scratch1, x_scratch2, x_scratch3, y_scratch) = state
    (; c, q, K, Kᵀ, l, u, m₁, m₂) = sad

    qᵀy = dot(q, y)
    cᵀx = dot(c, x)

    # Kxᵀ = (Gxᵀ, Axᵀ), qᵀ = (hᵀ, bᵀ)
    Kx = y_scratch
    mul!(Kx, K, x)
    Gx = @view Kx[Ti(1):m₁]
    h = @view q[Ti(1):m₁]
    Ax = @view Kx[(m₁ + Ti(1)):(m₁ + m₂)]
    b = @view q[(m₁ + Ti(1)):(m₁ + m₂)]

    # λ = proj_Λ(c - Kᵀ * y)  from cuPDLP-C paper
    λ = x_scratch1
    λ .= c
    mul!(λ, Kᵀ, y, -Ti(1), Ti(1))
    proj_Λ!(λ, l, u)

    λ⁺, l_noinf = x_scratch2, x_scratch3
    λ⁺ .= positive_part.(λ)
    l_noinf .= ifelse.(iszero.(λ⁺), zero(T), l)
    lᵀλ⁺ = dot(l_noinf, λ⁺)

    λ⁻, u_noinf = x_scratch2, x_scratch3
    λ⁻ .= negative_part.(λ)
    u_noinf .= ifelse.(iszero.(λ⁻), zero(T), u)
    uᵀλ⁻ = dot(u_noinf, λ⁻)

    # err_primal = sqrt(sqnorm(Ax - b) + sqnorm((h - Gx)⁺)
    err_primal_scratch = y_scratch
    err_primal_scratch[Ti(1):m₁] .= positive_part.(h .- Gx)
    err_primal_scratch[(m₁ + Ti(1)):(m₁ + m₂)] .= Ax .- b

    err_primal = norm(err_primal_scratch)
    err_primal_denominator = one(T) + norm(q)

    # err_dual = norm(c - Kᵀ * y - λ)
    err_dual_scratch = x_scratch1
    err_dual_scratch .= c .- λ
    mul!(err_dual_scratch, Kᵀ, y, -Ti(1), Ti(1))

    err_dual = norm(err_dual_scratch)
    err_dual_denominator = one(T) + norm(c)

    err_gap = abs(qᵀy + lᵀλ⁺ - uᵀλ⁻ - cᵀx)
    err_gap_denominator = one(T) + abs(qᵀy + lᵀλ⁺ - uᵀλ⁻) + abs(cᵀx)

    return (;
        err_primal,
        err_dual,
        err_gap,
        err_primal_denominator,
        err_dual_denominator,
        err_gap_denominator,
    )
end


function relative_kkt_error!(
        state::PDHGState{T},
        sad::SaddlePointProblem{T},
    ) where {T}
    (;
        err_primal, err_dual, err_gap,
        err_primal_denominator, err_dual_denominator, err_gap_denominator,
    ) = individual_kkt_errors!(state, sad)

    relative_erroror_primal = err_primal / err_primal_denominator
    relative_erroror_dual = err_dual / err_dual_denominator
    relative_erroror_gap = err_gap / err_gap_denominator

    return max(relative_erroror_primal, relative_erroror_dual, relative_erroror_gap)
end


function termination_check!(
        state::PDHGState{T},
        sad::SaddlePointProblem{T},
        params::PDHGParameters,
    ) where {T}
    (; starting_time) = state
    (; tol_termination, time_limit, max_kkt_passes, record_error_history) = params
    state.elapsed = time() - starting_time
    state.relative_error = relative_kkt_error!(state, sad)
    if record_error_history
        push!(state.relative_error_history, (state.kkt_passes, state.relative_error))
    end

    if state.relative_error <= tol_termination
        state.termination_reason = CONVERGENCE
        return true
    elseif state.elapsed >= time_limit
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
