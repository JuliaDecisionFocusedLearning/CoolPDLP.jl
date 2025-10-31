@kwdef struct PDHGParameters{T <: Number}
    ϵ_termination::T = 1.0e-4
    max_kkt_passes::Int = 100_000
    time_limit::Float64 = 100.0
    check_every::Int = 40
end

@enum TerminationReason CONVERGENCE TIME ITERATIONS STILL_RUNNING

@kwdef mutable struct PDHGState{T <: Number, V <: AbstractVector{T}}
    z::PrimalDualVariable{T, V}
    η::T
    ω::T = one(η)
    starting_time::Float64 = time()
    elapsed::Float64 = 0.0
    kkt_passes::Int = 0
    rel_err::T = NaN
    termination_reason::TerminationReason = STILL_RUNNING
end


function pdhg(
        problem::SaddlePointProblem{T},
        params::PDHGParameters{T},
        z_init::PrimalDualVariable{T} = default_init(problem);
        show_progress::Bool = true
    ) where {T}
    (; K) = problem
    z = copy(z_init)
    η = T(0.9) * inv(spectral_norm(K))
    state = PDHGState(; z, η)
    prog = ProgressUnknown(desc = "PDHG iterations:", enabled = show_progress)
    while true
        yield()
        next!(prog, showvalues = [("rel_err", state.rel_err)])
        pdhg_step!(state, problem)
        if (state.kkt_passes % params.check_every == 0) &&
                termination_check!(state, problem, params)
            break
        end
    end
    finish!(prog)
    return state
end


function proj_X(problem::SaddlePointProblem{T}, x::AbstractVector{T}) where {T}
    (; l, u) = problem
    return proj_box(x, l, u)
end

function proj_Y(problem::SaddlePointProblem{T}, y::AbstractVector{T}) where {T}
    (; m₁, m₂) = problem
    return vcat(positive_part(y[1:m₁]), y[(m₁ + 1):(m₁ + m₂)])
end

function proj_λ(lᵢ::T, uᵢ::T, λᵢ::T) where {T}
    if lᵢ == typemin(T) && uᵢ == typemax(T)
        return zero(T)  # project on {0}
    elseif lᵢ == typemin(T)
        return -negative_part(λᵢ)  # project on ℝ⁻
    elseif uᵢ == typemax(T)
        return positive_part(λᵢ)  # project on ℝ⁺
    else
        return λᵢ  # project on ℝ
    end
end

function proj_Λ(problem::SaddlePointProblem{T}, λ::AbstractVector{T}) where {T}
    (; l, u) = problem
    return map(proj_λ, l, u, λ)
end


function pdhg_step!(
        state::PDHGState{T},
        problem::SaddlePointProblem{T},
    ) where {T}
    (; z, η, ω) = state
    (; c, q, K, Kᵀ) = problem
    (; x, y) = z
    τ, σ = η / ω, η * ω
    xp = proj_X(problem, x - τ * (c - Kᵀ * y))
    yp = proj_Y(problem, y + σ * (q - K * (2 * xp - x)))
    zp = PrimalDualVariable(xp, yp)
    copyto!(z, zp)
    state.kkt_passes += 1
    return nothing
end


function individual_kkt_errors(
        state::PDHGState{T},
        problem::SaddlePointProblem{T},
    ) where {T}
    (; z) = state
    (; c, q, K, Kᵀ, l, u, m₁, m₂) = problem
    (; x, y) = z

    Kx = K * x
    Gx, h = Kx[1:m₁], q[1:m₁]
    Ax, b = Kx[(m₁ + 1):(m₁ + m₂)], q[(m₁ + 1):(m₁ + m₂)]

    λ = proj_Λ(problem, c - Kᵀ * y)  # from cuPDLP-C paper
    λ⁺ = positive_part(λ)
    λ⁻ = negative_part(λ)

    qᵀy = dot(q, y)
    lᵀλ⁺ = dot(l, λ⁺)
    uᵀλ⁻ = dot(u, λ⁻)
    cᵀx = dot(c, x)

    err_primal = sqrt(sqnorm(Ax - b) + sqnorm(positive_part(h - Gx)))
    err_dual = norm(c - Kᵀ * y - λ)
    err_gap = abs(qᵀy + lᵀλ⁺ - uᵀλ⁻ - cᵀx)

    err_primal_denominator = 1 + norm(q)
    err_dual_denominator = 1 + norm(c)
    err_gap_denominator = 1 + abs(qᵀy + lᵀλ⁺ - uᵀλ⁻) + abs(cᵀx)

    return (;
        err_primal,
        err_dual,
        err_gap,
        err_primal_denominator,
        err_dual_denominator,
        err_gap_denominator,
    )
end


function relative_kkt_error(
        state::PDHGState{T},
        problem::SaddlePointProblem{T},
    ) where {T}
    (;
        err_primal, err_dual, err_gap,
        err_primal_denominator, err_dual_denominator, err_gap_denominator,
    ) = individual_kkt_errors(state, problem)

    rel_error_primal = err_primal / err_primal_denominator
    rel_error_dual = err_dual / err_dual_denominator
    rel_error_gap = err_gap / err_gap_denominator

    return max(rel_error_primal, rel_error_dual, rel_error_gap)
end


function termination_check!(
        state::PDHGState{T},
        problem::SaddlePointProblem{T},
        params::PDHGParameters{T},
    ) where {T}
    (; starting_time) = state
    (; ϵ_termination, time_limit, max_kkt_passes) = params
    state.elapsed = time() - starting_time
    state.rel_err = relative_kkt_error(state, problem)
    if state.rel_err <= ϵ_termination
        state.termination_reason = CONVERGENCE
        return true
    elseif state.elapsed > time_limit
        state.termination_reason = TIME
        return true
    elseif state.kkt_passes > max_kkt_passes
        state.termination_reason = ITERATIONS
        return true
    else
        state.termination_reason = STILL_RUNNING
        return false
    end
end
