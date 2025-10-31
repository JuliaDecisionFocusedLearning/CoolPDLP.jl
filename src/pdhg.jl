@kwdef struct PDHGParameters{T <: Number}
    tol_termination::T = 1.0e-4
    max_kkt_passes::Int = 100_000
    time_limit::Float64 = 100.0
    check_every::Int = 40
end

function change_eltype(::Type{T}, params::PDHGParameters) where {T}
    (; tol_termination, max_kkt_passes, time_limit, check_every) = params
    return PDHGParameters(;
        tol_termination = T(tol_termination),
        max_kkt_passes,
        time_limit,
        check_every
    )
end

@enum TerminationReason CONVERGENCE TIME ITERATIONS STILL_RUNNING

@kwdef mutable struct PDHGState{T <: Number, V <: AbstractVector{T}}
    z::PrimalDualVariable{T, V}
    η::T
    ω::T = one(η)
    z_scratch::PrimalDualVariable{T, V} = copy(z)
    λ_scratch::V = copy(z.x)
    starting_time::Float64 = time()
    elapsed::Float64 = 0.0
    kkt_passes::Int = 0
    rel_err::T = typemax(typeof(η))
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
    showvalues = [("rel_err", state.rel_err)]
    while true
        yield()
        showvalues[1] = (showvalues[1][1], state.rel_err)
        next!(prog; showvalues)
        pdhg_step!(state, problem)
        if (state.kkt_passes % params.check_every == 0) &&
                termination_check!(state, problem, params)
            break
        end
    end
    finish!(prog)
    return state
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
        m₁::Int
    )
    return y[1:m₁] .= positive_part.(@view(y[1:m₁]))
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
        problem::SaddlePointProblem{T},
    ) where {T}
    (; z, η, ω, z_scratch) = state
    (; c, q, K, Kᵀ, l, u, m₁) = problem
    (; x, y) = z
    xp, yp = z_scratch.x, z_scratch.y
    τ, σ = η / ω, η * ω

    # xp = proj_X(x - τ * (c - Kᵀ * y))
    xp .= x .- τ .* c
    mul!(xp, Kᵀ, y, τ, 1)
    proj_X!(xp, l, u)

    # yp = proj_Y(y + σ * (q - K * (2 * xp - x)))
    yp .= y .+ σ .* q
    x .= 2 .* xp .- x
    mul!(yp, K, x, -σ, 1)
    proj_Y!(yp, m₁)

    copyto!(x, xp)
    copyto!(y, yp)

    state.kkt_passes += 1
    return nothing
end


function individual_kkt_errors!(
        state::PDHGState{T},
        problem::SaddlePointProblem{T},
    ) where {T}
    (; z, z_scratch, λ_scratch) = state
    (; c, q, K, Kᵀ, l, u, m₁, m₂) = problem
    (; x, y) = z
    x_scratch, y_scratch = z_scratch.x, z_scratch.y

    qᵀy = dot(q, y)
    cᵀx = dot(c, x)

    # Kxᵀ = (Gxᵀ, Axᵀ), qᵀ = (hᵀ, bᵀ)
    Kx = y_scratch
    mul!(Kx, K, x)
    Gx = @view Kx[1:m₁]
    h = @view q[1:m₁]
    Ax = @view Kx[(m₁ + 1):(m₁ + m₂)]
    b = @view q[(m₁ + 1):(m₁ + m₂)]

    # λ = proj_Λ(c - Kᵀ * y)  from cuPDLP-C paper
    λ = x_scratch
    λ .= c
    mul!(λ, Kᵀ, y, -1, 1)
    proj_Λ!(λ, l, u)
    λ_scratch .= positive_part.(λ)
    lᵀλ⁺ = dot(l, λ_scratch)
    λ_scratch .= negative_part.(λ)
    uᵀλ⁻ = dot(u, λ_scratch)

    # err_primal = sqrt(sqnorm(Ax - b) + sqnorm((h - Gx)⁺)
    err_primal_scratch = y_scratch
    err_primal_scratch[1:m₁] .= positive_part.(h .- Gx)
    err_primal_scratch[(m₁ + 1):(m₁ + m₂)] .= Ax .- b

    err_primal = norm(err_primal_scratch)
    err_primal_denominator = 1 + norm(q)

    # err_dual = norm(c - Kᵀ * y - λ)
    err_dual_scratch = x_scratch
    err_dual_scratch .= c .- λ
    mul!(err_dual_scratch, Kᵀ, y, -1, 1)

    err_dual = norm(err_dual_scratch)
    err_dual_denominator = 1 + norm(c)

    err_gap = abs(qᵀy + lᵀλ⁺ - uᵀλ⁻ - cᵀx)
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


function relative_kkt_error!(
        state::PDHGState{T},
        problem::SaddlePointProblem{T},
    ) where {T}
    (;
        err_primal, err_dual, err_gap,
        err_primal_denominator, err_dual_denominator, err_gap_denominator,
    ) = individual_kkt_errors!(state, problem)

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
    (; tol_termination, time_limit, max_kkt_passes) = params
    state.elapsed = time() - starting_time
    state.rel_err = relative_kkt_error!(state, problem)
    if state.rel_err <= tol_termination
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
