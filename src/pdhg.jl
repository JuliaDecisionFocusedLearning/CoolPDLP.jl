"""
    PDHGParameters

Parameters for configuration of the baseline primal-dual hybrid gradient.
    
# Fields

$(TYPEDFIELDS)
"""
@kwdef struct PDHGParameters{T <: Number}
    "tolerance when checking KKT relative errors to decide termination"
    tol_termination::T = 1.0e-4
    "maximum number of multiplications by both the KKT matrix `K` and its transpose `Kᵀ`"
    max_kkt_passes::Int = 100_000
    "time limit in seconds"
    time_limit::Float64 = 100.0
    "frequency of termination checks"
    check_every::Int = 40
end

function Base.show(io::IO, params::PDHGParameters)
    (; tol_termination, max_kkt_passes, time_limit, check_every) = params
    return print(io, "PDHGParameters(; tol_termination=$tol_termination, max_kkt_passes=$max_kkt_passes, time_limit=$time_limit, check_every=$check_every)")
end

"""
    TerminationReason

Enum type listing possible reasons for algorithm termination:

- `CONVERGENCE`
- `TIME`
- `ITERATIONS`
- `STILL_RUNNING`
"""
@enum TerminationReason CONVERGENCE TIME ITERATIONS STILL_RUNNING

"""
    PDHGState

Current solution, step sizes and various buffers / metrics for the baseline primal-dual hybrid gradient.

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct PDHGState{T <: Number, V <: AbstractVector{T}}
    "current solution"
    z::PrimalDualVariable{T, V}
    "step size"
    η::T
    "primal weight"
    ω::T = one(η)
    "buffer"
    z_scratch::PrimalDualVariable{T, V} = copy(z)
    "buffer"
    λ_scratch::V = copy(z.x)
    "time at which the algorithm started, in seconds"
    starting_time::Float64 = time()
    "time elapsed since the algorithm started, in seconds"
    elapsed::Float64 = 0.0
    "number of multiplications by both the KKT matrix and its transpose"
    kkt_passes::Int = 0
    "current relative KKT error"
    rel_err::T = typemax(typeof(η))
    "termination reason (should be `STILL_RUNNING` until the algorithm actuall terminates)"
    termination_reason::TerminationReason = STILL_RUNNING
end

function Base.show(io::IO, state::PDHGState)
    (; elapsed, kkt_passes, rel_err, termination_reason) = state
    return print(io, "PDHG state with termination reason $termination_reason: $rel_err relative KKT error after $elapsed seconds elapsed and $kkt_passes KKT passes")
end

"""
    pdhg(
        sad::SaddlePointProblem,
        params::PDHGParameters,
        z_init::PrimalDualVariable;
        show_progress::Bool=true
    )
    
Apply the primal-dual hybrid gradient algorithm to solve `sad` using configuration `params`, starting from `z_init`.
"""
function pdhg(
        sad::SaddlePointProblem{T},
        params::PDHGParameters,
        z_init::PrimalDualVariable{T} = default_init(sad);
        show_progress::Bool = true
    ) where {T}
    (; K, Kᵀ) = sad
    z = copy(z_init)
    η = T(0.9) * inv(spectral_norm(K, Kᵀ))
    state = PDHGState(; z, η)
    prog = ProgressUnknown(desc = "PDHG iterations:", enabled = show_progress)
    while true
        yield()
        next!(prog; showvalues = (("rel_err", state.rel_err),))
        pdhg_step!(state, sad)
        if (state.kkt_passes % params.check_every == 0) &&
                termination_check!(state, sad, params)
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
    (; z, η, ω, z_scratch) = state
    (; c, q, K, Kᵀ, l, u, m₁) = sad
    (; x, y) = z
    xp, yp = z_scratch.x, z_scratch.y
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
    (; z, z_scratch, λ_scratch) = state
    (; c, q, K, Kᵀ, l, u, m₁, m₂) = sad
    (; x, y) = z
    x_scratch, y_scratch = z_scratch.x, z_scratch.y

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
    λ = x_scratch
    λ .= c
    mul!(λ, Kᵀ, y, -Ti(1), Ti(1))
    proj_Λ!(λ, l, u)
    λ_scratch .= positive_part.(λ)
    lᵀλ⁺ = dot(l, λ_scratch)
    λ_scratch .= negative_part.(λ)
    uᵀλ⁻ = dot(u, λ_scratch)

    # err_primal = sqrt(sqnorm(Ax - b) + sqnorm((h - Gx)⁺)
    err_primal_scratch = y_scratch
    err_primal_scratch[Ti(1):m₁] .= positive_part.(h .- Gx)
    err_primal_scratch[(m₁ + Ti(1)):(m₁ + m₂)] .= Ax .- b

    err_primal = norm(err_primal_scratch)
    err_primal_denominator = one(T) + norm(q)

    # err_dual = norm(c - Kᵀ * y - λ)
    err_dual_scratch = x_scratch
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

    rel_error_primal = err_primal / err_primal_denominator
    rel_error_dual = err_dual / err_dual_denominator
    rel_error_gap = err_gap / err_gap_denominator

    return max(rel_error_primal, rel_error_dual, rel_error_gap)
end


function termination_check!(
        state::PDHGState{T},
        sad::SaddlePointProblem{T},
        params::PDHGParameters,
    ) where {T}
    (; starting_time) = state
    (; tol_termination, time_limit, max_kkt_passes) = params
    state.elapsed = time() - starting_time
    state.rel_err = relative_kkt_error!(state, sad)
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
