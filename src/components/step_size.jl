"""
    StepSizeParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct StepSizeParameters{T <: Number}
    "scaling of the inverse spectral norm of `A` when defining the non-adaptive step size"
    invnorm_scaling::T
    "primal weight update damping"
    primal_weight_damping::T
    "tolerance in absolute comparisons to zero"
    zero_tol::T
end

function Base.show(io::IO, params::StepSizeParameters)
    (; invnorm_scaling, primal_weight_damping, zero_tol) = params
    return print(io, "StepSizeParameters: invnorm_scaling=$invnorm_scaling, primal_weight_damping=$primal_weight_damping, zero_tol=$zero_tol")
end

"""
    compute_eta(norm_A, norm_Q, ω, invnorm_scaling)

Compute step size `η` satisfying the Vu-Condat convergence condition:
`η²‖A‖² + (η/(2ω))‖Q‖ ≤ 1`.
"""
function compute_eta(norm_A::T, norm_Q::T, ω::T, invnorm_scaling::T) where {T}
    a = norm_A^2
    b = norm_Q / (2ω)
    if iszero(a) && iszero(b)
        return one(T)
    elseif iszero(a)
        return invnorm_scaling * 2ω / norm_Q
    else
        η_max = 2 / (hypot(b, 2norm_A) + b)
        return invnorm_scaling * η_max
    end
end

function primal_weight_init(milp::AbstractProgram{T}, params::StepSizeParameters) where {T}
    (; c, lc, uc) = milp
    (; zero_tol) = params
    c_norm = norm(c)
    combined_bounds = map(combine, lc, uc)
    combined_norm = norm(combined_bounds)
    if c_norm > zero_tol && combined_norm > zero_tol
        return c_norm / combined_norm
    else
        return one(T)
    end
end

function init_stepsize(milp::LinearProgram{T}, params::StepSizeParameters) where {T}
    (; A, At) = milp
    norm_A = T(spectral_norm(A, At))
    ω = one(T)
    η = T(params.invnorm_scaling) * inv(norm_A)
    return η, ω, norm_A, norm_Q
end

function init_stepsize(milp::QuadraticProgram{T}, params::StepSizeParameters) where {T}
    (; A, At, Q) = milp
    norm_A = T(spectral_norm(A, At))
    norm_Q = T(spectral_norm(Q, Q))
    ω = primal_weight_init(milp, params)
    η = compute_eta(norm_A, norm_Q, ω, T(params.invnorm_scaling))
    return η, ω, norm_A, norm_Q
end

"""
    StepSizes

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct StepSizes{T <: Number}
    "step size"
    η::T
    "cumulated step size since last restart"
    η_sum::T = zero(η)
    "primal weight"
    ω::T
    "spectral norm of A"
    norm_A::T = zero(η)
    "spectral norm of Q"
    norm_Q::T = zero(η)
end

update_step_size!(step_sizes::StepSizes, ::LinearProgram, ::StepSizeParameters) = (step_sizes.η, step_sizes.ω)
function update_step_size!(step_sizes::StepSizes, ::QuadraticProgram, params::StepSizeParameters)
    step_sizes.η = compute_eta(
        step_sizes.norm_A, step_sizes.norm_Q, step_sizes.ω, params.invnorm_scaling,
    )
    return step_sizes.η, step_sizes.ω
end

function primal_weight_update!(
        scratch::Scratch,
        step_sizes::StepSizes,
        sol_cand::PrimalDualSolution,
        sol_restart::PrimalDualSolution,
        params::StepSizeParameters
    )
    (; ω) = step_sizes
    (; primal_weight_damping, zero_tol) = params
    Δx = norm(@. scratch.x = sol_cand.x - sol_restart.x)
    Δy = norm(@. scratch.y = sol_cand.y - sol_restart.y)
    θ = primal_weight_damping
    if Δx > zero_tol && Δy > zero_tol
        return exp(θ * log(Δy / Δx) + (1 - θ) * log(ω))
    else
        return ω
    end
end
