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

function fixed_stepsize(milp::MILP{T}, params::StepSizeParameters) where {T}
    (; A, At) = milp
    (; invnorm_scaling) = params
    η = T(invnorm_scaling) * inv(spectral_norm(A, At))
    return η
end

function primal_weight_init(milp::MILP{T}, params::StepSizeParameters) where {T}
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
