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
    norm_A = spectral_norm(A, At)
    return @. T(invnorm_scaling) * inv(norm_A)
end

function primal_weight_init(milp::MILP{T}, params::StepSizeParameters) where {T}
    (; c, lc, uc) = milp
    (; zero_tol) = params
    c_norm = colnorm(c)
    combined_bounds = map(combine, lc, uc)
    combined_norm = colnorm(combined_bounds)
    return @. ifelse(
        (c_norm > zero_tol) & (combined_norm > zero_tol),
        c_norm / combined_norm,
        one(T)
    )
end

"""
    StepSizes

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct StepSizes{T <: BatchedNumber}
    "step size"
    η::T
    "cumulated step size since last restart"
    η_sum::T = zero(η)
    "primal weight"
    ω::T
end

batch(step_sizes::StepSizes, i::Int) = StepSizes(
    batch_num(step_sizes.η, i),
    batch_num(step_sizes.η_sum, i),
    batch_num(step_sizes.ω, i),
)

add_stepsize!(step_sizes::StepSizes{<:Number}) = (step_sizes.η_sum += step_sizes.η; nothing)
add_stepsize!(step_sizes::StepSizes{<:AbstractVector}) = (step_sizes.η_sum .+= step_sizes.η; nothing)

reset_stepsize!(step_sizes::StepSizes{<:Number}) = (step_sizes.η_sum = zero(step_sizes.η_sum); nothing)
reset_stepsize!(step_sizes::StepSizes{<:AbstractVector}) = (zero!(step_sizes.η_sum); nothing)

function primal_weight_update!(
        scratch::Scratch,
        step_sizes::StepSizes,
        sol_cand::PrimalDualSolution,
        sol_restart::PrimalDualSolution,
        params::StepSizeParameters
    )
    (; ω) = step_sizes
    (; primal_weight_damping, zero_tol) = params
    Δx = colnorm(@. scratch.x = sol_cand.x - sol_restart.x)
    Δy = colnorm(@. scratch.y = sol_cand.y - sol_restart.y)
    θ = primal_weight_damping
    return @. ifelse(
        (Δx > zero_tol) & (Δy > zero_tol),
        exp(θ * log(Δy / Δx) + (1 - θ) * log(ω)),
        ω
    )
end
