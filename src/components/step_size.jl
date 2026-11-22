"""
    StepSizeParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct StepSizeParameters{T <: Number, I <: Number}
    "scaling of the inverse spectral norm of `A` when defining the non-adaptive step size"
    invnorm_scaling::T
    "primal weight update damping"
    primal_weight_damping::T
    "tolerance in absolute comparisons to zero"
    zero_tol::T
    """
    relative residual tolerance in the iterative (Lanczos) spectral norm computation.

    !!! warning "Breaking change"
        This used to be the absolute tolerance on the eigenvalue estimate of `AᵀA` in the power
        method. It is now the relative Lanczos residual `‖AᵀA y - θ y‖ ≤ tol * θ` (see
        [`spectral_norm`](@ref)), with the same default `1e-3`: a custom value tuned for the
        old meaning must be revisited. With the default, the converged estimate of `‖A‖₂` lies
        below the true norm by at most about `tol / 2` relative, which `invnorm_scaling`
        must absorb (`invnorm_scaling ≤ 1 - tol / 2`), unless `spectral_norm_upper` is set.
    """
    spectral_norm_tol::T
    "maximum number of Lanczos iterations in the iterative spectral norm computation"
    spectral_norm_maxiter::I
    """
    return the upper estimate `sqrt(θ + residual)` instead of the Ritz value `sqrt(θ)` once the
    spectral norm computation converged (see [`spectral_norm`](@ref)). Turn it on when
    `invnorm_scaling` is close to `1`, where the Ritz value leaves too little margin.
    """
    spectral_norm_upper::Bool = false
end

function Base.show(io::IO, params::StepSizeParameters)
    (;
        invnorm_scaling, primal_weight_damping, zero_tol, spectral_norm_tol, spectral_norm_maxiter,
        spectral_norm_upper,
    ) = params
    return print(io, "StepSizeParameters: invnorm_scaling=$invnorm_scaling, primal_weight_damping=$primal_weight_damping, zero_tol=$zero_tol, spectral_norm_tol=$spectral_norm_tol, spectral_norm_maxiter=$spectral_norm_maxiter, spectral_norm_upper=$spectral_norm_upper")
end

"""
    spectral_norm(K, Kᵀ, params::StepSizeParameters)

Call [`spectral_norm`](@ref) with the tolerance, budget and estimate of `params`.
"""
function spectral_norm(K::AbstractMatrix, Kᵀ::AbstractMatrix, params::StepSizeParameters)
    (; spectral_norm_tol, spectral_norm_maxiter, spectral_norm_upper) = params
    return spectral_norm(K, Kᵀ; tol = spectral_norm_tol, maxiter = spectral_norm_maxiter, upper = spectral_norm_upper)
end

function fixed_stepsize(milp::MILP{T}, params::StepSizeParameters) where {T}
    (; A, At) = milp
    (; invnorm_scaling) = params
    norm = spectral_norm(A, At, params)
    # a zero spectral norm (no constraint rows, or an all-zero `A`) means `A'y` never
    # contributes to the primal step regardless of `η`, so any finite step size is safe;
    # `inv(norm)` would otherwise be `Inf` and corrupt the first primal step
    η = iszero(norm) ? one(T) : T(invnorm_scaling) * inv(norm)
    return η
end

function primal_weight_init(milp::MILP{T}, params::StepSizeParameters) where {T}
    (; c, lc, uc) = milp
    (; zero_tol) = params
    c_norm = colnorm(c)
    # a broadcast rather than a `map`, because one bound may be batched and the other shared
    combined_bounds = combine.(lc, uc)
    combined_norm = colnorm(combined_bounds)
    return broadcast(c_norm, combined_norm) do c_norm, combined_norm
        if c_norm > zero_tol && combined_norm > zero_tol
            c_norm / combined_norm
        else
            one(T)
        end
    end
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

instance(step_sizes::StepSizes, i::Int) = StepSizes(
    instance_num(step_sizes.η, i),
    instance_num(step_sizes.η_sum, i),
    instance_num(step_sizes.ω, i),
)

function add_stepsize!(step_sizes::StepSizes)
    (; η, η_sum) = step_sizes
    step_sizes.η_sum = add!!(η_sum, η)
    return nothing
end

function reset_stepsize!(step_sizes::StepSizes)
    (; η_sum) = step_sizes
    step_sizes.η_sum = broadcast!!(zero, η_sum, η_sum)
    return nothing
end

"""
    primal_weight_update!!(scratch, step_sizes, sol_cand, sol_restart, params)

Compute the new primal weight, column by column, into `step_sizes.ω`.
"""
function primal_weight_update!!(
        scratch::Scratch,
        step_sizes::StepSizes,
        sol_cand::PrimalDualSolution,
        sol_restart::PrimalDualSolution,
        params::StepSizeParameters
    )
    (; ω) = step_sizes
    (; primal_weight_damping, zero_tol) = params
    Δx = colnorm!!(scratch.b1, @. scratch.x = sol_cand.x - sol_restart.x)
    Δy = colnorm!!(scratch.b2, @. scratch.y = sol_cand.y - sol_restart.y)
    θ = primal_weight_damping
    return broadcast!!(ω, Δx, Δy, ω) do Δx, Δy, w
        ifelse(
            (Δx > zero_tol) & (Δy > zero_tol),
            exp(θ * log(Δy / Δx) + (1 - θ) * log(w)),
            w
        )
    end
end
