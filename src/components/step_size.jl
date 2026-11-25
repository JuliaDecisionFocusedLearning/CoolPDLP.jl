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
    "absolute tolerance in iterative spectral norm computation"
    spectral_norm_tol::T
    "maximum number of power method iterations in iterative spectral norm computation"
    spectral_norm_maxiter::I
end

function Base.show(io::IO, params::StepSizeParameters)
    (; invnorm_scaling, primal_weight_damping, zero_tol, spectral_norm_tol, spectral_norm_maxiter) = params
    return print(io, "StepSizeParameters: invnorm_scaling=$invnorm_scaling, primal_weight_damping=$primal_weight_damping, zero_tol=$zero_tol, spectral_norm_tol=$spectral_norm_tol, spectral_norm_maxiter=$spectral_norm_maxiter")
end

function fixed_stepsize(milp::MILP{T}, params::StepSizeParameters) where {T}
    (; A, At) = milp
    (; invnorm_scaling, spectral_norm_tol, spectral_norm_maxiter) = params
    norm = spectral_norm(A, At; tol = spectral_norm_tol, maxiter = spectral_norm_maxiter)
    # a zero spectral norm (no constraint rows, or an all-zero `A`) means `A'y` never
    # contributes to the primal step regardless of `η`, so any finite step size is safe;
    # `inv(norm)` would otherwise be `Inf` and corrupt the first primal step
    η = iszero(norm) ? one(T) : T(invnorm_scaling) * inv(norm)
    return η
end

"""
    quadratic_norm(Q, params)

Return the spectral norm `‖Q‖₂` of the objective matrix, `0` for a [`ZeroMatrix`](@ref).
"""
quadratic_norm(::ZeroMatrix{T}, ::StepSizeParameters) where {T} = zero(T)
# the largest instance norm, so that the shared step sizes are safe for every instance
function quadratic_norm(S::BatchScaledMatrix, params::StepSizeParameters)
    return convert(eltype(S), maximum(S.scale)) * quadratic_norm(S.Q, params)
end
function quadratic_norm(Q::AbstractMatrix, params::StepSizeParameters)
    isquadratic(Q) || return zero(eltype(Q))
    return spectral_norm(Q, Q; tol = params.spectral_norm_tol, maxiter = params.spectral_norm_maxiter)
end

"""
    quadratic_stepsize(norm_A, norm_Q, ω, scaling)

Return the largest step size `η` such that the primal and dual steps `τ = η / ω`, `σ = η ω` of
the PDHG linearized in `Q` satisfy `1/τ - ‖Q‖ ≥ σ ‖A‖²`, scaled by `scaling`:

    η = 2 / (‖Q‖/ω + √(‖Q‖²/ω² + 4‖A‖²))

This is the step size of PDQP (Lu & Yang, 2023) at the start of a restart cycle, which reduces to
`1 / ‖A‖` for a linear program. The condition makes the metric `[I/τ - Q  -Aᵀ; -A  I/σ]` of the
proximal point form of PDHG positive semidefinite (the sign of the off-diagonal blocks, `+` with
the dual sign convention of this package, does not matter for the condition).
"""
function quadratic_stepsize(norm_A, norm_Q, ω, scaling)
    q = norm_Q / ω
    denominator = q + sqrt(q^2 + 4 * norm_A^2)
    return ifelse(iszero(denominator), one(ω), scaling * 2 / denominator)
end

"""
    initial_stepsize(milp, ω, params)

Return the step size `η` of `milp` for the primal weight `ω`, one per instance if `ω` is
batched, and the spectral norms `(‖A‖₂, ‖Q‖₂)` it was computed from.
"""
function initial_stepsize(milp::MILP{T}, ω::BatchedNumber, params::StepSizeParameters) where {T}
    (; A, At, Q) = milp
    norm_A = spectral_norm(A, At; tol = params.spectral_norm_tol, maxiter = params.spectral_norm_maxiter)
    norm_Q = quadratic_norm(Q, params)
    return initial_stepsize(Q, ω, norm_A, norm_Q, params), norm_A, norm_Q
end

function initial_stepsize(
        ::ZeroMatrix, ω::BatchedNumber, norm_A::T, norm_Q, params::StepSizeParameters
    ) where {T}
    # same as `fixed_stepsize`, from the norm already computed
    η = iszero(norm_A) ? one(T) : T(params.invnorm_scaling) * inv(norm_A)
    return broadcast!!((_, η) -> η, batched_similar(ω), ω, η)
end

function initial_stepsize(
        Q::AbstractMatrix, ω::BatchedNumber, norm_A, norm_Q, params::StepSizeParameters
    )
    # an empty sparse `Q` gets exactly the step size of a linear program
    isquadratic(Q) || return initial_stepsize(ZeroMatrix{eltype(Q)}(size(Q, 1)), ω, norm_A, norm_Q, params)
    scaling = params.invnorm_scaling
    return broadcast!!(batched_similar(ω), ω) do ω
        quadratic_stepsize(norm_A, norm_Q, ω, scaling)
    end
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
@kwdef mutable struct StepSizes{T <: BatchedNumber, N <: Number}
    "step size"
    η::T
    "cumulated step size since last restart"
    η_sum::T = zero(η)
    "primal weight"
    ω::T
    "spectral norm of the constraint matrix (only read by the step size of a quadratic program)"
    norm_A::N = zero(eltype(η))
    "spectral norm of the quadratic objective matrix (zero for a linear program)"
    norm_Q::N = zero(eltype(η))
end

instance(step_sizes::StepSizes, i::Int) = StepSizes(
    instance_num(step_sizes.η, i),
    instance_num(step_sizes.η_sum, i),
    instance_num(step_sizes.ω, i),
    step_sizes.norm_A,
    step_sizes.norm_Q,
)

function add_stepsize!(step_sizes::StepSizes, ::Bool)
    (; η, η_sum) = step_sizes
    step_sizes.η_sum = add!!(η_sum, η)
    return nothing
end
function add_stepsize!(step_sizes::StepSizes, active::AbstractVector)
    (; η, η_sum) = step_sizes
    step_sizes.η_sum = masked_apply!!(+, η_sum, active, η_sum, η)
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

"""
    refresh_stepsize!(step_sizes, Q, params)

Recompute the step size of a quadratic program after its primal weight changed, since the
condition of [`quadratic_stepsize`](@ref) depends on it. A no-op for a linear program.
"""
refresh_stepsize!(::StepSizes, ::ZeroMatrix, ::StepSizeParameters) = nothing
function refresh_stepsize!(step_sizes::StepSizes, Q::AbstractMatrix, params::StepSizeParameters)
    isquadratic(Q) || return nothing
    (; norm_A, norm_Q) = step_sizes
    scaling = params.invnorm_scaling
    step_sizes.η = broadcast!!(step_sizes.η, step_sizes.ω) do ω
        quadratic_stepsize(norm_A, norm_Q, ω, scaling)
    end
    return nothing
end
