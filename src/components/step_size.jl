"""
    StepSizeParameters

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct StepSizeParameters{T <: Number}
    "scaling of the inverse spectral norm of `K` when defining the non-adaptive step size"
    invnorm_scaling::T
end

function Base.show(io::IO, params::StepSizeParameters)
    (; invnorm_scaling) = params
    return print(io, "Step size: invnorm_scaling=$invnorm_scaling")
end

function fixed_stepsize(milp::MILP{T}, params::StepSizeParameters) where {T}
    (; A, At) = milp
    (; invnorm_scaling) = params
    η = T(invnorm_scaling) * inv(spectral_norm(A, At))
    return η
end
