"""
    PrimalDualSolution

# Fields

$(TYPEDFIELDS)
"""
@kwdef mutable struct PrimalDualSolution{T <: Number, V <: AbstractVector{T}}
    const x::V
    const y::V
    const Kx::V
    const Kᵀy::V
    const λ::V
end

Base.eltype(::PrimalDualSolution{T}) where {T} = T

function PrimalDualSolution(
        sad::SaddlePointProblem{T, V},
        x::V,
        y::V,
    ) where {T, V}
    (; c, K, Kᵀ, l, u) = sad
    Kx = K * x
    Kᵀy = Kᵀ * y
    λ = proj_λ.(c - Kᵀy, l, u)
    z = PrimalDualSolution(; x, y, Kx, Kᵀy, λ)
    return z
end

function Base.copy(z::PrimalDualSolution)
    return PrimalDualSolution(
        copy(z.x),
        copy(z.y),
        copy(z.Kx),
        copy(z.Kᵀy),
        copy(z.λ),
    )
end

function Base.zero(z::PrimalDualSolution{T}) where {T}
    return PrimalDualSolution(
        zero(z.x),
        zero(z.y),
        zero(z.Kx),
        zero(z.Kᵀy),
        zero(z.λ),
    )
end

function zero!(z::PrimalDualSolution{T}) where {T}
    zero!(z.x)
    zero!(z.y)
    zero!(z.Kx)
    zero!(z.Kᵀy)
    zero!(z.λ)
    return nothing
end

function Base.copy!(z1::PrimalDualSolution, z2::PrimalDualSolution)
    copy!(z1.x, z2.x)
    copy!(z1.y, z2.y)
    copy!(z1.Kx, z2.Kx)
    copy!(z1.Kᵀy, z2.Kᵀy)
    copy!(z1.λ, z2.λ)
    return z1
end

function LinearAlgebra.axpby!(
        a::T, x::PrimalDualSolution{T, V}, b::T, y::PrimalDualSolution{T, V},
    ) where {T, V}
    axpby!(a, x.x, b, y.x)
    axpby!(a, x.y, b, y.y)
    axpby!(a, x.Kx, b, y.Kx)
    axpby!(a, x.Kᵀy, b, y.Kᵀy)
    axpby!(a, x.λ, b, y.λ)
    return y
end
