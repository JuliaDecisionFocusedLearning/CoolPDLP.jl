proj_box(x::Number, l::Number, u::Number) = min(u, max(l, x))

function proj_λ(λ::T, l::T, u::T) where {T <: Number}
    if l == typemin(T) && u == typemax(T)
        return zero(T)  # project on {0}
    elseif l == typemin(T)
        return -negative_part(λ)  # project on ℝ⁻
    elseif u == typemax(T)
        return positive_part(λ)  # project on ℝ⁺
    else
        return λ  # project on ℝ
    end
end

positive_part(a::Number) = max(a, zero(a))
negative_part(a::Number) = -min(a, zero(a))

sqnorm(v::AbstractVector{<:Number}) = dot(v, v)

struct Symmetrized{T <: Number, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    K::M
    Kᵀ::M
    scratch::V
end

function Symmetrized(K::AbstractMatrix, Kᵀ::AbstractMatrix)
    scratch = allocate(get_backend(K), eltype(K), size(K, 1))
    return Symmetrized(K, Kᵀ, scratch)
end

Base.eltype(sym::Symmetrized) = eltype(sym.K)
Base.size(sym::Symmetrized, ::Int) = size(sym.K, 2)

function LinearAlgebra.mul!(y, sym::Symmetrized, x)
    (; K, Kᵀ, scratch) = sym
    mul!(scratch, K, x)
    mul!(y, Kᵀ, scratch)
    return y
end

function spectral_norm(
        K::AbstractMatrix{<:Number},
        Kᵀ::AbstractMatrix{<:Number};
        kwargs...
    )
    x0 = allocate(get_backend(K), eltype(K), size(K, 2))
    randn!(x0)
    KᵀK = Symmetrized(K, Kᵀ)
    λ, _ = powm!(KᵀK, x0; kwargs...)
    return sqrt(λ)
end
