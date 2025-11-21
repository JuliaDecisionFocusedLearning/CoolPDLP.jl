zero!(x::AbstractArray) = fill!(x, zero(eltype(x)))

@inline positive_part(a::Number) = max(a, zero(a))
@inline negative_part(a::Number) = -min(a, zero(a))

@inline proj_box(x::Number, l::Number, u::Number) = min(u, max(l, x))

function proj_λ(λ::T, l::T, u::T) where {T <: Number}
    lmin = l == typemin(T)
    umax = u == typemax(T)
    return ifelse(
        lmin, ifelse(umax, zero(T), -negative_part(λ)), ifelse(umax, positive_part(λ), λ)
    )
end

sqnorm(v::AbstractVector{<:Number}) = dot(v, v)

custom_sqnorm(x, y, ω) = sqrt(ω * sqnorm(x) + inv(ω) * sqnorm(y))

function safeprod_rightpos(left, right)
    return ifelse(isinf(left), positive_part(right), left * positive_part(right))
end
function safeprod_rightneg(left, right)
    return ifelse(isinf(left), negative_part(right), left * negative_part(right))
end

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

function spectral_norm(K::AbstractMatrix{<:Number}, Kᵀ::AbstractMatrix{<:Number}; kwargs...)
    x0 = allocate(get_backend(K), eltype(K), size(K, 2))
    randn!(StableRNG(0), x0)
    KᵀK = Symmetrized(K, Kᵀ)
    λ, _ = powm!(KᵀK, x0; kwargs...)
    return sqrt(λ)
end

column_norm(A::SparseMatrixCSC, j::Integer, p) = norm(view(nonzeros(A), nzrange(A, j)), p)
