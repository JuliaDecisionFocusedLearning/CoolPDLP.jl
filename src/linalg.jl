proj_box(x::Number, l::Number, u::Number) = min(u, max(l, x))

function proj_box(
        x::AbstractVector{T},
        l::AbstractVector{T},
        u::AbstractVector{T}
    ) where {T <: Number}
    return map(proj_box, x, l, u)
end


positive_part(a::Number) = max(a, zero(a))
negative_part(a::Number) = -min(a, zero(a))

positive_part(a::AbstractVector{<:Number}) = map(positive_part, a)
negative_part(a::AbstractVector{<:Number}) = map(negative_part, a)


sqnorm(v::AbstractVector{<:Number}) = dot(v, v)


spectral_norm(K::AbstractMatrix{<:Number}) = svdsolve(K)[1][1]
