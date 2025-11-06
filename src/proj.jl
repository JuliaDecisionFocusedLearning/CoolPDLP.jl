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
