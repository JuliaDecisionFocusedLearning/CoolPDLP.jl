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


function proj_X!(
        x::AbstractVector,
        l::AbstractVector,
        u::AbstractVector
    )
    return x .= proj_box.(x, l, u)
end

function proj_Y!(
        y::AbstractVector,
        ineq_cons::AbstractVector{Bool}
    )
    return y .= ifelse.(ineq_cons, positive_part.(y), y)
end

function proj_Λ!(
        λ::AbstractVector,
        l::AbstractVector,
        u::AbstractVector
    )
    return λ .= proj_λ.(λ, l, u)
end
