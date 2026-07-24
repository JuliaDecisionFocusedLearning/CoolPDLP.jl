@kwdef struct Scratch{T <: Number, V <: AbstractVecOrMat{T}, S <: BatchedNumber, R, C}
    "primal scratch (length `nvar`)"
    x::V
    "dual scratch (length `ncons`)"
    y::V
    "dual scratch (length `nvar`)"
    z::V
    "per-column scratch (one entry per batch column)"
    b1::S
    "per-column scratch (one entry per batch column)"
    b2::S
    "destination of column-wise reductions, a `1 × nbatch` alias of `red_vec`"
    red::R
    "per-column alias of `red`"
    red_vec::S
    "per-column condition scratch"
    cond::C
end

function Scratch(sol::PrimalDualSolution{T}) where {T}
    red_vec = batched_expand(sol.x, zero(T))
    return Scratch(
        similar(sol.x),
        similar(sol.y),
        similar(sol.x),
        batched_expand(sol.x, zero(T)),
        batched_expand(sol.x, zero(T)),
        batched_row(red_vec),
        red_vec,
        batched_expand(sol.x, false),
    )
end

nbinstances((; x)::Scratch) = size(x, 2)
function instance(scratch::Scratch, i::Int)
    red_vec = instance_num(scratch.red_vec, i)
    return Scratch(
        instance_vec(scratch.x, i),
        instance_vec(scratch.y, i),
        instance_vec(scratch.z, i),
        instance_num(scratch.b1, i),
        instance_num(scratch.b2, i),
        batched_row(red_vec),
        red_vec,
        instance_num(scratch.cond, i),
    )
end

"""
    colnorm!(dest, scratch, x)

Compute the Euclidean norm of `x`, or one norm per column if `x` is batched, into `dest`.
"""
colnorm!(::Number, ::Scratch, v::AbstractVector) = norm(v)
function colnorm!(dest::AbstractVector, scratch::Scratch, m::AbstractMatrix)
    sum!(abs2, scratch.red, m)
    dest .= sqrt.(scratch.red_vec)
    return dest
end

"""
    colsum!(dest, scratch, x)

Compute the sum of `x`, or one sum per column if `x` is batched, into `dest`.
"""
colsum!(::Number, ::Scratch, v::AbstractVector) = sum(v)
function colsum!(dest::AbstractVector, scratch::Scratch, m::AbstractMatrix)
    sum!(scratch.red, m)
    dest .= scratch.red_vec
    return dest
end
