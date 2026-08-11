@kwdef struct Scratch{T <: Number, V <: AbstractVecOrMat{T}, S <: BatchedNumber{T}}
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
end

function Scratch(sol::PrimalDualSolution{T}) where {T}
    return Scratch(
        similar(sol.x),
        similar(sol.y),
        similar(sol.x),
        batched_expand(sol.x, zero(T)),
        batched_expand(sol.x, zero(T)),
    )
end

nbinstances((; x)::Scratch) = size(x, 2)
function instance(scratch::Scratch, i::Int)
    return Scratch(
        instance_vec(scratch.x, i),
        instance_vec(scratch.y, i),
        instance_vec(scratch.z, i),
        instance_num(scratch.b1, i),
        instance_num(scratch.b2, i),
    )
end
