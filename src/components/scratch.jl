@kwdef struct Scratch{T <: Number, V <: StridedVecOrMat{T}, S <: BatchedNumber}
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
        similar(sol.x), similar(sol.y), similar(sol.x),
        batch_expand(sol.x, zero(T)), batch_expand(sol.x, zero(T)),
    )
end

batch_size((; x)::Scratch) = size(x, 2)
function batch(scratch::Scratch, i::Int)
    return Scratch(
        batch_vec(scratch.x, i),
        batch_vec(scratch.y, i),
        batch_vec(scratch.z, i),
        batch_num(scratch.b1, i),
        batch_num(scratch.b2, i),
    )
end
