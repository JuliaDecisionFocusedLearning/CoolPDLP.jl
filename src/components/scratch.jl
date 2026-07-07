@kwdef struct Scratch{T <: Number, V <: DenseVecOrMat{T}}
    "primal scratch (length `nvar`)"
    x::V
    "dual scratch (length `ncons`)"
    y::V
    "dual scratch (length `nvar`)"
    z::V
end

Scratch(sol::PrimalDualSolution) = Scratch(similar(sol.x), similar(sol.y), similar(sol.x))

batch_size((; x)::Scratch) = size(x, 2)
function batch(scratch::Scratch, i::Int)
    return Scratch(
        batch_vec(scratch.x, i),
        batch_vec(scratch.y, i),
        batch_vec(scratch.z, i),
    )
end
