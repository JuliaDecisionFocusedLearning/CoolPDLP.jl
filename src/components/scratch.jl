@kwdef struct Scratch{T <: Number, V <: DenseVector{T}}
    "primal scratch (length `nvar`)"
    x::V
    "dual scratch (length `ncons`)"
    y::V
    "dual scratch (length `nvar`)"
    z::V
end

Scratch(sol::PrimalDualSolution) = Scratch(similar(sol.x), similar(sol.y), similar(sol.x))
