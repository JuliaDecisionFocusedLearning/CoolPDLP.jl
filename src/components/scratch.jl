@kwdef struct Scratch{T <: Number, V <: DenseVector{T}}
    "primal scratch (length `nvar`)"
    x::V
    "dual scratch (length `ncons`)"
    y::V
    "dual scratch (length `nvar`)"
    r::V
end

Scratch(sol::PrimalDualSolution) = Scratch(;
    x = similar(sol.x), y = similar(sol.y), r = similar(sol.x),
)

Scratch(sol::PrimalDualSolution, ::AbstractProgram) = Scratch(sol)
