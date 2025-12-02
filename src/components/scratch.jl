@kwdef struct Scratch{T <: Number, V <: DenseVector{T}}
    "primal scratch (length `nvar`)"
    x::V
    "dual scratch (length `ncons`)"
    y::V
    "dual scratch (length `nvar`)"
    r::V
end
