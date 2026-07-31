"""
    BatchedDiagonal

Diagonal matrix of a batch, holding the diagonal of each instance in a column.

# Fields

$(TYPEDFIELDS)
"""
struct BatchedDiagonal{T <: Number, M <: AbstractMatrix{T}}
    "one diagonal per instance"
    diag::M
end

Adapt.@adapt_structure BatchedDiagonal

"""
    DiagonalScaling

Type of the preconditioner scalings attached to a [`MILP`](@ref), shared by the whole batch or not.
"""
const DiagonalScaling{T} = Union{Diagonal{T}, BatchedDiagonal{T}}

Base.eltype(::Type{<:BatchedDiagonal{T}}) where {T} = T
Base.size(D::BatchedDiagonal) = (size(D.diag, 1), size(D.diag, 1), size(D.diag, 2))
Base.size(D::BatchedDiagonal, i::Integer) = i <= 3 ? size(D)[i] : 1
LinearAlgebra.diag(D::BatchedDiagonal) = D.diag
KernelAbstractions.get_backend(D::BatchedDiagonal) = get_backend(D.diag)

nbinstances(D::BatchedDiagonal) = size(D.diag, 2)
instance(D::Diagonal, ::Int) = D
instance(D::BatchedDiagonal, i::Int) = Diagonal(view(D.diag, :, i))

function Base.isapprox(D1::BatchedDiagonal, D2::BatchedDiagonal; kwargs...)
    return isapprox(D1.diag, D2.diag; kwargs...)
end

Base.inv(D::BatchedDiagonal) = BatchedDiagonal(inv.(D.diag))
Base.:*(D1::BatchedDiagonal, D2::BatchedDiagonal) = BatchedDiagonal(D1.diag .* D2.diag)
Base.:*(D1::BatchedDiagonal, D2::Diagonal) = BatchedDiagonal(D1.diag .* diag(D2))
Base.:*(D1::Diagonal, D2::BatchedDiagonal) = BatchedDiagonal(diag(D1) .* D2.diag)
Base.:*(D::BatchedDiagonal, x::AbstractVecOrMat) = D.diag .* x
Base.:\(D::BatchedDiagonal, x::AbstractVecOrMat) = x ./ D.diag
