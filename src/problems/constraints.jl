"""
    ConstraintMatrix

# Fields

$(TYPEDFIELDS)
"""
struct ConstraintMatrix{T <: Number, Ti <: Integer, M <: AbstractSparseMatrix{T, Ti}}
    A::M
    At::M
end
