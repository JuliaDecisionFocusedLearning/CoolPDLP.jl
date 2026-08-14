"""
    common_backend(args...)

Return the common GPU backend of several arguments, if it exists, and throw an error otherwise.
"""
function common_backend(args::Vararg{Any, N}) where {N}
    backends = map(get_backend, args)
    if !all(==(backends[1]), backends)
        throw(ArgumentError("There are several different backends among the arguments: $(unique(backends))"))
    end
    return backends[1]
end

"""
    check_mul_dims(c, A, b)

Throw a `DimensionMismatch` unless `c`, `A` and `b` have compatible sizes for `mul!(c, A, b, α, β)`,
i.e. `size(c, 1) == size(A, 1)` and `size(b, 1) == size(A, 2)`. `A` need not be an `AbstractMatrix`,
only support `size`, which covers matrix-free operators like [`Symmetrized`](@ref).
"""
function check_mul_dims(c::AbstractVecOrMat, A, b::AbstractVecOrMat)
    if size(c, 1) != size(A, 1) || size(b, 1) != size(A, 2)
        throw(
            DimensionMismatch(
                "A has size $(size(A)), c has size $(size(c)), b has size $(size(b))"
            )
        )
    end
    return nothing
end
