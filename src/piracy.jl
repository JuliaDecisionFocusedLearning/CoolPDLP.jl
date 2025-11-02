function Base.similar(A::DeviceSparseMatrixCSR, ::Type{T}, dims::Vararg{Union{Integer, AbstractUnitRange}, N}) where {T, N}
    return similar(A.nzval, T, dims...)
end
