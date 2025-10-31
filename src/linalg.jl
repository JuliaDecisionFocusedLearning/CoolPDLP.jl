proj_box(x::Number, l::Number, u::Number) = min(u, max(l, x))

function proj_λ(λ::T, l::T, u::T) where {T <: Number}
    if l == typemin(T) && u == typemax(T)
        return zero(T)  # project on {0}
    elseif l == typemin(T)
        return -negative_part(λ)  # project on ℝ⁻
    elseif u == typemax(T)
        return positive_part(λ)  # project on ℝ⁺
    else
        return λ  # project on ℝ
    end
end

positive_part(a::Number) = max(a, zero(a))
negative_part(a::Number) = -min(a, zero(a))

sqnorm(v::AbstractVector{<:Number}) = dot(v, v)

spectral_norm(K::AbstractMatrix{<:Number}) = svdsolve(K)[1][1]

change_eltype(::Type{T}, A::AbstractArray) where {T} = map(T, A)

function change_eltype(::Type{T}, A::DeviceSparseMatrixCSR) where {T}
    return DeviceSparseMatrixCSR(
        A.m,
        A.n,
        A.rowptr,
        A.colval,
        map(T, A.nzval)
    )
end


# piracy

function Base.show(io::IO, A::DeviceSparseMatrixCSR)
    return show(io, SparseMatrixCSC(A))
end

function Base.show(io::IO, mime::MIME"text/plain", A::DeviceSparseMatrixCSR)
    return show(io, mime, SparseMatrixCSC(A))
end

function Base.similar(A::DeviceSparseMatrixCSR, ::Type{T}, dims::Vararg{Union{Integer, AbstractUnitRange}, N}) where {T, N}
    return similar(A.nzval, T, dims...)
end
