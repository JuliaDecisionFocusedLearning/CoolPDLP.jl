function batch_size end
function batch end

struct EachBatch{ElTy, T} <: AbstractVector{ElTy}
    data::T
    batch_size::Int
    function EachBatch(data::T) where {T}
        ElTy = Core.Compiler.return_type(batch, Tuple{T, Int})
        return new{ElTy, T}(data, batch_size(data))
    end
end

Base.size((; batch_size)::EachBatch) = (batch_size,)
function Base.getindex(eb::EachBatch, i::Int)
    i in eachindex(eb) || throw(BoundsError(eb, i))
    return batch(eb.data, i)
end

batch_vec(v::AbstractVector, ::Int) = v
batch_vec(m::AbstractMatrix, i::Int) = view(m, :, i)

batch_mat(m::AbstractMatrix, ::Int) = m
batch_mat(a::AbstractArray{T, 3}, i::Int) where {T} = view(a, :, :, i)

"""
    BatchedNumber

Type of a quantity which is scalar without batching, and holds one value per column of the batch otherwise.
"""
const BatchedNumber = Union{Number, AbstractVector{<:Number}}

"""
    batch_expand(x, val)

Return `val` itself if the array `x` is not batched, or one copy of `val` per column of `x` otherwise.
"""
batch_expand(::AbstractVector, val::Number) = val
function batch_expand(x::AbstractMatrix, val::Number)
    return fill!(allocate(get_backend(x), typeof(val), size(x, 2)), val)
end
batch_expand(x::AbstractMatrix, val::AbstractVector) = adapt(get_backend(x), val)

"""
    batch_row(val)

Return a `1 × nbatch` alias of the per-column quantity `val`, suitable as a reduction destination.
"""
batch_row(val::Number) = val
batch_row(val::AbstractVector) = reshape(val, 1, length(val))

"""
    batch_similar(val)

Return an uninitialized per-column quantity with the same shape as `val`.
"""
batch_similar(val::Number) = val
batch_similar(val::AbstractVector) = similar(val)

"""
    batch_apply!(f, dest, args...)

Apply `f` to per-column quantities, storing the result inside `dest` when batched.

The result is returned rather than only written, because `dest` is a plain number without batching.
"""
batch_apply!(f::F, ::Number, a::Number, b::Number) where {F} = f(a, b)
function batch_apply!(f::F, dest::AbstractVector, a::BatchedNumber, b::BatchedNumber) where {F}
    dest .= f.(a, b)
    return dest
end
batch_apply!(f::F, ::Number, a::Number, b::Number, c::Number) where {F} = f(a, b, c)
function batch_apply!(
        f::F, dest::AbstractVector, a::BatchedNumber, b::BatchedNumber, c::BatchedNumber
    ) where {F}
    dest .= f.(a, b, c)
    return dest
end

"""
    batch_num(val, i)

Extract the value of `val` for the `i`-th column of the batch.
"""
batch_num(val::Number, ::Int) = val
batch_num(val::AbstractVector, i::Int) = val[i]

"""
    batch_all(f, args...)

Reduce the per-column conditions `f(args...)` to a single decision for the whole batch.
"""
batch_all(f::F, a::Number) where {F} = f(a)
batch_all(f::F, a::AbstractVector) where {F} = all(f, a)

"""
    batch_mean(val)

Average a per-column quantity over the whole batch, yielding a single number.
"""
batch_mean(val::Number) = val
batch_mean(val::AbstractVector) = sum(val) / length(val)
