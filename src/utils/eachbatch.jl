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
batch_expand(::StridedVector, val::Number) = val
function batch_expand(x::StridedMatrix, val::Number)
    return fill!(allocate(get_backend(x), typeof(val), size(x, 2)), val)
end
batch_expand(x::StridedMatrix, val::AbstractVector) = adapt(get_backend(x), val)

"""
    batch_apply!(f, dest, a, b)

Apply `f` to per-column quantities, storing the result inside `dest` when batched.
"""
batch_apply!(f::F, ::Number, a::Number, b::Number) where {F} = f(a, b)
function batch_apply!(f::F, dest::AbstractVector, a::BatchedNumber, b::BatchedNumber) where {F}
    dest .= f.(a, b)
    return dest
end

"""
    batch_num(val, i)

Extract the value of `val` for the `i`-th column of the batch.
"""
batch_num(val::Number, ::Int) = val
batch_num(val::AbstractVector, i::Int) = val[i]

"""
    batch_all(cond)

Reduce a per-column condition to a single decision for the whole batch.
"""
batch_all(cond::Bool) = cond
batch_all(cond::AbstractVector{Bool}) = all(cond)
