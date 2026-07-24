function nbinstances end
function instance end

struct EachInstance{ElTy, T} <: AbstractVector{ElTy}
    data::T
    nbinstances::Int
    function EachInstance(data::T) where {T}
        ElTy = Core.Compiler.return_type(instance, Tuple{T, Int})
        return new{ElTy, T}(data, nbinstances(data))
    end
end

Base.size((; nbinstances)::EachInstance) = (nbinstances,)
function Base.getindex(ei::EachInstance, i::Int)
    i in eachindex(ei) || throw(BoundsError(ei, i))
    return instance(ei.data, i)
end

instance_vec(v::AbstractVector, ::Int) = v
instance_vec(m::AbstractMatrix, i::Int) = view(m, :, i)

instance_mat(m::AbstractMatrix, ::Int) = m
instance_mat(a::AbstractArray{T, 3}, i::Int) where {T} = view(a, :, :, i)

"""
    BatchedNumber

Type of a quantity which is scalar without batching, and holds one value per instance otherwise.
"""
const BatchedNumber = Union{Number, AbstractVector{<:Number}}

"""
    batched_expand(x, val)

Return `val` itself if the array `x` is not batched, or one copy of `val` per instance otherwise.
"""
batched_expand(::AbstractVector, val::Number) = val
function batched_expand(x::AbstractMatrix, val::Number)
    return fill!(allocate(get_backend(x), typeof(val), size(x, 2)), val)
end
batched_expand(x::AbstractMatrix, val::AbstractVector) = adapt(get_backend(x), val)

"""
    batched_row(val)

Return a `1 × nbinstances` alias of the per-instance quantity `val`, suitable as a reduction destination.
"""
batched_row(val::Number) = val
batched_row(val::AbstractVector) = reshape(val, 1, length(val))

"""
    batched_similar(val)

Return an uninitialized per-instance quantity with the same shape as `val`.
"""
batched_similar(val::Number) = val
batched_similar(val::AbstractVector) = similar(val)

"""
    batched_apply!(f, dest, args...)

Apply `f` to per-instance quantities, storing the result inside `dest` when batched.

The result is returned rather than only written, because `dest` is a plain number without batching.
"""
batched_apply!(f::F, ::Number, a::Number, b::Number) where {F} = f(a, b)
function batched_apply!(f::F, dest::AbstractVector, a::BatchedNumber, b::BatchedNumber) where {F}
    dest .= f.(a, b)
    return dest
end
batched_apply!(f::F, ::Number, a::Number, b::Number, c::Number) where {F} = f(a, b, c)
function batched_apply!(
        f::F, dest::AbstractVector, a::BatchedNumber, b::BatchedNumber, c::BatchedNumber
    ) where {F}
    dest .= f.(a, b, c)
    return dest
end

"""
    instance_num(val, i)

Extract the value of `val` for the `i`-th instance of the batch.
"""
instance_num(val::Number, ::Int) = val
instance_num(val::AbstractVector, i::Int) = val[i]

"""
    batched_all(f, args...)

Reduce the per-instance conditions `f(args...)` to a single decision for the whole batch.
"""
batched_all(f::F, a::Number) where {F} = f(a)
batched_all(f::F, a::AbstractVector) where {F} = all(f, a)

"""
    batched_mean(val)

Average a per-instance quantity over all the instances, yielding a single number.
"""
batched_mean(val::Number) = val
batched_mean(val::AbstractVector) = sum(val) / length(val)
