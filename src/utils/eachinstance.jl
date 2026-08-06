"""
    nbinstances(x)

Return the number of problem instances batched inside `x`.
"""
function nbinstances end

"""
    instance(x, i)

Return the `i`-th instance of the batch held by `x`, sharing its memory whenever possible.
"""
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

"""
    BatchedNumber

Equivalent to `Union{Number, AbstractVector{<:Number}}`. Represents a quantity which is scalar without batching, and holds one value per instance otherwise.

Combine such quantities with `BangBang.broadcast!!(f, dest, args...)`, which writes into `dest` when batched and returns a fresh number otherwise, so the result must always be used.
"""
const BatchedNumber = Union{Number, AbstractVector{<:Number}}

"""
    batched_expand(x, val)

Return `val` itself if the array `x` is not batched, or one copy of `val` per instance otherwise.
"""
batched_expand(::AbstractVector, val::Number) = val
function batched_expand(x::AbstractMatrix, val::Number)
    return fill!(similar(x, typeof(val), size(x, 2)), val)
end
batched_expand(x::AbstractMatrix, val::AbstractVector) = adapt(get_backend(x), val)

"""
    batched_zeros(x, n, nbinstances, Val(batched))

Allocate a zeroed vector of length `n` with the same array type as `x`, or a matrix holding
one such column per instance when `batched` is `true`.

Batching is passed as a `Val` because the number of instances is only known at run time,
while the shape of the result must be inferrable.
"""
function batched_zeros(
        x::AbstractVecOrMat, n::Int, nbinstances::Int, ::Val{batched}
    ) where {batched}
    dims = batched ? (n, nbinstances) : (n,)
    return zero!(similar(x, dims))
end

"""
    batched_similar(val)

Return an uninitialized per-instance quantity with the same shape as `val`.
"""
batched_similar(val::Number) = val
batched_similar(val::AbstractVector) = similar(val)

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
