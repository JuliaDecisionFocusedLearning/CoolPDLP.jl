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
    batched_bool_type(v)

Return the type of the boolean obtained by reducing the per-instance vector `v` to a single decision.

For an ordinary array this is simply `Bool`, but it is a hook for array types whose scalars
are wrapped, and the `Reactant` extension overloads it. Reducing a batch (with `all`, `sum`,
`maximum`, ...) yields one scalar, yet Reactant types every such reduction over a
`TracedRArray` as `Union{TracedRArray, TracedRNumber}`. Left alone, that abstract type escapes
through the reductions below into the return types of the termination and restart checks of a
batched solve, which are what the solve loops branch on. Asserting the scalar type costs
nothing at run time and keeps every caller inferrable.
"""
batched_bool_type(::AbstractVector) = Bool

"""
    batched_all(f, args...)

Reduce the per-instance conditions `f(args...)` to a single decision for the whole batch.
"""
batched_all(f::F, a::Number) where {F} = f(a)
function batched_all(f::F, a::AbstractVector) where {F}
    return all(f, a)::batched_bool_type(a)
end

"""
    batched_mean(val)

Average a per-instance quantity over all the instances, yielding a single number.

The average of a batch is one of its elements, and asserting that keeps the restart check
inferrable for the same reason as in [`batched_bool_type`](@ref). This assumes a
floating-point `eltype`, which every per-instance quantity here has.
"""
batched_mean(val::Number) = val
function batched_mean(val::AbstractVector)
    return (sum(val) / length(val))::eltype(val)
end
