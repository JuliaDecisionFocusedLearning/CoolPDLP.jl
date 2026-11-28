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

"""
    batched_apply!!(f, dest, args...)

Compute `f(args...)`, column by column into `dest`: directly when every argument is a number,
with `broadcast!!` otherwise.

Under Reactant, a zero-dimensional broadcast between traced numbers can fail to compile
("Missing Julia object for global"), depending on what was compiled before it, so unbatched
quantities never go through `broadcast!!`.
"""
# `Vararg{Any, N}` rather than `Vararg{BatchedNumber}`: Julia does not specialize on a vararg of
# an abstract union, and then allocates at every call
batched_apply!!(f::F, ::Number, args::Vararg{Number, N}) where {F, N} = f(args...)
batched_apply!!(f::F, dest::AbstractVector, args::Vararg{Any, N}) where {F, N} = broadcast!!(f, dest, args...)

#=
Freezing terminated instances

A batch keeps one `Bool` per instance telling whether it is still running (see
`ConvergenceStats`). Every update of the iterates goes through the helpers below, which leave the
columns of the terminated instances untouched. Without batching, the mask is the constant
`true` and the helpers dispatch on its type (`Bool`, as opposed to a vector) to the plain,
unmasked update: an unbatched solve runs exactly the code it ran before. Under Reactant, the
masks are traced arrays and the selections are `ifelse`s, so they stay part of the compiled
program.
=#

"""
    KeepInactive(f)

Elementwise `(a, d, xs...) -> ifelse(a, f(xs...), d)`, as a callable struct so that it
specializes on the number of arguments (a closure with a splatted argument does not).
"""
struct KeepInactive{F}
    f::F
end
@inline (k::KeepInactive)(a, d, xs::Vararg{Any, N}) where {N} = ifelse(a, k.f(xs...), d)

"""
    masked_copyto!(dest, active, src)

Copy the columns of `src` into `dest`, for the active instances only.
"""
masked_copyto!(dest::AbstractVecOrMat, ::Bool, src::AbstractVecOrMat) = copyto!(dest, src)
function masked_copyto!(dest::AbstractMatrix, active::AbstractVector, src::AbstractMatrix)
    broadcast!(ifelse, dest, transpose(active), src, dest)
    return dest
end

"""
    masked_broadcast!(f, dest, active, args...)

Compute `dest .= f.(args...)`, for the columns of the active instances only.
"""
function masked_broadcast!(f::F, dest::AbstractVecOrMat, ::Bool, args::Vararg{Any, N}) where {F, N}
    return broadcast!(f, dest, args...)
end
function masked_broadcast!(
        f::F, dest::AbstractMatrix, active::AbstractVector, args::Vararg{Any, N}
    ) where {F, N}
    return broadcast!(KeepInactive(f), dest, transpose(active), dest, args...)
end

"""
    masked_apply!!(f, dest, active, args...)

Compute the per-instance quantity `f(args...)` into `dest` like `broadcast!!`, keeping the
previous value of `dest` for the instances which are not active.
"""
function masked_apply!!(f::F, dest::BatchedNumber, ::Bool, args::Vararg{Any, N}) where {F, N}
    return batched_apply!!(f, dest, args...)
end
function masked_apply!!(
        f::F, dest::AbstractVector, active::AbstractVector, args::Vararg{Any, N}
    ) where {F, N}
    return broadcast!!(KeepInactive(f), dest, active, dest, args...)
end

"""
    mask_and!!(cond, active)

Restrict the per-instance condition `cond` to the active instances.
"""
mask_and!!(cond::BatchedNumber, ::Bool) = cond
mask_and!!(cond::AbstractVector, active::AbstractVector) = broadcast!!(&, cond, cond, active)
