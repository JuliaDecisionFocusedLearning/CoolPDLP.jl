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
