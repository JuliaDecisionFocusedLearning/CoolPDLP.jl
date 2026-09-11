module CoolPDLPReactantExt

using Adapt: Adapt, adapt
using CoolPDLP: CoolPDLP
using KernelAbstractions: KernelAbstractions, get_backend
using LinearAlgebra: LinearAlgebra
using Reactant: Reactant, TracedRArray, TracedRNumber, @reactant_overlay

"""
    CoolPDLP.batched_bool_type(v)

Give the type of the boolean that reducing a traced batch yields.

Reactant types every reduction over a `TracedRArray` as `Union{TracedRArray, TracedRNumber}`,
accurate enough to trace but too coarse for the callers: the abstract type would escape into
the return type of the termination and restart checks of a batched solve. The reduced value is
in fact a traced scalar, which is what this returns.
Once https://github.com/EnzymeAD/Reactant.jl/issues/3261 is solved upstream, this can be removed.
"""
CoolPDLP.batched_bool_type(::TracedRArray) = TracedRNumber{Bool}

# A sparse matrix's shape and pattern are structure, so trace it without number tracking.
function Reactant.traced_type_inner(
        @nospecialize(T::Type{<:CoolPDLP.GPUSparseMatrix}),
        seen,
        mode::Reactant.TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(ndevices),
        @nospecialize(runtime),
    )
    return @invoke Reactant.traced_type_inner(
        T::Type, seen::Any, mode::Reactant.TraceMode, Union{}::Type, ndevices::Any, runtime::Any
    )
end

function Reactant.make_tracer(
        seen,
        @nospecialize(prev::CoolPDLP.GPUSparseMatrix),
        @nospecialize(path),
        mode;
        @nospecialize(track_numbers::Type = Union{}),
        kwargs...,
    )
    return Reactant.make_tracer_unknown(
        seen, prev, path, mode; track_numbers = Union{}, kwargs...
    )
end

# Reactant's own `mul!` overlays densify the matrix, so these send the product to its kernels.
@reactant_overlay function LinearAlgebra.mul!(
        c::AbstractVector, A::CoolPDLP.GPUSparseMatrix, b::AbstractVector, α::Number, β::Number
    )
    return flat_mul!(c, A, b, α, β)
end

@reactant_overlay function LinearAlgebra.mul!(
        c::AbstractMatrix, A::CoolPDLP.GPUSparseMatrix, b::AbstractMatrix, α::Number, β::Number
    )
    return flat_mul!(c, A, b, α, β)
end

@reactant_overlay function LinearAlgebra.mul!(
        c::AbstractVector, A::CoolPDLP.GPUSparseMatrix, b::AbstractVector
    )
    return flat_mul!(c, A, b, true, false)
end

@reactant_overlay function LinearAlgebra.mul!(
        c::AbstractMatrix, A::CoolPDLP.GPUSparseMatrix, b::AbstractMatrix
    )
    return flat_mul!(c, A, b, true, false)
end

"""
    FlatArray

Column-major array view of a flat traced buffer.

A kernel indexes it like the original array, but only ever receives 1-D operands, which XLA
cannot reorder (a kernel call pins no memory layout, so a 2-D operand can come back transposed).
"""
struct FlatArray{T, N, V <: AbstractVector{T}} <: DenseArray{T, N}
    data::V
    dims::NTuple{N, Int}
end

Base.size(x::FlatArray) = x.dims
Base.IndexStyle(::Type{<:FlatArray}) = Base.IndexLinear()
Base.@propagate_inbounds Base.getindex(x::FlatArray, i::Int) = x.data[i]
Base.@propagate_inbounds Base.setindex!(x::FlatArray, v, i::Int) = (x.data[i] = v)
Base.pointer(x::FlatArray) = pointer(x.data)  # for `Atomix.@atomic`
Base.pointer(x::FlatArray, i::Integer) = pointer(x.data, i)

Adapt.adapt_structure(to, x::FlatArray) = FlatArray(adapt(to, x.data), x.dims)

"""
    Flatten

Adaptor turning every traced array into a [`FlatArray`](@ref).
"""
struct Flatten end

Adapt.adapt_structure(::Flatten, x::TracedRArray) = FlatArray(Reactant.Ops.reshape(x, length(x)), size(x))

"""
    FlatBackend

Backend of a [`FlatArray`](@ref), whose kernels are launched by Reactant even from native code.
"""
struct FlatBackend{B <: KernelAbstractions.GPU} <: KernelAbstractions.GPU
    backend::B
end

KernelAbstractions.get_backend(x::FlatArray) = FlatBackend(get_backend(x.data))

function (kernel::KernelAbstractions.Kernel{FlatBackend{B}, W, N, F})(
        args...; ndrange = nothing, workgroupsize = nothing
    ) where {B, W, N, F}
    (; backend) = kernel.backend
    inner = KernelAbstractions.Kernel{B, W, N, F}(backend, kernel.f)
    return Reactant.call_with_reactant(
        Reactant.ka_with_reactant, ndrange, workgroupsize, inner, args...
    )
end

"""
    flat_mul!(c, A, b, α, β)

Run the ordinary `mul!` of `A` on flattened operands, inside a compiled program.
"""
function flat_mul!(c::AbstractVecOrMat, A, b::AbstractVecOrMat, α::Number, β::Number)
    αAb = fill!(similar(c, length(c)), zero(eltype(c)))
    # native dispatch reaches the format's own `mul!`, which adds into the zeroed buffer
    Reactant.call_with_native(
        LinearAlgebra.mul!, FlatArray(αAb, size(c)), adapt(Flatten(), A), adapt(Flatten(), b), α, true
    )
    αAb = Reactant.Ops.reshape(αAb, size(c)...)
    if !(β isa TracedRNumber) && iszero(β)
        c .= αAb
    else
        c .= αAb .+ β .* c
    end
    return c
end

"""
    write_time!(out)

Write the current host time into the single-element output buffer of a Reactant callback.

`Reactant.Ops.julia_callback` hands the callback its output buffers first and its inputs
afterwards, and a `()`-shaped output arrives dereferenced, as a plain `Float64` with nothing to
write into. The output is therefore declared with shape `(1,)` and reduced back to a scalar on
the traced side.

Uses `fill!` because CUDA.jl refuses `out[1] = ...` on a device buffer.
"""
write_time!(out::AbstractVector{Float64}) = (fill!(out, time()); nothing)

"""
    host_callbacks_supported()

Whether `Reactant.Ops.julia_callback` can be serviced on the backend currently in use.

`Reactant.Ops._wrap_buffers` hands the callback its buffers directly on the host, goes through
`CUDA.jl` on the CUDA backend, and raises on every other one. A callback that raises is caught
inside Reactant's trampoline, which logs it and reports failure *on every call* rather than
stopping the run, so an unusable callback surfaces as a hang rather than as an error. Better
not to emit one at all.
"""
function host_callbacks_supported()
    platform = lowercase(Reactant.XLA.platform_name(Reactant.XLA.default_backend()))
    platform == "cpu" && return true
    platform == "cuda" && return Reactant.is_extension_loaded(Val(:CUDA))
    return false
end

"""
    frozen_clock()

Read the clock the old way, for backends that cannot run a host callback.

The value is read once while tracing and baked into the compiled program as a constant, so the
elapsed time never advances and the time limit never fires. Warn rather than fail: everything
else about a compiled solve still works, and the KKT pass budget still bounds it.
"""
function frozen_clock()
    @warn """
    This Reactant backend cannot run a host callback, so the elapsed time inside a compiled \
    solve stays frozen at its compilation-time value and `time_limit` will not be enforced. \
    Loading CUDA.jl lifts this on the CUDA backend.""" maxlog = 1
    return time()
end

"""
    CoolPDLP.current_time()

Read the host clock from inside a compiled program.

Tracing `Base.time()` would freeze its trace-time value into the compiled program as a
constant, so the elapsed time would never advance and the time limit could never fire.
`Reactant.Ops.julia_callback` emits a `stablehlo.custom_call` back into Julia instead, which is
re-evaluated at every iteration of the compiled loop.

`has_side_effect = true` marks that call impure, so the compiler may not hoist it out of the
loop, share it across iterations or drop it when its result looks unused — each of which would
put the frozen clock back. With `has_side_effect = false` the emitted call is pure and all
three become legal.

The single-element reduction that turns the callback's output back into a scalar costs nothing:
it compiles down to a `stablehlo.reshape`.
"""
@reactant_overlay function CoolPDLP.current_time()
    host_callbacks_supported() || return frozen_clock()
    out = Reactant.Ops.julia_callback(
        write_time!, ((Float64, (1,)),); has_side_effect = true
    )
    return sum(out)
end

end
