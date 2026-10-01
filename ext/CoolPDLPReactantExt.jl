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

# Reactant's own `mul!` overlays lower the product to a dense `dot_general`, which a sparse
# format cannot serve, so these send it back to the format's kernels.
@reactant_overlay function LinearAlgebra.mul!(
        c::AbstractVector, A::CoolPDLP.GPUSparseMatrix, b::AbstractVector, α::Number, β::Number
    )
    return native_mul!(c, A, b, α, β)
end

@reactant_overlay function LinearAlgebra.mul!(
        c::AbstractMatrix, A::CoolPDLP.GPUSparseMatrix, b::AbstractMatrix, α::Number, β::Number
    )
    return spmm_error()
end

@reactant_overlay function LinearAlgebra.mul!(
        c::AbstractVector, A::CoolPDLP.GPUSparseMatrix, b::AbstractVector
    )
    return native_mul!(c, A, b, true, false)
end

@reactant_overlay function LinearAlgebra.mul!(
        c::AbstractMatrix, A::CoolPDLP.GPUSparseMatrix, b::AbstractMatrix
    )
    return spmm_error()
end

"""
    spmm_error()

Refuse a product of a sparse format with a matrix, which Reactant can currently miscompile.
"""
function spmm_error()
    return error(
        "Reactant cannot yet compile the product of a CoolPDLP sparse format with a matrix " *
            "(as in a batched solve): on CUDA, XLA may silently reorder the 2-D output of a " *
            "kernel. See https://github.com/EnzymeAD/Reactant.jl/issues/3269."
    )
end

"""
    KernelArray

Array whose kernels Reactant keeps compiling, even when they are launched by native dispatch.
"""
struct KernelArray{T, N, A <: AbstractArray{T, N}} <: DenseArray{T, N}
    data::A
end

Base.size(x::KernelArray) = size(x.data)
Base.IndexStyle(::Type{<:KernelArray}) = Base.IndexLinear()
Base.@propagate_inbounds Base.getindex(x::KernelArray, i::Int) = x.data[i]
Base.@propagate_inbounds Base.setindex!(x::KernelArray, v, i::Int) = (x.data[i] = v)
# an `Atomix.@atomic` update takes a pointer to the element
Base.pointer(x::KernelArray) = pointer(x.data)
Base.pointer(x::KernelArray, i::Integer) = pointer(x.data, i)

Adapt.adapt_structure(to, x::KernelArray) = KernelArray(adapt(to, x.data))

"""
    Wrap

Adaptor putting every traced array in a [`KernelArray`](@ref).
"""
struct Wrap end

Adapt.adapt_structure(::Wrap, x::TracedRArray) = KernelArray(x)

"""
    NativeLaunch

Backend of a [`KernelArray`](@ref), which hands its kernel launches back to Reactant.

[`native_mul!`](@ref) reaches the format's own `mul!` through `Reactant.call_with_native`, because
Reactant resolves every call — `invoke` included — through its overlay table, and would otherwise
catch `mul!` again. Native dispatch then sends the launch to the plain `KernelAbstractions`
method, which nests a `Reactant.@jit` and fails on arguments that are already traced.
"""
struct NativeLaunch{B <: KernelAbstractions.GPU} <: KernelAbstractions.GPU
    backend::B
end

KernelAbstractions.get_backend(x::KernelArray) = NativeLaunch(get_backend(x.data))

function (kernel::KernelAbstractions.Kernel{NativeLaunch{B}, W, N, F})(
        args...; ndrange = nothing, workgroupsize = nothing
    ) where {B, W, N, F}
    (; backend) = kernel.backend
    inner = KernelAbstractions.Kernel{B, W, N, F}(backend, kernel.f)
    return Reactant.call_with_reactant(
        Reactant.ka_with_reactant, ndrange, workgroupsize, inner, args...
    )
end

"""
    scale!!(c, β)

Multiply `c` by `β` in place, so that its product can be run with a static `β = true`.

A format scales its destination itself, but only after branching on `iszero(β)` and `isone(β)`,
which a traced `β` cannot answer, and through a `fill!` or a broadcast that a
[`KernelArray`](@ref) would serve one element at a time.
"""
function scale!!(c::AbstractArray, β::Number)
    if !(β isa TracedRNumber) && iszero(β)
        fill!(c, false)
    elseif !(β isa TracedRNumber) && isone(β)
        c
    else
        c .= β .* c
    end
    return c
end

"""
    native_mul!(c, A, b, α, β)

Run the ordinary `mul!` of `A`, with its kernels launched by Reactant.
"""
function native_mul!(c::AbstractVector, A, b::AbstractVector, α::Number, β::Number)
    scale!!(c, β)
    Reactant.call_with_native(
        LinearAlgebra.mul!, adapt(Wrap(), c), adapt(Wrap(), A), adapt(Wrap(), b), α, true
    )
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
